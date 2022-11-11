import math
import copy
from torch import nn
import torch.nn.functional as F
import torch
from inspect import isfunction
from functools import partial

from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms, utils

from einops import rearrange

from PIL import Image
from torch import linalg as LA

try:
    from apex import amp

    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

from utils_metrics.utils import exists, default, cycle, cycle_cat, num_to_groups, loss_backwards, \
                         extract, noise_like, cosine_beta_schedule, patchify, unpatchify, random_masking
from utils_metrics.fid_score import calculate_fid_given_samples
from models.Unet import Unet, EMA

LOG = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True

class GaussianDiffusion(nn.Module):
    def __init__(self,defade_fn,cfg):

        super().__init__()
        self.cfg = cfg
        LOG.setLevel(os.environ.get("LOGLEVEL", self.cfg.trainer.log_level))
        self.channels = self.cfg.trainer.diffusion.channels
        self.image_size = self.cfg.trainer.diffusion.image_size
        self.denoise_fn = denoise_fn
        self.num_timesteps = self.cfg.trainer.diffusion.timesteps
        self.loss_type = self.cfg.trainer.diffusion.loss_type
        self.device_of_kernel = self.cfg.trainer.device_of_kernel

        self.start_fade_factor = self.cfg.trainer.diffusion.start_fade_factor
        self.fade_routine = self.cfg.trainer.diffusion.fade_routine

        self.fade_factors = self.get_fade_factors()
        self.train_routine = self.cfg.trainer.diffusion.train_routine
        self.sampling_routine= self.cfg.trainer.diffusion.sampling_routine

    def get_fade_factors(self, img_size = 32, mask_size = 4):
        fade_factors = []
        for i in range(self.num_timesteps):
            if self.fade_routine == 'Incremental':
                fade_factors.append(1 - self.start_fade_factor * (i + 1))
            elif self.fade_routine == 'Constant':
                fade_factors.append(1 - self.start_fade_factor)
            elif self.fade_routine == 'Spiral':
                A = np.arange(img_size * img_size).reshape(img_size, img_size)
                spiral = to_spiral(A)
                k = spiral > i
                k = torch.tensor(k).float()
                fade_factors.append(k.cuda())
            elif self.fade_routine == 'Spiral_2':
                A = np.arange(img_size * img_size).reshape(img_size, img_size)
                spiral = to_spiral(A)
                k = spiral > i
                k = torch.tensor(k).float()
                fade_factors.append(k.cuda())
            elif self.fade_routine == 'Mask':
                patched_inputs = patchify(img_tensor.unsqueeze(0))
                patched_inputs_size = torch.ones((img_size**2/mask_size**2, self.channels*mask_size**2))
                img_masked, mask, ids = random_masking(patched_inputs, i/self.num_timesteps)
                new_mask = 1 - mask.unsqueeze(2).repeat(1, 1,patched_inputs.size(2))
                fade_factors.append(unpatchify(new_mask).cuda())



        return fade_factors

    @torch.no_grad()
    def sample(self, batch_size=16, img=None, t=None):

        if t is None:
            t = self.num_timesteps        
        xt = img
        direct_recons = None
        while t:
            step = torch.full((batch_size,), t - 1, dtype=torch.long).cuda()
            x = self.defade_fn(img, step)

            if "Final" in self.train_routine:
                if direct_recons is None:
                    direct_recons = x

                if self.sampling_routine == 'default':
                    for i in range(t - 1):
                        with torch.no_grad():
                            x = self.fade_factors[i] * x

                elif self.sampling_routine == 'x0_step_down':
                    x_times = x
                    x_times_sub_1 = x

                    for i in range(t):
                        with torch.no_grad():
                            x_times_sub_1 = x_times
                            x_times = self.fade_factors[i] * x_times

                    x = img - x_times + x_times_sub_1

                elif self.sampling_routine == 'x0_step_down_spiral_2_fix':
                    x_times = x
                    x_times_sub_1 = x

                    for i in range(t):
                        with torch.no_grad():
                            x_times_sub_1 = x_times
                            x_times = self.fade_factors[i] * x_times + new_mean * (
                                        torch.ones_like(self.fade_factors[i]) - self.fade_factors[i])

                    x = img - x_times + x_times_sub_1

                elif self.sampling_routine == 'x0_step_down_spiral_2_rand':
                    x_times = x
                    x_times_sub_1 = x

                    for i in range(t):
                        new_mean = torch.rand((img.shape[0], 3))
                        new_mean = new_mean.unsqueeze(2).repeat(1, 1, img.shape[2])
                        new_mean = new_mean.unsqueeze(3).repeat(1, 1, 1, img.shape[3]).cuda()

                        with torch.no_grad():
                            x_times_sub_1 = x_times
                            x_times = self.fade_factors[i] * x_times + new_mean * (
                                        torch.ones_like(self.fade_factors[i]) - self.fade_factors[i])

                    x = img - x_times + x_times_sub_1

            elif self.train_routine == 'Step':
                if direct_recons is None:
                    direct_recons = x

            elif self.train_routine == 'Gradient_norm':
                if direct_recons is None:
                    direct_recons = img - x
                x = img - x

            img = x
            t = t - 1

        return xt, direct_recons, img

    @torch.no_grad()
    def all_sample(self, batch_size=16, img=None, t=None, times=None):

        if t is None:
            t = self.num_timesteps
        if times is None:
            times = t

        if self.fade_routine == 'Spiral_2':
            new_mean = torch.rand((img.shape[0], 3))
            new_mean = new_mean.unsqueeze(2).repeat(1, 1, img.shape[2])
            new_mean = new_mean.unsqueeze(3).repeat(1, 1, 1, img.shape[3]).cuda()

        for i in range(t):
            with torch.no_grad():
                if self.fade_routine == 'Spiral_2':
                    img = self.fade_factors[i] * img + new_mean * (
                                torch.ones_like(self.fade_factors[i]) - self.fade_factors[i])
                else:
                    img = self.fade_factors[i] * img


        x0_list = []
        xt_list = []

        while times:
            step = torch.full((batch_size,), times - 1, dtype=torch.long).cuda()
            x = self.defade_fn(img, step)
            x0_list.append(x)

            if "Final" in self.train_routine:
                if self.sampling_routine == 'default':
                    print("Normal")

                    x_times_sub_1 = x
                    for i in range(times - 1):
                        with torch.no_grad():
                            x_times_sub_1 = self.fade_factors[i] * x_times_sub_1

                    x = x_times_sub_1

                elif self.sampling_routine == 'x0_step_down':
                    print("x0_step_down")

                    x_times = x
                    for i in range(times):
                        with torch.no_grad():
                            x_times = self.fade_factors[i] * x_times

                    x_times_sub_1 = x
                    for i in range(times - 1):
                        with torch.no_grad():
                            x_times_sub_1 = self.fade_factors[i] * x_times_sub_1

                    x = img - x_times + x_times_sub_1

                elif self.sampling_routine == 'x0_step_down_spiral_2_fix':
                    x_times = x
                    x_times_sub_1 = x

                    for i in range(times):
                        with torch.no_grad():
                            x_times_sub_1 = x_times
                            x_times = self.fade_factors[i] * x_times + new_mean * (
                                        torch.ones_like(self.fade_factors[i]) - self.fade_factors[i])

                    x = img - x_times + x_times_sub_1

                elif self.sampling_routine == 'x0_step_down_spiral_2_rand':
                    x_times = x
                    x_times_sub_1 = x

                    for i in range(times):
                        new_mean = torch.rand((img.shape[0], 3))
                        new_mean = new_mean.unsqueeze(2).repeat(1, 1, img.shape[2])
                        new_mean = new_mean.unsqueeze(3).repeat(1, 1, 1, img.shape[3]).cuda()

                        with torch.no_grad():
                            x_times_sub_1 = x_times
                            x_times = self.fade_factors[i] * x_times + new_mean * (
                                        torch.ones_like(self.fade_factors[i]) - self.fade_factors[i])

                    x = img + x_times_sub_1 - img #- x_times

                elif self.sampling_routine == 'no_time_embed':
                    x = x
                    for i in range(100):
                        with torch.no_grad():
                            x = self.fade_factors[i] * x

            elif self.train_routine == 'Gradient_norm':
                x = img - 0.1 * x
                for i in range(10):
                    with torch.no_grad():
                        x = self.fade_factors[i] * x

            img = x
            xt_list.append(img)
            times = times - 1

        return x0_list, xt_list

    def q_sample(self, x_start, t):

        if self.fade_routine == 'Spiral':
            choose_fade = []
            for img_index in range(t.shape[0]):
                choose_fade.append(x_start[img_index,:] * self.fade_factors[t[img_index]] )

            choose_fade = torch.stack(choose_fade)
            return choose_fade

        elif self.fade_routine == 'Spiral_2':

            choose_fade = []
            for img_index in range(t.shape[0]):
                new_mean = torch.rand((1, 3))
                new_mean = new_mean.unsqueeze(2).repeat(1, 1, x_start.shape[2])
                new_mean = new_mean.unsqueeze(3).repeat(1, 1, 1, x_start.shape[3]).cuda()

                cf = x_start[img_index,:] * self.fade_factors[t[img_index]] + new_mean * (torch.ones_like(self.fade_factors[t[img_index]]) - self.fade_factors[t[img_index]])
                choose_fade.append(cf[0,:])

            choose_fade = torch.stack(choose_fade)
            return choose_fade

        else:
            max_iters = torch.max(t)
            all_fades = []
            x = x_start
            for i in range(max_iters + 1):
                with torch.no_grad():
                    x = self.fade_factors[i] * x
                    all_fades.append(x)

            all_fades = torch.stack(all_fades)

            choose_fade = []
            for step in range(t.shape[0]):
                if step != -1:
                    choose_fade.append(all_fades[t[step], step])
                else:
                    choose_fade.append(x_start[step])

            choose_fade = torch.stack(choose_fade)
            return choose_fade


    def p_losses(self, x_start, t):
        if self.train_routine == 'Final':
            x_fade = self.q_sample(x_start=x_start, t=t)
            x_recon = self.defade_fn(x_fade, t)

            if self.loss_type == 'l1':
                loss = (x_start - x_recon).abs().mean()
            elif self.loss_type == 'l2':
                loss = F.mse_loss(x_start, x_recon)
            else:
                raise NotImplementedError()

        elif self.train_routine == 'Gradient_norm':
            x_fade = self.q_sample(x_start=x_start, t=t)
            grad_pred = self.defade_fn(x_fade, t)
            gradient = (x_fade - x_start)
            norm = LA.norm(gradient, dim=(1, 2, 3), keepdim=True)
            gradient_norm = gradient / (norm + 1e-5)

            if self.loss_type == 'l1':
                loss = (gradient_norm - grad_pred).abs().mean()
            elif self.loss_type == 'l2':
                loss = F.mse_loss(gradient_norm, grad_pred)
            else:
                raise NotImplementedError()

        # elif self.train_routine == 'Step':
        #     x_fade = self.q_sample(x_start=x_start, t=t)
        #     x_fade_sub = self.q_sample(x_start=x_start, t=t - 1)
        #     x_blur_sub_pred = self.defade_fn(x_fade, t)

        #     if self.loss_type == 'l1':
        #         loss = (x_fade_sub - x_blur_sub_pred).abs().mean()
        #     elif self.loss_type == 'l2':
        #         loss = F.mse_loss(x_fade_sub, x_blur_sub_pred)
        #     else:
        #         raise NotImplementedError()

        return loss

    def forward(self, x, *args, **kwargs):
        b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, t, *args, **kwargs)


class Dataset(data.Dataset):
    def __init__(self, folder, image_size, exts=['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)


class Dataset_Cifar10(data.Dataset):
    def __init__(self, folder, image_size, exts=['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)


class Trainer(object):
    def __init__(self, diffusion_model, cfg, writer):
            diffusion_model,
            folder,
            *,
            ema_decay=0.995,
            image_size=128,
            train_batch_size=32,
            train_lr=2e-5,
            train_num_steps=100000,
            gradient_accumulate_every=2,
            fp16=False,
            step_start_ema=2000,
            update_ema_every=10,
            save_and_sample_every=1000,
            results_folder='./results',
            load_path=None,
            dataset=None
    ):
        super().__init__()
        self.cfg = cfg
        LOG.setLevel(os.environ.get("LOGLEVEL", self.cfg.trainer.log_level))
        self.folder = self.cfg.dataset.folder

        self.ema_decay = self.cfg.trainer.ema_decay
        self.image_size = self.cfg.trainer.image_size
        self.batch_size = self.cfg.trainer.train_batch_size
        self.train_lr = self.cfg.trainer.lr
        self.train_num_steps = self.cfg.trainer.train_num_steps
        self.gradient_accumulate_every = self.cfg.trainer.gradient_accumulate_every
        self.fp16 = self.cfg.trainer.fp16
        self.step_start_ema = self.cfg.trainer.step_start_ema
        self.update_ema_every = self.cfg.trainer.update_ema_every
        self.save_and_sample_every = self.cfg.trainer.save_and_sample_every
        self.results_folder = self.cfg.trainer.results_folder
        self.load_path = self.cfg.trainer.checkpointpath
        self.dataset = self.cfg.dataset.name
        self.mode = self.cfg.dataset.mode
        self.writer = writer
        self.gpu = self.cfg.trainer.gpu

        self.model = diffusion_model
        self.ema = EMA(self.ema_decay)
        self.ema_model = copy.deepcopy(self.model)


        if dataset == 'cifar10':
            self.ds = Dataset_Cifar10(self.folder, self.image_size)
        else:
            self.ds = Dataset(self.folder, self.image_size)
        self.sampler = DistributedSampler(self.ds, num_replicas=self.cfg.trainer.world_size, seed = self.cfg.trainer.seed,  rank=self.cfg.trainer.rank)
        self.dl = cycle(data.DataLoader(self.ds, batch_size=self.train_batch_size, sampler=self.sampler, pin_memory=True))
        self.opt = Adam(self.model.parameters(), lr=self.train_lr)

        self.step = 0

        assert not self.fp16 or self.fp16 and APEX_AVAILABLE, 'Apex must be installed in order for mixed precision training to be turned on'

        if self.fp16:
            (self.model, self.ema_model), self.opt = amp.initialize([self.model, self.ema_model], self.opt,
                                                                    opt_level='O1')

        self.results_folder = Path(self.results_folder)
        self.results_folder.mkdir(exist_ok=True)

        self.reset_parameters()

        if self.load_path is not None:
            self.load(self.load_path)

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        torch.save(data, str(self.results_folder / f'model.pt'))

    def load(self, load_path):
        print("Loading : ", load_path)
        data = torch.load(load_path)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    def add_title(self, path, title):
        import cv2
        import numpy as np

        img1 = cv2.imread(path)
        black = [0, 0, 0]
        constant = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
        height = 20
        violet = np.zeros((height, constant.shape[1], 3), np.uint8)
        violet[:] = (255, 0, 180)

        vcat = cv2.vconcat((violet, constant))

        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(vcat, str(title), (violet.shape[1] // 2, height - 2), font, 0.5, (0, 0, 0), 1, 0)
        cv2.imwrite(path, vcat)


    def save_image_and_log(self,og_img, all_images, direct_recons, xt, milestone, mode = 'Train'):
        try:
            og_img = (og_img + 1) * 0.5
            utils.save_image(og_img, str(self.results_folder / f'{mode}/sample-og-{milestone}.png'), nrow=6)

            all_images = (all_images + 1) * 0.5
            utils.save_image(all_images, str(self.results_folder / f'{mode}/sample-recon-{milestone}.png'), nrow = 6)

            direct_recons = (direct_recons + 1) * 0.5
            utils.save_image(direct_recons, str(self.results_folder / f'{mode}/sample-direct_recons-{milestone}.png'), nrow=6)

            xt = (xt + 1) * 0.5
            utils.save_image(xt, str(self.results_folder / f'{mode}/sample-xt-{milestone}.png'),nrow=6)
            time.sleep(1)
            
            LOG.info("Logging image")
            wandb.log({f"{mode}_img/target_sample_{milestone}": wandb.Image(str(self.results_folder / f'{mode}/sample-og-{milestone}.png'))})
            # wandb.log({f"{mode}_img/noise_sample_{milestone}": wandb.Image(str(self.results_folder / f'{mode}/sample-og2-{milestone}.png'))})
            wandb.log({f"{mode}_img/full_reconstruction{milestone}": wandb.Image(str(self.results_folder / f'{mode}/sample-recon-{milestone}.png'))})
            wandb.log({f"{mode}_img/direct_reconstruction_{milestone}": wandb.Image(str(self.results_folder / f'{mode}/sample-direct_recons-{milestone}.png'))})
            wandb.log({f"{mode}_img/xt_{milestone}": wandb.Image(str(self.results_folder / f'{mode}/sample-xt-{milestone}.png'))})
            # wandb.log({f"{mode}_img/noise_reconstruction_{milestone}": wandb.Image(str(self.results_folder / f'{mode}/noise_image_recons-{milestone}.png'))})
        except:
            LOG.error(f"Issue when logging images for {milestone}")
    def train(self):
        backwards = partial(loss_backwards, self.fp16)

        acc_loss = 0
        while self.step < self.train_num_steps:
            u_loss = 0
            for i in range(self.gradient_accumulate_every):
                data = next(self.dl).cuda()
                loss = self.model(data)
                if self.step % 100 == 0:
                    if self.cfg.trainer.platform == 'slurm':
                        print(f'{self.step}: {loss.item()}')
                    LOG.info(f'{self.step}: {loss.item()}')
                if self.writer is not None:
                     self.writer.add_scalar("Train/Loss", loss.item(), self.step)
                u_loss += loss.item()
                backwards(loss / self.gradient_accumulate_every, self.opt)

            acc_loss = acc_loss + (u_loss / self.gradient_accumulate_every)

            self.opt.step()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                if self.cfg.trainer.gpu == 0:
                    milestone = self.step // self.save_and_sample_every
                    batches = self.batch_size
                    og_img = next(self.dl).cuda()

                    xt, direct_recons, all_images = self.ema_model.sample(batch_size=batches, faded_recon_sample=og_img)
                    self.save_image_and_log(og_img, all_images, direct_recons, xt, milestone, mode = 'Train')

                    acc_loss = acc_loss / (self.save_and_sample_every + 1)
                    LOG.info(f'Mean of last {self.step}: {acc_loss}')
                    

                    self.save()
                acc_loss = 0
                dist.barrier()
            self.step += 1

        print('training completed')

    def test_from_data(self, extra_path, s_times=None):
        batches = self.batch_size
        og_img = next(self.dl).cuda()
        x0_list, xt_list = self.ema_model.all_sample(batch_size=batches, faded_recon_sample=og_img, times=s_times)

        og_img = (og_img + 1) * 0.5
        utils.save_image(og_img, str(self.results_folder / f'og-{extra_path}.png'), nrow=6)

        import imageio
        frames_t = []
        frames_0 = []

        for i in range(len(x0_list)):
            print(i)

            x_0 = x0_list[i]
            x_0 = (x_0 + 1) * 0.5
            utils.save_image(x_0, str(self.results_folder / f'sample-{i}-{extra_path}-x0.png'), nrow=6)
            self.add_title(str(self.results_folder / f'sample-{i}-{extra_path}-x0.png'), str(i))
            frames_0.append(imageio.imread(str(self.results_folder / f'sample-{i}-{extra_path}-x0.png')))

            x_t = xt_list[i]
            all_images = (x_t + 1) * 0.5
            utils.save_image(all_images, str(self.results_folder / f'sample-{i}-{extra_path}-xt.png'), nrow=6)
            self.add_title(str(self.results_folder / f'sample-{i}-{extra_path}-xt.png'), str(i))
            frames_t.append(imageio.imread(str(self.results_folder / f'sample-{i}-{extra_path}-xt.png')))

        imageio.mimsave(str(self.results_folder / f'Gif-{extra_path}-x0.gif'), frames_0)
        imageio.mimsave(str(self.results_folder / f'Gif-{extra_path}-xt.gif'), frames_t)

    def test_with_mixup(self, extra_path):
        batches = self.batch_size
        og_img_1 = next(self.dl).cuda()
        og_img_2 = next(self.dl).cuda()
        og_img = (og_img_1 + og_img_2) / 2

        x0_list, xt_list = self.ema_model.all_sample(batch_size=batches, faded_recon_sample=og_img)

        og_img_1 = (og_img_1 + 1) * 0.5
        utils.save_image(og_img_1, str(self.results_folder / f'og1-{extra_path}.png'), nrow=6)

        og_img_2 = (og_img_2 + 1) * 0.5
        utils.save_image(og_img_2, str(self.results_folder / f'og2-{extra_path}.png'), nrow=6)

        og_img = (og_img + 1) * 0.5
        utils.save_image(og_img, str(self.results_folder / f'og-{extra_path}.png'), nrow=6)

        frames_t = []
        frames_0 = []

        for i in range(len(x0_list)):
            print(i)
            x_0 = x0_list[i]
            x_0 = (x_0 + 1) * 0.5
            utils.save_image(x_0, str(self.results_folder / f'sample-{i}-{extra_path}-x0.png'), nrow=6)
            self.add_title(str(self.results_folder / f'sample-{i}-{extra_path}-x0.png'), str(i))
            frames_0.append(Image.open(str(self.results_folder / f'sample-{i}-{extra_path}-x0.png')))

            x_t = xt_list[i]
            all_images = (x_t + 1) * 0.5
            utils.save_image(all_images, str(self.results_folder / f'sample-{i}-{extra_path}-xt.png'), nrow=6)
            self.add_title(str(self.results_folder / f'sample-{i}-{extra_path}-xt.png'), str(i))
            frames_t.append(Image.open(str(self.results_folder / f'sample-{i}-{extra_path}-xt.png')))

        frame_one = frames_0[0]
        frame_one.save(str(self.results_folder / f'Gif-{extra_path}-x0.gif'), format="GIF", append_images=frames_0,
                       save_all=True, duration=100, loop=0)

        frame_one = frames_t[0]
        frame_one.save(str(self.results_folder / f'Gif-{extra_path}-xt.gif'), format="GIF", append_images=frames_t,
                       save_all=True, duration=100, loop=0)

    def test_from_random(self, extra_path):
        batches = self.batch_size
        og_img = next(self.dl).cuda()
        og_img = og_img * 0.9  # torch.randn_like(og_img) + 0.1
        x0_list, xt_list = self.ema_model.all_sample(batch_size=batches, faded_recon_sample=og_img)

        og_img = (og_img + 1) * 0.5
        utils.save_image(og_img, str(self.results_folder / f'og-{extra_path}.png'), nrow=6)

        frames_t_names = []
        frames_0_names = []

        for i in range(len(x0_list)):
            print(i)

            x_0 = x0_list[i]
            x_0 = (x_0 + 1) * 0.5
            utils.save_image(x_0, str(self.results_folder / f'sample-{i}-{extra_path}-x0.png'), nrow=6)
            self.add_title(str(self.results_folder / f'sample-{i}-{extra_path}-x0.png'), str(i))
            frames_0_names.append(str(self.results_folder / f'sample-{i}-{extra_path}-x0.png'))

            x_t = xt_list[i]
            all_images = (x_t + 1) * 0.5
            utils.save_image(all_images, str(self.results_folder / f'sample-{i}-{extra_path}-xt.png'), nrow=6)
            self.add_title(str(self.results_folder / f'sample-{i}-{extra_path}-xt.png'), str(i))
            frames_t_names.append(str(self.results_folder / f'sample-{i}-{extra_path}-xt.png'))

        import imageio
        frames_0 = []
        frames_t = []
        for i in range(len(x0_list)):
            print(i)
            frames_0.append(imageio.imread(frames_0_names[i]))
            frames_t.append(imageio.imread(frames_t_names[i]))

        imageio.mimsave(str(self.results_folder / f'Gif-{extra_path}-x0.gif'), frames_0)
        imageio.mimsave(str(self.results_folder / f'Gif-{extra_path}-xt.gif'), frames_t)

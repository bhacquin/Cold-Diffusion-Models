import math
import time 
import wandb
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms, utils
import torchvision
from PIL import Image
import torch.distributed as dist

import numpy as np
from tqdm import tqdm
from einops import rearrange

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler

import torchgeometry as tgm
import glob
import os
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch import linalg as LA
from sklearn.mixture import GaussianMixture
import logging
try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False


from utils_metrics.utils import exists, default, cycle, cycle_cat, num_to_groups, loss_backwards, extract, noise_like, cosine_beta_schedule
from models.Unet import Unet, EMA


LOG = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True


class GaussianDiffusion(nn.Module):
    def __init__(self, denoise_fn, cfg):

        super().__init__()
        self.cfg = cfg
        LOG.setLevel(os.environ.get("LOGLEVEL", self.cfg.trainer.log_level))
        self.channels = self.cfg.trainer.diffusion.channels
        self.image_size = self.cfg.trainer.diffusion.image_size
        self.denoise_fn = denoise_fn
        self.num_timesteps = self.cfg.trainer.diffusion.timesteps
        self.loss_type = self.cfg.trainer.diffusion.loss_type
        self.train_routine = self.cfg.trainer.diffusion.train_routine
        self.sampling_routine = self.cfg.trainer.diffusion.sampling_routine
        self.discrete= self.cfg.trainer.diffusion.discrete
        try:
            self.gpu = self.cfg.trainer.gpu 
        except:
            self.gpu = 0

        betas = cosine_beta_schedule(self.num_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(torch.from_numpy(alphas), axis=0)

        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

    @torch.no_grad()
    def sample(self, batch_size = 16, img=None, t=None):

        self.denoise_fn.eval()
        if t == None:
            t = self.num_timesteps

        xt = img
        direct_recons = None

        while (t):
            step = torch.full((batch_size,), t - 1, dtype=torch.long).cuda()
            x1_bar = self.denoise_fn(img, step)
            x2_bar = self.get_x2_bar_from_xt(x1_bar, img, step)

            if direct_recons is None:
                direct_recons = x1_bar

            xt_bar = x1_bar
            if t != 0:
                xt_bar = self.q_sample(x_start=xt_bar, x_end=x2_bar, t=step)

            xt_sub1_bar = x1_bar
            if t - 1 != 0:
                step2 = torch.full((batch_size,), t - 2, dtype=torch.long).cuda()
                xt_sub1_bar = self.q_sample(x_start=xt_sub1_bar, x_end=x2_bar, t=step2)

            x = img - xt_bar + xt_sub1_bar
            img = x
            t = t - 1

        self.denoise_fn.train()

        return xt, direct_recons, img

    def get_x2_bar_from_xt(self, x1_bar, xt, t):
        return (
                (xt - extract(self.sqrt_alphas_cumprod, t, x1_bar.shape) * x1_bar) /
                extract(self.sqrt_one_minus_alphas_cumprod, t, x1_bar.shape)
        )

    @torch.no_grad()
    def gen_sample(self, batch_size=16, img=None, t=None, number_of_loop = 1):
        # LOG.info(f"Sampling routine : {self.sampling_routine}")
        self.denoise_fn.eval()
        
        if t == None:
            t = self.num_timesteps
            t_original = self.num_timesteps
        else:
            t_original = t

        noise = img
        direct_recons = None
        first_noisification = True

        for k in range(number_of_loop):
            LOG.info(f"Loop number : {k}")
            t = t_original
            if self.sampling_routine == 'ddim':
                if self.cfg.generator.model.add_gaussian:
                    step = torch.full((batch_size,), t - 1, dtype=torch.long, device=img.device)
                    noise_gaussian = torch.randn_like(img)
                    img = self.q_sample(x_start=img, x_end=noise_gaussian, t=step)
                while (t):
                    step = torch.full((batch_size,), t - 1, dtype=torch.long, device=img.device)
                    x1_bar = self.denoise_fn(img, step)
                    x2_bar = self.get_x2_bar_from_xt(x1_bar, img, step)
                    if direct_recons == None:
                        direct_recons = x1_bar

                    xt_bar = x1_bar
                    if t != 0:
                        xt_bar = self.q_sample(x_start=xt_bar, x_end=x2_bar, t=step)

                    xt_sub1_bar = x1_bar
                    if t - 1 != 0:
                        step2 = torch.full((batch_size,), t - 2, dtype=torch.long, device=img.device)
                        xt_sub1_bar = self.q_sample(x_start=xt_sub1_bar, x_end=x2_bar, t=step2)

                    x = img - xt_bar + xt_sub1_bar
                    img = x
                    t = t - 1
                    if first_noisification:
                        x_t_minus_1 = img
                        first_noisification = False

            elif self.sampling_routine == 'x0_step_down':
                if self.cfg.generator.model.add_gaussian:
                    step = torch.full((batch_size,), t - 1, dtype=torch.long, device=img.device)
                    noise_gaussian = torch.randn_like(img)
                    img = self.q_sample(x_start=img, x_end=noise_gaussian, t=step)
                while (t):
                    step = torch.full((batch_size,), t - 1, dtype=torch.long, device=img.device)
                    x1_bar = self.denoise_fn(img, step)
                    x2_bar = torch.randn_like(img)
                    if direct_recons == None:
                        direct_recons = x1_bar

                    xt_bar = x1_bar
                    if t != 0:
                        xt_bar = self.q_sample(x_start=xt_bar, x_end=x2_bar, t=step)

                    xt_sub1_bar = x1_bar
                    if t - 1 != 0:
                        step2 = torch.full((batch_size,), t - 2, dtype=torch.long, device=img.device)
                        xt_sub1_bar = self.q_sample(x_start=xt_sub1_bar, x_end=x2_bar, t=step2)

                    x = img - xt_bar + xt_sub1_bar
                    img = x
                    t = t - 1
                    if first_noisification:
                        x_t_minus_1 = img
                        first_noisification = False
            
            elif self.sampling_routine == 'classic': 
                while (t):
                    step = torch.full((batch_size,), t - 1, dtype=torch.long, device=img.device)
                    x1_bar = self.denoise_fn(img, step)
                    x2_bar = torch.randn_like(img)

                    if direct_recons == None:
                        direct_recons = x1_bar

                    xt_bar = x1_bar
                    if t != 0:
                        # xt_bar = self.q_sample(x_start=xt_bar, x_end=x2_bar, t=step)
                        xt_bar = xt_bar * (1 - ((t-1) / self.num_timesteps)) + ((t-1) / self.num_timesteps) * x2_bar

                    xt_sub1_bar = x1_bar
                    if t - 1 != 0:
                        step2 = torch.full((batch_size,), t - 2, dtype=torch.long, device=img.device)
                        xt_sub1_bar = self.q_sample(x_start=xt_sub1_bar, x_end=x2_bar, t=step2)

                    # x = img - xt_bar + xt_sub1_bar
                    # img = x
                    img = xt_sub1_bar
                    t = t - 1
                    if first_noisification:
                        x_t_minus_1 = img
                        first_noisification = False

        return noise, direct_recons, img, x_t_minus_1


    @torch.no_grad()
    def forward_and_backward(self, batch_size=16, img=None, t=None, times=None, eval=True):

        self.denoise_fn.eval()

        if t == None:
            t = self.num_timesteps

        Forward = []
        Forward.append(img)

        noise = torch.randn_like(img)

        for i in range(t):
            with torch.no_grad():
                step = torch.full((batch_size,), i, dtype=torch.long, device=img.device)
                n_img = self.q_sample(x_start=img, x_end=noise, t=step)
                Forward.append(n_img)

        Backward = []
        img = n_img
        while (t):
            step = torch.full((batch_size,), t - 1, dtype=torch.long, device=img.device)
            x1_bar = self.denoise_fn(img, step)
            x2_bar = noise #self.get_x2_bar_from_xt(x1_bar, img, step)

            Backward.append(img)

            xt_bar = x1_bar
            if t != 0:
                xt_bar = self.q_sample(x_start=xt_bar, x_end=x2_bar, t=step)

            xt_sub1_bar = x1_bar
            if t - 1 != 0:
                step2 = torch.full((batch_size,), t - 2, dtype=torch.long, device=img.device)
                xt_sub1_bar = self.q_sample(x_start=xt_sub1_bar, x_end=x2_bar, t=step2)

            x = img - xt_bar + xt_sub1_bar
            img = x
            t = t - 1

        return Forward, Backward, img


    @torch.no_grad()
    def all_sample(self, batch_size=16, img=None, t=None, times=None, eval=True):

        if eval:
            self.denoise_fn.eval()

        if t == None:
            t = self.num_timesteps

        X1_0s, X2_0s, X_ts = [], [], []
        while (t):

            step = torch.full((batch_size,), t - 1, dtype=torch.long).cuda()
            x1_bar = self.denoise_fn(img, step)
            x2_bar = self.get_x2_bar_from_xt(x1_bar, img, step)


            X1_0s.append(x1_bar.detach().cpu())
            X2_0s.append(x2_bar.detach().cpu())
            X_ts.append(img.detach().cpu())

            xt_bar = x1_bar
            if t != 0:
                xt_bar = self.q_sample(x_start=xt_bar, x_end=x2_bar, t=step)

            xt_sub1_bar = x1_bar
            if t - 1 != 0:
                step2 = torch.full((batch_size,), t - 2, dtype=torch.long).cuda()
                xt_sub1_bar = self.q_sample(x_start=xt_sub1_bar, x_end=x2_bar, t=step2)

            x = img - xt_bar + xt_sub1_bar
            img = x
            t = t - 1

        return X1_0s, X2_0s, X_ts

    def q_sample(self, x_start, x_end, t):
        # simply use the alphas to interpolate
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_end
        )

    def p_losses(self, x_start, x_end, t):
        b, c, h, w = x_start.shape
        if self.train_routine == 'Final':
            x_mix = self.q_sample(x_start=x_start, x_end=x_end, t=t)
            x_recon = self.denoise_fn(x_mix, t)
            if self.loss_type == 'l1':
                loss = (x_start - x_recon).abs().mean()
            elif self.loss_type == 'l2':
                loss = F.mse_loss(x_start, x_recon)
            else:
                raise NotImplementedError()

        return loss

    def forward(self, x1, x2, *args, **kwargs):
        b, c, h, w, device, img_size, = *x1.shape, x1.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x1, x2, t, *args, **kwargs)

class Dataset_Aug1(data.Dataset):
    def __init__(self, folder, image_size, exts = ['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.transform = transforms.Compose([
            transforms.Resize((int(image_size*1.12), int(image_size*1.12))),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = img.convert('RGB')
        return self.transform(img)

class Dataset(data.Dataset):
    def __init__(self, folder, image_size, exts=['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.transform = transforms.Compose([
            transforms.Resize((int(image_size*1.12), int(image_size*1.12))),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = img.convert('RGB')
        return self.transform(img)
# trainer class
import os
import errno
def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

from collections import OrderedDict
def remove_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        name = k.replace('.module', '')  # remove `module.`
        new_state_dict[name] = v

    return new_state_dict

def adjust_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        name = k.replace('denoise_fn.module', 'module.denoise_fn')  # remove `module.`
        new_state_dict[name] = v

    return new_state_dict

class Trainer(object):
    def __init__(self, diffusion_model, cfg, writer):
    
        super().__init__()
        self.cfg = cfg
        # LOG.setLevel(os.environ.get("LOGLEVEL", self.cfg.trainer.log_level))
        self.folder = self.cfg.dataset.folder
        # self.folder_test = self.cfg.dataset.folder_test
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
 
        if self.mode == 'train':
            print(self.dataset, "DA used")
            self.ds = Dataset_Aug1(self.folder, self.image_size)
        else:
            print(self.dataset)
            self.ds = Dataset(self.folder, self.image_size)
            
        self.sampler = DistributedSampler(self.ds, num_replicas=self.cfg.trainer.world_size, seed = self.cfg.trainer.seed,  rank=self.cfg.trainer.rank)
        self.dl = cycle(data.DataLoader(self.ds, batch_size = self.batch_size, sampler=self.sampler, pin_memory=True, num_workers=16, drop_last=True))

        self.opt = Adam(diffusion_model.parameters(), lr=self.train_lr)
        self.step = 0

        self.results_folder = Path(self.results_folder)
        self.results_folder.mkdir(exist_ok = True)

        self.reset_parameters()

        if self.load_path != None:
            self.load(self.load_path)


    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, itrs=None):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        if itrs is None:
            torch.save(data, str(self.results_folder / f'model.pt'))
        else:
            torch.save(data, str(self.results_folder / f'model_{itrs}.pt'))

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

        # --- Here I am creating the border---
        black = [0, 0, 0]  # ---Color of the border---
        constant = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
        height = 20
        violet = np.zeros((height, constant.shape[1], 3), np.uint8)
        violet[:] = (255, 0, 180)

        vcat = cv2.vconcat((violet, constant))

        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(vcat, str(title), (violet.shape[1] // 2, height-2), font, 0.5, (0, 0, 0), 1, 0)
        cv2.imwrite(path, vcat)

    def save_image_and_log(self,og_img1, all_images, direct_recons, xt, milestone, mode = 'Train'):
        try:
            og_img1 = (og_img1 + 1) * 0.5
            utils.save_image(og_img1, str(self.results_folder) + f'/sample-og1-{milestone}.png', nrow=6)

            all_images = (all_images + 1) * 0.5
            utils.save_image(all_images, str(self.results_folder) + f'/sample-recon-{milestone}.png', nrow = 6)

            direct_recons = (direct_recons + 1) * 0.5
            utils.save_image(direct_recons, str(self.results_folder) + f'/sample-direct_recons-{milestone}.png', nrow=6)

            xt = (xt + 1) * 0.5
            utils.save_image(xt, str(self.results_folder) + f'/sample-xt-{milestone}.png',nrow=6)

            time.sleep(1)
            
            LOG.info("Logging image")
            wandb.log({f"{mode}_img/target_sample_{milestone}_": wandb.Image(str(self.results_folder) + f'/sample-og1-{milestone}.png')})
            # wandb.log({f"{mode}_img/noise_sample_{milestone}_{time}": wandb.Image(str(self.results_folder / f'{mode}/sample-og2-{milestone}.png'))})
            wandb.log({f"{mode}_img/full_reconstruction{milestone}": wandb.Image(str(self.results_folder) + f'/sample-recon-{milestone}.png')})
            wandb.log({f"{mode}_img/direct_reconstruction_{milestone}": wandb.Image(str(self.results_folder) + f'/sample-direct_recons-{milestone}.png')})
            wandb.log({f"{mode}_img/xt_{milestone}": wandb.Image(str(self.results_folder) + f'/sample-xt-{milestone}.png')})
            # wandb.log({f"{mode}_img/noise_reconstruction_{milestone}_{time}": wandb.Image(str(self.results_folder / f'{mode}/noise_image_recons-{milestone}.png'))})
        except Exception as e:
            LOG.error(f"Issue {e} when logging images for {milestone}")        

    def train(self):

        backwards = partial(loss_backwards, self.fp16)

        acc_loss = 0
        while self.step < self.train_num_steps:
            u_loss = 0
            for i in range(self.gradient_accumulate_every):
                data_1 = next(self.dl)
                data_2 = torch.randn_like(data_1)

                data_1, data_2 = data_1.cuda(), data_2.cuda()
                loss = torch.mean(self.model(data_1, data_2))
                if self.step % 100 == 0:
                    print(f'{self.step}: {loss.item()}')
                u_loss += loss.item()
                backwards(loss / self.gradient_accumulate_every, self.opt)
                if self.writer is not None:
                    self.writer.add_scalar("Train/Loss", loss.item(), self.step)

            acc_loss = acc_loss + (u_loss/self.gradient_accumulate_every)

            self.opt.step()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                dist.barrier()
                if self.cfg.trainer.gpu == 0:
                    milestone = self.step // self.save_and_sample_every
                    batches = self.batch_size

                    data_1 = next(self.dl)
                    data_2 = torch.randn_like(data_1)
                    og_img = data_2.cuda()

                    xt, direct_recons, all_images = self.ema_model.module.sample(batch_size=batches, img=og_img)
                    print(f'direct_recon {direct_recons[0]}')
                    self.save_image_and_log(data_1, all_images, direct_recons, xt, milestone, mode = 'Train')
                    acc_loss = acc_loss/(self.save_and_sample_every+1)
                    print(f'Mean of last {self.step}: {acc_loss}')
                    acc_loss=0

                    self.save()
                    if self.step % (self.save_and_sample_every * 100) == 0:
                        self.save(self.step)
                dist.barrier()
            self.step += 1

        print('training completed')

    def generate(self):      
        if self.cfg.trainer.gpu == 0:
            milestone = 0
            batches = self.batch_size

            data_1 = next(self.dl)
            data_1 = data_1.cuda()
            ### depending on noise:
            if self.cfg.trainer.noise.type == 'gaussian':
                data_2 = torch.randn_like(data_1)
            elif self.cfg.trainer.noise.type == 'blur':
                self.blur_routine = self.cfg.trainer.noise.blur_routine
                self.kernel_std = cfg.trainer.noise.kernel_std
                self.kernel_size = cfg.trainer.noise.kernel_size

                def blur(self, dims, std):
                    return tgm.image.get_gaussian_kernel2d(dims, std)

                @torch.no_grad()
                def get_conv(self, dims, std, mode='circular', requires_grad = False):
                    kernel = self.blur(dims, std)
                    conv = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=dims, padding=int((dims[0]-1)/2), padding_mode=mode,
                                    bias=False, groups=self.channels)
                    with torch.no_grad():
                        kernel = torch.unsqueeze(kernel, 0)
                        kernel = torch.unsqueeze(kernel, 0)
                        kernel = kernel.repeat(self.channels, 1, 1, 1)
                        conv.weight = nn.Parameter(kernel)
                    if hasattr(conv, 'weight') and conv.bias is not None:
                        conv.weight.requires_grad_(requires_grad)
                    if hasattr(conv, 'bias') and conv.bias is not None:
                        conv.bias.requires_grad_(requires_grad)
                    return conv

                @torch.no_grad()
                def get_kernels(self):
                    kernels = []
                    for i in range(self.num_timesteps):
                        if self.blur_routine == 'Incremental':
                            kernels.append(self.get_conv((self.kernel_size, self.kernel_size), (self.kernel_std*(i+1), self.kernel_std*(i+1)) ) )
                        elif self.blur_routine == 'Constant':
                            kernels.append(self.get_conv((self.kernel_size, self.kernel_size), (self.kernel_std, self.kernel_std) ) )
                        elif self.blur_routine == 'Constant_reflect':
                            kernels.append(self.get_conv((self.kernel_size, self.kernel_size), (self.kernel_std, self.kernel_std), mode='reflect') )
                        elif self.blur_routine == 'Exponential_reflect':
                            ks = self.kernel_size
                            kstd = np.exp(self.kernel_std * i)
                            kernels.append(self.get_conv((ks, ks), (kstd, kstd), mode='reflect'))
                        elif self.blur_routine == 'Exponential':
                            ks = self.kernel_size
                            kstd = np.exp(self.kernel_std * i)
                            kernels.append(self.get_conv((ks, ks), (kstd, kstd)))
                        elif self.blur_routine == 'Individual_Incremental':
                            ks = 2*i+1
                            kstd = 2*ks
                            kernels.append(self.get_conv((ks, ks), (kstd, kstd)))
                        elif self.blur_routine == 'Special_6_routine':
                            ks = 11
                            kstd = i/100 + 0.35
                            kernels.append(self.get_conv((ks, ks), (kstd, kstd), mode='reflect'))
                    return kernels
                self.gaussian_kernels = nn.ModuleList(self.get_kernels())
                for p in self.gaussian_kernels.parameters():
                    p.requires_grad_(False)

                ### BLURIFICATION OF DATA 1 
                img = data_1
                if self.blur_routine == 'Individual_Incremental':
                    img = self.gaussian_kernels[t - 1](img)
                else:
                    for i in range(t):
                        with torch.no_grad():
                            img = self.gaussian_kernels[i](img)
                data_2 = img.cuda()

            elif self.cfg.trainer.noise.type == 'snow':
                pass
            elif self.cfg.trainer.noise.type == 'masked':
                pass
            elif self.cfg.trainer.noise.type == 'decolor':
                pass

            og_img = data_2.cuda()

            xt, direct_recons, all_images = self.ema_model.module.gen_sample(batch_size=batches, img=og_img)
            self.save_image_and_log(data_1, all_images, direct_recons, xt, milestone, mode = 'Test')
        print('Generation completed')

    def test_from_data(self, extra_path, s_times=None):
        batches = self.batch_size
        og_img = next(self.dl).cuda()
        X_0s, X_ts = self.ema_model.module.all_sample(batch_size=batches, img=og_img, times=s_times)

        og_img = (og_img + 1) * 0.5
        utils.save_image(og_img, str(self.results_folder / f'og-{extra_path}.png'), nrow=6)

        import imageio
        frames_t = []
        frames_0 = []

        for i in range(len(X_0s)):
            print(i)

            x_0 = X_0s[i]
            x_0 = (x_0 + 1) * 0.5
            utils.save_image(x_0, str(self.results_folder / f'sample-{i}-{extra_path}-x0.png'), nrow=6)
            self.add_title(str(self.results_folder / f'sample-{i}-{extra_path}-x0.png'), str(i))
            frames_0.append(imageio.imread(str(self.results_folder / f'sample-{i}-{extra_path}-x0.png')))

            x_t = X_ts[i]
            all_images = (x_t + 1) * 0.5
            utils.save_image(all_images, str(self.results_folder / f'sample-{i}-{extra_path}-xt.png'), nrow=6)
            self.add_title(str(self.results_folder / f'sample-{i}-{extra_path}-xt.png'), str(i))
            frames_t.append(imageio.imread(str(self.results_folder / f'sample-{i}-{extra_path}-xt.png')))

        imageio.mimsave(str(self.results_folder / f'Gif-{extra_path}-x0.gif'), frames_0)
        imageio.mimsave(str(self.results_folder / f'Gif-{extra_path}-xt.gif'), frames_t)

    def sample_and_save_for_fid(self, noise=0):

        # xt_folder = f'{self.results_folder}_xt'
        # create_folder(xt_folder)

        out_folder = f'{self.results_folder}_out'
        create_folder(out_folder)

        # direct_recons_folder = f'{self.results_folder}_dir_recons'
        # create_folder(direct_recons_folder)

        # data_1 = next(self.dl)

        cnt = 0
        bs = 128
        for j in range(int(6400/bs)):

            data_2 = torch.randn(bs, 3, 128, 128)
            og_img = data_2.cuda()
            print(og_img.shape)

            xt, direct_recons, all_images = self.ema_model.module.gen_sample(batch_size=bs, img=og_img)

            for i in range(all_images.shape[0]):
                utils.save_image((all_images[i] + 1) * 0.5,
                                 str(f'{out_folder}/' + f'sample-x0-{cnt}.png'))

                # utils.save_image((xt[i] + 1) * 0.5,
                #                  str(f'{xt_folder}/' + f'sample-x0-{cnt}.png'))
                #
                # utils.save_image((direct_recons[i] + 1) * 0.5,
                #                  str(f'{direct_recons_folder}/' + f'sample-x0-{cnt}.png'))

                cnt += 1

    def paper_showing_diffusion_images_cover_page(self):

        import cv2
        cnt = 0
        # for 200 steps
        # to_show = [2, 4, 8, 16, 32, 64, 128, 192]
        to_show = [2, 4, 16, 64, 128, 256, 384, 448, 480]

        for i in range(5):
            batches = self.batch_size
            og_img = next(self.dl).cuda()
            print(og_img.shape)

            Forward, Backward, final_all = self.ema_model.module.forward_and_backward(batch_size=batches, img=og_img)
            og_img = (og_img + 1) * 0.5
            final_all = (final_all + 1) * 0.5



            for k in range(Forward[0].shape[0]):
                l = []

                utils.save_image(og_img[k], str(self.results_folder / f'og_img_{cnt}.png'), nrow=1)
                start = cv2.imread(f'{self.results_folder}/og_img_{cnt}.png')
                l.append(start)

                for j in range(len(Forward)):
                    x_t = Forward[j][k]
                    x_t = (x_t + 1) * 0.5
                    utils.save_image(x_t, str(self.results_folder / f'temp.png'), nrow=1)
                    x_t = cv2.imread(f'{self.results_folder}/temp.png')
                    if j in to_show:
                        l.append(x_t)

                for j in range(len(Backward)):
                    x_t = Backward[j][k]
                    x_t = (x_t + 1) * 0.5
                    utils.save_image(x_t, str(self.results_folder / f'temp.png'), nrow=1)
                    x_t = cv2.imread(f'{self.results_folder}/temp.png')
                    if (len(Backward) - j) in to_show:
                        l.append(x_t)


                utils.save_image(final_all[k], str(self.results_folder / f'final_{cnt}.png'), nrow=1)
                final = cv2.imread(f'{self.results_folder}/final_{cnt}.png')
                l.append(final)


                im_h = cv2.hconcat(l)
                cv2.imwrite(f'{self.results_folder}/all_{cnt}.png', im_h)

                cnt+=1


    def paper_invert_section_images(self, s_times=None):

        cnt = 0
        for i in range(50):
            batches = self.batch_size
            og_img = next(self.dl).cuda()
            print(og_img.shape)

            X_0s, X_ts = self.ema_model.all_sample(batch_size=batches, img=og_img, times=s_times)
            og_img = (og_img + 1) * 0.5

            for j in range(og_img.shape[0]//3):
                original = og_img[j: j + 1]
                utils.save_image(original, str(self.results_folder / f'original_{cnt}.png'), nrow=3)

                direct_recons = X_0s[0][j: j + 1]
                direct_recons = (direct_recons + 1) * 0.5
                utils.save_image(direct_recons, str(self.results_folder / f'direct_recons_{cnt}.png'), nrow=3)

                sampling_recons = X_0s[-1][j: j + 1]
                sampling_recons = (sampling_recons + 1) * 0.5
                utils.save_image(sampling_recons, str(self.results_folder / f'sampling_recons_{cnt}.png'), nrow=3)

                blurry_image = X_ts[0][j: j + 1]
                blurry_image = (blurry_image + 1) * 0.5
                utils.save_image(blurry_image, str(self.results_folder / f'blurry_image_{cnt}.png'), nrow=3)



                import cv2

                blurry_image = cv2.imread(f'{self.results_folder}/blurry_image_{cnt}.png')
                direct_recons = cv2.imread(f'{self.results_folder}/direct_recons_{cnt}.png')
                sampling_recons = cv2.imread(f'{self.results_folder}/sampling_recons_{cnt}.png')
                original = cv2.imread(f'{self.results_folder}/original_{cnt}.png')

                black = [0, 0, 0]
                blurry_image = cv2.copyMakeBorder(blurry_image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
                direct_recons = cv2.copyMakeBorder(direct_recons, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
                sampling_recons = cv2.copyMakeBorder(sampling_recons, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
                original = cv2.copyMakeBorder(original, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)

                im_h = cv2.hconcat([blurry_image, direct_recons, sampling_recons, original])
                cv2.imwrite(f'{self.results_folder}/all_{cnt}.png', im_h)

                cnt += 1

    def paper_showing_diffusion_images(self, s_times=None):

        import cv2
        cnt = 0
        # to_show = [0, 1, 2, 4, 8, 10, 12, 16, 17, 18, 19, 20]
        # to_show = [0, 1, 2, 4, 8, 16, 20, 24, 32, 36, 38, 39, 40]
        # to_show = [0, 1, 2, 4, 8, 16, 24, 32, 40, 44, 46, 48, 49]
        to_show = [0, 2, 4, 8, 16, 32, 64, 80, 88, 92, 96, 98, 99]

        for i in range(50):
            batches = self.batch_size
            og_img = next(self.dl).cuda()
            print(og_img.shape)

            X_0s, X_ts = self.ema_model.all_sample(batch_size=batches, img=og_img, times=s_times)
            og_img = (og_img + 1) * 0.5

            for k in range(X_ts[0].shape[0]):
                l = []

                for j in range(len(X_ts)):
                    x_t = X_ts[j][k]
                    x_t = (x_t + 1) * 0.5
                    utils.save_image(x_t, str(self.results_folder / f'x_{len(X_ts)-j}_{cnt}.png'), nrow=1)
                    x_t = cv2.imread(f'{self.results_folder}/x_{len(X_ts)-j}_{cnt}.png')
                    if j in to_show:
                        l.append(x_t)


                x_0 = X_0s[-1][k]
                x_0 = (x_0 + 1) * 0.5
                utils.save_image(x_0, str(self.results_folder / f'x_best_{cnt}.png'), nrow=1)
                x_0 = cv2.imread(f'{self.results_folder}/x_best_{cnt}.png')
                l.append(x_0)
                im_h = cv2.hconcat(l)
                cv2.imwrite(f'{self.results_folder}/all_{cnt}.png', im_h)

                cnt+=1


    def paper_showing_diffusion_images_diff(self, s_times=None):

        import cv2
        for i in range(50):
            batches = self.batch_size
            og_img = next(self.dl).cuda()
            print(og_img.shape)

            X_0s_alg2, X_ts_alg2 = self.ema_model.all_sample_both_sample(sampling_routine='x0_step_down', batch_size=batches,
                                                                 img=og_img, times=s_times)
            X_0s_alg1, X_ts_alg1 = self.ema_model.all_sample_both_sample(sampling_routine='default', batch_size=batches,
                                                                 img=og_img, times=s_times)

            og_img = (og_img + 1) * 0.5

            alg2 = []
            alg1 = []

            #to_show = [0, 1, 2, 4, 8, 16, 20, 24, 32, 36, 38, 39, 40]
            to_show = [0, 1, 2, 4, 8, 10, 12, 16, 17, 18, 19, 20]

            for j in range(len(X_ts_alg2)):
                x_t = X_ts_alg2[j][0]
                x_t = (x_t + 1) * 0.5
                utils.save_image(x_t, str(self.results_folder / f'x_alg2_{len(X_ts_alg2)-j}_{i}.png'), nrow=1)
                x_t = cv2.imread(f'{self.results_folder}/x_alg2_{len(X_ts_alg2)-j}_{i}.png')
                if j in to_show:
                    alg2.append(x_t)

                x_t = X_ts_alg1[j][0]
                x_t = (x_t + 1) * 0.5
                utils.save_image(x_t, str(self.results_folder / f'x_alg2_{len(X_ts_alg1) - j}_{i}.png'), nrow=1)
                x_t = cv2.imread(f'{self.results_folder}/x_alg2_{len(X_ts_alg1) - j}_{i}.png')
                if j in to_show:
                    alg1.append(x_t)


            x_0 = X_0s_alg2[-1][0]
            x_0 = (x_0 + 1) * 0.5
            utils.save_image(x_0, str(self.results_folder / f'x_best_alg2_{i}.png'), nrow=1)
            x_0 = cv2.imread(f'{self.results_folder}/x_best_alg2_{i}.png')
            alg2.append(x_0)
            im_h = cv2.hconcat(alg2)
            cv2.imwrite(f'{self.results_folder}/all_alg2_{i}.png', im_h)

            x_0 = X_0s_alg1[-1][0]
            x_0 = (x_0 + 1) * 0.5
            utils.save_image(x_0, str(self.results_folder / f'x_best_alg1_{i}.png'), nrow=1)
            x_0 = cv2.imread(f'{self.results_folder}/x_best_alg1_{i}.png')
            alg1.append(x_0)
            im_h = cv2.hconcat(alg1)
            cv2.imwrite(f'{self.results_folder}/all_alg1_{i}.png', im_h)


    def paper_showing_sampling_diff_images(self, s_times=None):

        import cv2
        cnt = 0
        for i in range(10):
            batches = self.batch_size
            og_img = next(self.dl).cuda()
            print(og_img.shape)

            X_0s_alg2, _ = self.ema_model.all_sample_both_sample(sampling_routine='x0_step_down', batch_size=batches, img=og_img, times=s_times)
            X_0s_alg1, _ = self.ema_model.all_sample_both_sample(sampling_routine='default', batch_size=batches,
                                                                 img=og_img, times=s_times)

            x0_alg1 = (X_0s_alg1[-1] + 1) * 0.5
            x0_alg2 = (X_0s_alg2[-1] + 1) * 0.5
            og_img = (og_img + 1) * 0.5


            for j in range(og_img.shape[0]):
                utils.save_image(x0_alg1[j], str(self.results_folder / f'x0_alg1_{cnt}.png'), nrow=1)
                utils.save_image(x0_alg2[j], str(self.results_folder / f'x0_alg2_{cnt}.png'), nrow=1)
                utils.save_image(og_img[j], str(self.results_folder / f'og_img_{cnt}.png'), nrow=1)



                alg1 = cv2.imread(f'{self.results_folder}/x0_alg1_{cnt}.png')
                alg2 = cv2.imread(f'{self.results_folder}/x0_alg2_{cnt}.png')
                og = cv2.imread(f'{self.results_folder}/og_img_{cnt}.png')


                black = [255, 255, 255]
                alg1 = cv2.copyMakeBorder(alg1, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
                alg2 = cv2.copyMakeBorder(alg2, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
                og = cv2.copyMakeBorder(og, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)

                im_h = cv2.hconcat([og, alg1, alg2])
                cv2.imwrite(f'{self.results_folder}/all_{cnt}.png', im_h)

                cnt += 1

    def sample_as_a_vector_gmm(self, start=0, end=1000, siz=64, ch=3, clusters=10):

        all_samples = []
        flatten = nn.Flatten()
        dataset = self.ds

        print(len(dataset))
        for idx in range(len(dataset)):
            img = dataset[idx]
            img = torch.unsqueeze(img, 0).cuda()
            img = F.interpolate(img, size=siz, mode='bilinear')
            img = flatten(img)
            if idx > start:
                all_samples.append(img[0])
            if idx % 1000 == 0:
                print(idx)
            if end != None:
                if idx == end:
                    print(idx)
                    break

        all_samples = torch.stack(all_samples)
        print(all_samples.shape)

        all_samples = all_samples.cpu().detach().numpy()

        num_samples = 100

        gm = GaussianMixture(n_components=clusters, random_state=0).fit(all_samples)
        og_x, og_y = gm.sample(n_samples=num_samples)
        og_x = og_x.reshape(num_samples, ch, siz, siz)
        og_x = torch.from_numpy(og_x).cuda()
        og_x = og_x.type(torch.cuda.FloatTensor)
        print(og_x.shape)
        og_img = F.interpolate(og_x, size=self.image_size, mode='bilinear')


        X_0s, X_ts = self.ema_model.all_sample(batch_size=1, img=og_img, times=None)

        extra_path = 'vec'
        og_img = (og_img + 1) * 0.5
        utils.save_image(og_img, str(self.results_folder / f'og-{start}-{end}-{siz}-{clusters}-{extra_path}.png'), nrow=6)

        import imageio
        frames_t = []
        frames_0 = []

        for i in range(len(X_0s)):
            print(i)

            x_0 = X_0s[i]
            x_0 = (x_0 + 1) * 0.5
            utils.save_image(x_0, str(self.results_folder / f'sample-{start}-{end}-{siz}-{clusters}-{i}-{extra_path}-x0.png'),
                             nrow=6)
            self.add_title(str(self.results_folder / f'sample-{start}-{end}-{siz}-{clusters}-{i}-{extra_path}-x0.png'), str(i))
            frames_0.append(
                imageio.imread(str(self.results_folder / f'sample-{start}-{end}-{siz}-{clusters}-{i}-{extra_path}-x0.png')))

            x_t = X_ts[i]
            all_images = (x_t + 1) * 0.5
            utils.save_image(all_images,
                             str(self.results_folder / f'sample-{start}-{end}-{siz}-{clusters}-{i}-{extra_path}-xt.png'), nrow=6)
            self.add_title(str(self.results_folder / f'sample-{start}-{end}-{siz}-{clusters}-{i}-{extra_path}-xt.png'), str(i))
            frames_t.append(
                imageio.imread(str(self.results_folder / f'sample-{start}-{end}-{siz}-{clusters}-{i}-{extra_path}-xt.png')))

        imageio.mimsave(str(self.results_folder / f'Gif-{start}-{end}-{siz}-{clusters}-{extra_path}-x0.gif'), frames_0)
        imageio.mimsave(str(self.results_folder / f'Gif-{start}-{end}-{siz}-{clusters}-{extra_path}-xt.gif'), frames_t)


    def sample_as_a_vector_gmm_and_save(self, start=0, end=1000, siz=64, ch=3, clusters=10, n_sample=10000):

        all_samples = []
        flatten = nn.Flatten()
        dataset = self.ds

        print(len(dataset))
        for idx in range(len(dataset)):
            img = dataset[idx]
            img = torch.unsqueeze(img, 0)
            img = F.interpolate(img, size=siz, mode='bilinear')
            img = flatten(img)
            if idx > start:
                all_samples.append(img[0])
            if idx % 1000 == 0:
                print(idx)
            if end != None:
                if idx == end:
                    print(idx)
                    break

        all_samples = torch.stack(all_samples)
        print(all_samples.shape)

        all_samples = all_samples.cpu().detach().numpy()
        gm = GaussianMixture(n_components=clusters, random_state=0).fit(all_samples)

        all_num = n_sample
        num_samples = 10000
        it = int(all_num/num_samples)

        create_folder(f'{self.results_folder}_{siz}_{clusters}/')

        print(f'{self.results_folder}_{siz}_{clusters}/')

        cnt=0
        while(it):
            og_x, og_y = gm.sample(n_samples=num_samples)
            og_x = og_x.reshape(num_samples, ch, siz, siz)
            og_x = torch.from_numpy(og_x).cuda()
            og_x = og_x.type(torch.cuda.FloatTensor)
            print(og_x.shape)
            og_img = F.interpolate(og_x, size=self.image_size, mode='bilinear')
            X_0s, X_ts = self.ema_model.all_sample(batch_size=1, img=og_img, times=None)

            x0s = X_0s[-1]
            for i in range(x0s.shape[0]):
                utils.save_image((x0s[i]+1)*0.5, str(f'{self.results_folder}_{siz}_{clusters}/' + f'sample-x0-{cnt}.png'))
                cnt += 1

            it = it - 1
            print(it)


    def sample_as_a_vector_pytorch_gmm_and_save(self, torch_gmm, start=0, end=1000, siz=64, ch=3, clusters=10, n_sample=10000):


        flatten = nn.Flatten()
        dataset = self.ds
        all_samples = None

        print(len(dataset))
        for idx in range(len(dataset)):
            img = dataset[idx]
            img = torch.unsqueeze(img, 0)
            img = F.interpolate(img, size=siz, mode='bilinear')
            img = flatten(img).cuda()
            if idx > start:
                if all_samples is None:
                    all_samples = img
                else:
                    all_samples = torch.cat((all_samples, img), dim=0)

            if idx % 1000 == 0:
                print(idx)
            if end != None:
                if idx == end:
                    print(idx)
                    break


        #all_samples = torch.stack(all_samples)
        print(all_samples.shape)

        model = torch_gmm(num_components=clusters, trainer_params=dict(gpus=1), covariance_type='full',
                          convergence_tolerance=0.001, batch_size=1000)
        model.fit(all_samples)

        all_num = n_sample
        num_samples = 100
        it = int(all_num / num_samples)

        create_folder(f'{self.results_folder}_{siz}_{clusters}/')
        create_folder(f'{self.results_folder}_gmm_{siz}_{clusters}/')
        create_folder(f'{self.results_folder}_gmm_blur_{siz}_{clusters}/')


        print(f'{self.results_folder}_{siz}_{clusters}/')

        cnt=0
        while(it):
            #og_x, _ = model.sample(n=num_samples)
            og_x = model.sample(num_datapoints=num_samples)
            og_x = og_x.reshape(num_samples, ch, siz, siz)

            og_x = og_x.cuda()
            og_x = og_x.type(torch.cuda.FloatTensor)
            print(og_x.shape)
            og_img = F.interpolate(og_x, size=self.image_size, mode='bilinear')
            X_0s, X_ts = self.ema_model.all_sample(batch_size=og_img.shape[0], img=og_img, times=None)

            x0s = X_0s[-1]
            blurs = X_ts[0]
            for i in range(x0s.shape[0]):
                utils.save_image((x0s[i]+1)*0.5, str(f'{self.results_folder}_{siz}_{clusters}/' + f'sample-x0-{cnt}.png'))

                utils.save_image((og_img[i] + 1) * 0.5,
                                 str(f'{self.results_folder}_gmm_{siz}_{clusters}/' + f'sample-{cnt}.png'))

                utils.save_image((blurs[i] + 1) * 0.5,
                                 str(f'{self.results_folder}_gmm_blur_{siz}_{clusters}/' + f'sample-blur-{cnt}.png'))

                cnt += 1

            it = it - 1
            print(it)

    def sample_as_a_vector_from_blur_pytorch_gmm_and_save(self, torch_gmm, start=0, end=1000, siz=64, ch=3, clusters=10, n_sample=10000):
        flatten = nn.Flatten()
        dataset = self.ds

        print(len(dataset))

        #sample_at = self.ema_model.num_timesteps // 2
        sample_at = self.ema_model.num_timesteps // 2
        all_samples = None

        for idx in range(len(dataset)):
            img = dataset[idx]
            img = torch.unsqueeze(img, 0)
            img = self.ema_model.opt(img.cuda(), t=sample_at)
            img = F.interpolate(img, size=siz, mode='bilinear')
            img = flatten(img).cuda()

            if idx > start:
                if all_samples is None:
                    all_samples = img
                else:
                    all_samples = torch.cat((all_samples, img), dim=0)
                #all_samples.append(img[0])

            if idx % 1000 == 0:
                print(idx)
            if end != None:
                if idx == end:
                    print(idx)
                    break

        # all_samples = torch.stack(all_samples)
        print(all_samples.shape)

        model = torch_gmm(num_components=clusters, trainer_params=dict(gpus=1), covariance_type='full',
                          convergence_tolerance=0.001, batch_size=1000)
        model.fit(all_samples)



        all_num = n_sample
        num_samples = 100
        it = int(all_num/num_samples)

        create_folder(f'{self.results_folder}_{siz}_{clusters}_{sample_at}/')
        create_folder(f'{self.results_folder}_gmm_{siz}_{clusters}_{sample_at}/')
        create_folder(f'{self.results_folder}_gmm_blur_{siz}_{clusters}_{sample_at}/')

        print(f'{self.results_folder}_{siz}_{clusters}/')

        cnt=0
        while(it):
            og_x = model.sample(num_datapoints=num_samples)
            og_x = og_x.reshape(num_samples, ch, siz, siz)

            og_x = og_x.cuda()
            og_x = og_x.type(torch.cuda.FloatTensor)
            print(og_x.shape)
            og_img = F.interpolate(og_x, size=self.image_size, mode='bilinear')
            X_0s, X_ts = self.ema_model.all_sample_from_blur(batch_size=og_img.shape[0], img=og_img, start_times=sample_at)

            x0s = X_0s[-1]
            for i in range(x0s.shape[0]):
                utils.save_image((x0s[i]+1)*0.5, str(f'{self.results_folder}_{siz}_{clusters}_{sample_at}/' + f'sample-x0-{cnt}.png'))

                utils.save_image((og_img[i] + 1) * 0.5,
                                 str(f'{self.results_folder}_gmm_{siz}_{clusters}_{sample_at}/' + f'sample-{cnt}.png'))

                cnt += 1

            it = it - 1



    def sample_from_data_save(self, start=0, end=1000):

        all_samples = []
        dataset = self.ds

        print(len(dataset))
        for idx in range(len(dataset)):
            img = dataset[idx]
            img = torch.unsqueeze(img, 0).cuda()
            if idx > start:
                all_samples.append(img[0])
            if idx % 1000 == 0:
                print(idx)
            if end != None:
                if idx == end:
                    print(idx)
                    break

        all_samples = torch.stack(all_samples)
        create_folder(f'{self.results_folder}/')

        cnt=0
        while(cnt < all_samples.shape[0]):
            og_x = all_samples[cnt: cnt + 1000]
            og_x = og_x.cuda()
            og_x = og_x.type(torch.cuda.FloatTensor)
            og_img = og_x
            print(og_img.shape)
            X_0s, X_ts = self.ema_model.all_sample(batch_size=og_img.shape[0], img=og_img, times=None)

            x0s = X_0s[-1]
            for i in range(x0s.shape[0]):
                utils.save_image( (x0s[i]+1)*0.5, str(f'{self.results_folder}/' + f'sample-x0-{cnt}.png'))
                cnt += 1

    def fid_distance_decrease_from_manifold(self, fid_func, start=0, end=1000):

        #from skimage.metrics import structural_similarity as ssim
        from pytorch_msssim import ssim

        all_samples = []
        dataset = self.ds

        print(len(dataset))
        for idx in range(len(dataset)):
            img = dataset[idx]
            img = torch.unsqueeze(img, 0).cuda()
            if idx > start:
                all_samples.append(img[0])
            if idx % 1000 == 0:
                print(idx)
            if end != None:
                if idx == end:
                    print(idx)
                    break

        all_samples = torch.stack(all_samples)
        # create_folder(f'{self.results_folder}/')
        blurred_samples = None
        original_sample = None
        deblurred_samples = None
        direct_deblurred_samples = None

        sanity_check = 1


        cnt=0
        while(cnt < all_samples.shape[0]):
            og_x = all_samples[cnt: cnt + 100]
            og_x = og_x.cuda()
            og_x = og_x.type(torch.cuda.FloatTensor)
            og_img = og_x
            print(og_img.shape)
            X_0s, X_ts = self.ema_model.all_sample(batch_size=og_img.shape[0], img=og_img, times=None)

            og_img = og_img.to('cpu')
            blurry_imgs = X_ts[0].to('cpu')
            deblurry_imgs = X_0s[-1].to('cpu')
            direct_deblurry_imgs = X_0s[0].to('cpu')

            og_img = og_img.repeat(1, 3 // og_img.shape[1], 1, 1)
            blurry_imgs = blurry_imgs.repeat(1, 3 // blurry_imgs.shape[1], 1, 1)
            deblurry_imgs = deblurry_imgs.repeat(1, 3 // deblurry_imgs.shape[1], 1, 1)
            direct_deblurry_imgs = direct_deblurry_imgs.repeat(1, 3 // direct_deblurry_imgs.shape[1], 1, 1)



            og_img = (og_img + 1) * 0.5
            blurry_imgs = (blurry_imgs + 1) * 0.5
            deblurry_imgs = (deblurry_imgs + 1) * 0.5
            direct_deblurry_imgs = (direct_deblurry_imgs + 1) * 0.5

            if cnt == 0:
                print(og_img.shape)
                print(blurry_imgs.shape)
                print(deblurry_imgs.shape)
                print(direct_deblurry_imgs.shape)

                if sanity_check:
                    folder = './sanity_check/'
                    create_folder(folder)

                    san_imgs = og_img[0: 32]
                    utils.save_image(san_imgs,str(folder + f'sample-og.png'), nrow=6)

                    san_imgs = blurry_imgs[0: 32]
                    utils.save_image(san_imgs, str(folder + f'sample-xt.png'), nrow=6)

                    san_imgs = deblurry_imgs[0: 32]
                    utils.save_image(san_imgs, str(folder + f'sample-recons.png'), nrow=6)

                    san_imgs = direct_deblurry_imgs[0: 32]
                    utils.save_image(san_imgs, str(folder + f'sample-direct-recons.png'), nrow=6)


            if blurred_samples is None:
                blurred_samples = blurry_imgs
            else:
                blurred_samples = torch.cat((blurred_samples, blurry_imgs), dim=0)


            if original_sample is None:
                original_sample = og_img
            else:
                original_sample = torch.cat((original_sample, og_img), dim=0)


            if deblurred_samples is None:
                deblurred_samples = deblurry_imgs
            else:
                deblurred_samples = torch.cat((deblurred_samples, deblurry_imgs), dim=0)


            if direct_deblurred_samples is None:
                direct_deblurred_samples = direct_deblurry_imgs
            else:
                direct_deblurred_samples = torch.cat((direct_deblurred_samples, direct_deblurry_imgs), dim=0)

            cnt += og_img.shape[0]

        print(blurred_samples.shape)
        print(original_sample.shape)
        print(deblurred_samples.shape)
        print(direct_deblurred_samples.shape)

        fid_blur = fid_func(samples=[original_sample, blurred_samples])
        rmse_blur = torch.sqrt(torch.mean( (original_sample - blurred_samples)**2 ))
        ssim_blur = ssim(original_sample, blurred_samples, data_range=1, size_average=True)
        # n_og = original_sample.cpu().detach().numpy()
        # n_bs = blurred_samples.cpu().detach().numpy()
        # ssim_blur = ssim(n_og, n_bs, data_range=n_og.max() - n_og.min(), multichannel=True)
        print(f'The FID of blurry images with original image is {fid_blur}')
        print(f'The RMSE of blurry images with original image is {rmse_blur}')
        print(f'The SSIM of blurry images with original image is {ssim_blur}')


        fid_deblur = fid_func(samples=[original_sample, deblurred_samples])
        rmse_deblur = torch.sqrt(torch.mean((original_sample - deblurred_samples) ** 2))
        ssim_deblur = ssim(original_sample, deblurred_samples, data_range=1, size_average=True)
        print(f'The FID of deblurred images with original image is {fid_deblur}')
        print(f'The RMSE of deblurred images with original image is {rmse_deblur}')
        print(f'The SSIM of deblurred images with original image is {ssim_deblur}')

        print(f'Hence the improvement in FID using sampling is {fid_blur - fid_deblur}')

        fid_direct_deblur = fid_func(samples=[original_sample, direct_deblurred_samples])
        rmse_direct_deblur = torch.sqrt(torch.mean((original_sample - direct_deblurred_samples) ** 2))
        ssim_direct_deblur = ssim(original_sample, direct_deblurred_samples, data_range=1, size_average=True)
        print(f'The FID of direct deblurred images with original image is {fid_direct_deblur}')
        print(f'The RMSE of direct deblurred images with original image is {rmse_direct_deblur}')
        print(f'The SSIM of direct deblurred images with original image is {ssim_direct_deblur}')

        print(f'Hence the improvement in FID using direct sampling is {fid_blur - fid_direct_deblur}')


            # x0s = X_0s[-1]
            # for i in range(x0s.shape[0]):
            #     utils.save_image( (x0s[i]+1)*0.5, str(f'{self.results_folder}/' + f'sample-x0-{cnt}.png'))
            #     cnt += 1

    def save_training_data(self):
        dataset = self.ds
        create_folder(f'{self.results_folder}/')

        print(len(dataset))
        for idx in range(len(dataset)):
            img = dataset[idx]
            img = (img + 1) * 0.5
            utils.save_image(img, str(f'{self.results_folder}/' + f'{idx}.png'))
            if idx%1000 == 0:
                print(idx)

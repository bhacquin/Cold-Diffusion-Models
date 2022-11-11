import torch
from torchvision import transforms, utils
from torch.utils import data
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler
import torch.distributed as dist
from pathlib import Path
from generators.base_generator import BaseGenerator
import wandb
from PIL import Image, ImageFile
import numpy as np
import logging
from pathlib import Path
import copy
import time
from omegaconf import DictConfig, open_dict

from trainers.deblurring_diffusion_pytorch import GaussianDiffusion as BlurringDiffusion
from trainers.denoising_diffusion_pytorch import GaussianDiffusion as Gaussian_Diffusion
from trainers.demixing_diffusion_pytorch import GaussianDiffusion as MixingDiffusion
from utils_metrics.utils import exists, default, cycle, cycle_cat, num_to_groups, loss_backwards, extract, noise_like, cosine_beta_schedule
from models.Unet import Unet, EMA

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

LOG = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Dataset_Aug1(data.Dataset):
    def __init__(self, folder, image_size, exts = ['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').rglob(f'**/*.{ext}')]

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
        self.paths = [p for ext in exts for p in Path(f'{folder}').rglob(f'**/*.{ext}')]

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

class Generator(BaseGenerator):
    def __init__(self,cfg):
        super().__init__(cfg)
        self.cfg = cfg   
        self.mode_model = self.cfg.trainer.diffusion.type
        self.writer = None
        # self.trainer = MainTrainer(cfg)

    def setup_generator(self) -> None:
        # print('Setting up Generator')
        # print('TYPE ',self.cfg.trainer.diffusion.type)
        if self.cfg.trainer.diffusion.type == 'gaussian': 
            self.model = Gaussian_Diffusion(Unet(self.cfg), self.cfg)
            self.model.sampling_routine = self.cfg.generator.model.sampling_routine
            self.model.number_of_loop = self.cfg.generator.model.number_of_loop
        elif self.cfg.trainer.diffusion.type == 'deblur':
            self.model = BlurringDiffusion(Unet(self.cfg), self.cfg)
            self.model.sampling_routine = self.cfg.generator.model.sampling_routine
            self.model.number_of_loop = self.cfg.generator.model.number_of_loop
        elif self.cfg.trainer.diffusion.type == 'mix':
            self.model = MixingDiffusion(Unet(self.cfg), self.cfg)
        else:
            raise NotImplementedError
            self.model = None
        if self.cfg.trainer.gpu is not None:
            torch.cuda.set_device(self.cfg.generator.gpu)
            self.model.cuda(self.cfg.generator.gpu)
        else:
            LOG.error(
                f"No training on GPU possible on rank : {self.cfg.generator.rank}, local_rank (gpu_id) : {self.cfg.generator.gpu}"
            )
            raise NotImplementedError 
        self.num_timesteps = self.model.num_timesteps
        self.model = DistributedDataParallel(self.model, device_ids=[self.cfg.generator.gpu])
        dist.barrier()
        LOG.info("Model initialization passed successfully.") 
        if self.cfg.generator.noise.type == 'gaussian':
            cfg = self.cfg
            with open_dict(cfg):
                cfg.trainer.diffusion.timesteps = self.num_timesteps
            self.noising_diffusion = Gaussian_Diffusion(Unet(cfg), cfg)
        elif self.cfg.generator.noise.type == 'deblur':
            cfg = self.cfg
            with open_dict(cfg):
                cfg.trainer.diffusion.kernel_std = 0.15
                cfg.trainer.diffusion.kernel_size = 3
                cfg.trainer.diffusion.blur_routine = 'Incremental'            
                cfg.trainer.diffusion.train_routine = 'Final'
                cfg.trainer.diffusion.sampling_routine = 'x0_step_down'
                cfg.trainer.diffusion.timesteps = self.cfg.generator.noise.timesteps
            self.noising_diffusion = BlurringDiffusion(Unet(cfg), cfg)
        elif self.cfg.generator.noise.type == 'mix':
            self.noising_diffusion = MixingDiffusion(Unet(self.cfg), self.cfg)
        else:
            self.noising_diffusion = None
        
        if self.cfg.generator.gpu is not None:
            self.noising_diffusion.cuda(self.cfg.generator.gpu)
            self.noising_diffusion = DistributedDataParallel(self.noising_diffusion, device_ids=[self.cfg.generator.gpu])
            dist.barrier()
        else:
            self.noising_diffusion.cuda()
        try:
            self.folder = self.cfg.dataset.folder
        except:
            self.folder = None
            self.folder1 = self.cfg.dataset.folder1
            self.folder2 = self.cfg.dataset.folder2

        self.ema_decay = 0.995
        self.fp16 = self.cfg.generator.fp16
        self.results_folder = self.cfg.generator.results_folder
        self.load_path = self.cfg.generator.checkpointpath
        self.dataset = self.cfg.dataset.name
        self.mode = self.cfg.dataset.mode
        self.gpu = self.cfg.generator.gpu
        self.image_size = self.model.module.image_size
        self.ema = EMA(self.ema_decay)
        self.ema_model = copy.deepcopy(self.model)

        self.batch_size = self.cfg.generator.batch_size
        self.noise_type = self.cfg.generator.noise.type
        self.noise_timesteps = list(np.arange(1.,0.,-self.cfg.generator.noise.sample_step)) # self.cfg.generator.noise.timestep or []
        self.t_sampling = list(np.arange(1.,0.,-self.cfg.generator.model.sample_step))
        try:
            self.noising_diffusion.alphas_cumprod = self.model.module.alphas_cumprod
            self.noising_diffusion.sqrt_alphas_cumprod = self.model.module.sqrt_alphas_cumprod
            self.noising_diffusion.sqrt_one_minus_alphas_cumprod = self.model.module.sqrt_one_minus_alphas_cumprod
        except:
            # self.num_timesteps = self.cfg.generator.model.timesteps
            betas = cosine_beta_schedule(self.num_timesteps)
            alphas = 1. - betas
            alphas_cumprod = torch.cumprod(torch.from_numpy(alphas), axis=0).cuda(self.cfg.generator.gpu)
            self.noising_diffusion.register_buffer('alphas_cumprod', alphas_cumprod)
            self.noising_diffusion.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
            self.noising_diffusion.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

        if self.folder is not None:
            self.ds = Dataset(self.folder, self.image_size)            
            self.sampler = DistributedSampler(self.ds, num_replicas=self.cfg.generator.world_size, seed = self.cfg.generator.seed,  rank=self.cfg.generator.rank)
            self.dl = cycle(data.DataLoader(self.ds, batch_size = self.batch_size, sampler=self.sampler, pin_memory=True, num_workers=16, drop_last=True))
        else:
            LOG.info(f"dataset {self.dataset}")
            self.ds1 = Dataset(self.folder1, self.image_size)
            self.ds2 = Dataset(self.folder2, self.image_size)
            self.sampler1 = DistributedSampler(self.ds1, num_replicas=self.cfg.generator.world_size, seed = self.cfg.generator.seed,  rank=self.cfg.generator.rank)
            self.sampler2 = DistributedSampler(self.ds2, num_replicas=self.cfg.generator.world_size,seed = self.cfg.generator.seed + 1, rank=self.cfg.generator.rank)
            self.dl1 = cycle(data.DataLoader(self.ds1, batch_size = self.batch_size, sampler=self.sampler1, pin_memory=True, num_workers=8, drop_last=True))
            self.dl2 = cycle(data.DataLoader(self.ds2, batch_size=self.batch_size, sampler=self.sampler2, pin_memory=True, num_workers=8, drop_last=True))

        self.results_folder = Path(self.results_folder)
        self.results_folder.mkdir(exist_ok = True)
        if self.cfg.generator.gpu == self.cfg.generator.desired_gpu:
            LOG.info(f"Total timesteps: {self.num_timesteps}")
            LOG.info(f"Result dir: {self.results_folder}")
        self.reset_parameters()

        if self.load_path != None:
            self.load(self.load_path)
        LOG.info(f'Model loaded on {self.cfg.generator.gpu} .')
        # self.num_timesteps = self.cfg.generator.model.timesteps
        # betas = cosine_beta_schedule(self.num_timesteps)
        # alphas = 1. - betas
        # alphas_cumprod = torch.cumprod(torch.from_numpy(alphas), axis=0).cuda(self.cfg.generator.gpu)
        # self.model.module.register_buffer('alphas_cumprod', alphas_cumprod)
        # self.model.module.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        # self.model.module.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        # self.ema_model.module.register_buffer('alphas_cumprod', alphas_cumprod)
        # self.ema_model.module.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        # self.ema_model.module.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())


    def load(self, load_path):
        print("Loading : ", load_path)
        data = torch.load(load_path)
        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])


    def save_image_and_log(self,og_img1, all_images, direct_recons, xt,x_t_minus_1, noise_mode = 'Gaussian', 
                            model_mode = 'Gaussian', noise_step = 0, sampling_step = 0, loop = 1):
        try:
            og_img1 = (og_img1 + 1) * 0.5
            utils.save_image(og_img1, str(self.results_folder) + f'/original.png', nrow=6)

            all_images = (all_images + 1) * 0.5
            utils.save_image(all_images, str(self.results_folder) + f'/full_recons.png', nrow = 6)

            direct_recons = (direct_recons + 1) * 0.5
            utils.save_image(direct_recons, str(self.results_folder) + f'/direct_recons.png', nrow=6)

            xt = (xt + 1) * 0.5
            utils.save_image(xt, str(self.results_folder) + f'/x_t.png',nrow=6)

            x_t_minus_1 = (x_t_minus_1 + 1) * 0.5
            utils.save_image(x_t_minus_1, str(self.results_folder) + f'/x_t_minus_1.png',nrow=6)

            time.sleep(1)
            
            LOG.info("Logging image")
            wandb.log({f"Noise_{noise_step}_Loop_{loop}/target_sample": wandb.Image(str(self.results_folder) + f'/original.png'),
            'sampling_step' : sampling_step })
            wandb.log({f"Noise_{noise_step}_Loop_{loop}/full_reconstruction": wandb.Image(str(self.results_folder) + f'/full_recons.png'),
            'sampling_step' : sampling_step })
            wandb.log({f"Noise_{noise_step}_Loop_{loop}/direct_reconstruction": wandb.Image(str(self.results_folder) + f'/direct_recons.png'),
            'sampling_step' : sampling_step })
            wandb.log({f"Noise_{noise_step}_Loop_{loop}/xt": wandb.Image(str(self.results_folder) + f'/x_t.png'),
            'sampling_step' : sampling_step })
            wandb.log({f"Noise_{noise_step}_Loop_{loop}/x_t_minus_1": wandb.Image(str(self.results_folder) + f'/x_t_minus_1.png'),
            'sampling_step' : sampling_step })
            # wandb.log({f"{model_mode}_{noise_mode}_{noise_step}/target_sample": wandb.Image(str(self.results_folder) + f'/original.png'),
            # 'sampling_step' : sampling_step })
            # wandb.log({f"{model_mode}_{noise_mode}_{noise_step}/full_reconstruction": wandb.Image(str(self.results_folder) + f'/full_recons.png'),
            # 'sampling_step' : sampling_step })
            # wandb.log({f"{model_mode}_{noise_mode}_{noise_step}/direct_reconstruction": wandb.Image(str(self.results_folder) + f'/direct_recons.png'),
            # 'sampling_step' : sampling_step })
            # wandb.log({f"{model_mode}_{noise_mode}_{noise_step}/xt": wandb.Image(str(self.results_folder) + f'/x_t.png'),
            # 'sampling_step' : sampling_step })
            # wandb.log({f"{model_mode}_{noise_mode}_{noise_step}/x_t_minus_1": wandb.Image(str(self.results_folder) + f'/x_t_minus_1.png'),
            # 'sampling_step' : sampling_step })
            
        except Exception as e:
            LOG.error(f"Issue {e} when logging images for ")        


    def generate(self,noise_step = 0, start_step = 0): 
        with torch.no_grad():
            # try:
            noise_step_int = min(int(noise_step*self.num_timesteps), self.num_timesteps-1)
            start_step_int = min(int(start_step*self.num_timesteps), self.num_timesteps-1)
            
            if self.cfg.generator.gpu == self.cfg.generator.desired_gpu:
                LOG.info(f'Starting generation with {noise_step_int} noise steps and {start_step_int} sampling steps')
                if self.folder is not None:
                    batches = self.batch_size
                    data_1 = next(self.dl)
                    data_1 = data_1.cuda()
                else:
                    batches = self.batch_size
                    data_1 = next(self.dl1)
                    data_1 = data_1.cuda()
                    data_2 = next(self.dl2)
                    data_2 = data_2.cuda()

                ### depending on noise:
                if self.cfg.generator.noise.type == 'gaussian':
                    data_2 = torch.randn_like(data_1)
                    step = torch.full((self.batch_size,), noise_step_int, dtype=torch.long).cuda()
                    img = self.noising_diffusion.module.q_sample(x_start=data_1, x_end=data_2, t=step).float()

                elif self.cfg.generator.noise.type == 'deblur':
                    step = torch.full((self.batch_size,), noise_step_int, dtype=torch.long).cuda()
                    img = self.noising_diffusion.module.q_sample(x_start=data_1, t=step).float()

                elif self.cfg.generator.noise.type == 'mix':
                    step = torch.full((self.batch_size,), noise_step_int, dtype=torch.long).cuda()
                    img = self.noising_diffusion.module.q_sample(x_start=data_1, x_end=data_2, t=step).float()

                elif self.cfg.generator.noise.type == 'snow':
                    pass

                elif self.cfg.generator.noise.type == 'masked':
                    pass

                elif self.cfg.generator.noise.type == 'decolor':
                    pass
                else:
                    raise NotImplementedError
                for k in range(self.cfg.generator.model.number_of_loop):
                    xt, direct_recons, all_images, x_t_minus_1 = self.ema_model.module.gen_sample(batch_size=batches, img=img, t=start_step_int, number_of_loop=k+1)

                    self.save_image_and_log(data_1, all_images, direct_recons, xt, x_t_minus_1,
                                        noise_mode = self.cfg.generator.noise.type, 
                                        model_mode = self.mode_model, 
                                        noise_step = int(100*noise_step),
                                        loop = k+1)
                print(f'Generation with {noise_step_int} noise steps and {start_step_int} sampling steps completed')
                LOG.info(f'Generation with {noise_step_int} noise steps and {start_step_int} sampling steps completed')
            # except Exception as e:
            #     LOG.error(f'Error {e}, Noise step {noise_step}, sampling step {start_step}')

    def generate_all(self):
        for noise_step in self.noise_timesteps:
            for start_step in reversed(self.t_sampling):
                self.generate(noise_step, start_step)




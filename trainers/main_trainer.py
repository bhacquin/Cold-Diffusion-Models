import torch
import logging
import os
from omegaconf import DictConfig, open_dict
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from trainers.base_trainer import BaseTrainer
from models.Unet import Unet , EMA

LOG = logging.getLogger(__name__)

class MainTrainer(BaseTrainer):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        global GaussianDiffusion
        global Trainer
   
        LOG.setLevel(os.environ.get("LOGLEVEL", self.cfg.trainer.log_level))
        LOG.info(f"Diffusion type : {cfg.trainer.diffusion.type}")
        print(cfg.trainer.diffusion.type)
        if cfg.trainer.diffusion.type == "deblur" :
            print("loaded GaussianDiffusion from deblur")
            from trainers.deblurring_diffusion_pytorch import GaussianDiffusion, Trainer
        elif cfg.trainer.diffusion.type == 'decolor':
            raise NotImplementedError
            from trainers.deblurring_diffusion_pytorch import GaussianDiffusion, Trainer
        elif cfg.trainer.diffusion.type == 'defade_gaussian':
            from trainers.defading_diffusion_gaussian import  GaussianDiffusion, Trainer
        elif cfg.trainer.diffusion.type == 'defade_generate':
            from trainers.defading_diffusion_pytorch import  GaussianDiffusion, Trainer
        elif cfg.trainer.diffusion.type == 'demix':
            from trainers.demixing_diffusion_pytorch import GaussianDiffusion, Trainer
        elif cfg.trainer.diffusion.type == 'denoise':
            from trainers.denoising_diffusion_pytorch import GaussianDiffusion, Trainer
        elif cfg.trainer.diffusion.type == 'resolution':
            from trainers.resolution_diffusion_pytorch import GaussianDiffusion, Trainer
        else:
            raise NotImplementedError


    def eval(self) -> None:
        pass

    def setup_trainer(self) -> None:
        global GaussianDiffusion
        global Trainer
        if self.cfg.trainer.diffusion.type == "deblur" :
            print("loaded GaussianDiffusion from deblur")
            from trainers.deblurring_diffusion_pytorch import GaussianDiffusion, Trainer
        elif self.cfg.trainer.diffusion.type == 'decolor':
            raise NotImplementedError
            from trainers.deblurring_diffusion_pytorch import GaussianDiffusion, Trainer
        elif self.cfg.trainer.diffusion.type == 'defade_gaussian':
            from trainers.defading_diffusion_gaussian import  GaussianDiffusion, Trainer
        elif self.cfg.trainer.diffusion.type == 'defade_generate':
            from trainers.defading_diffusion_pytorch import  GaussianDiffusion, Trainer
        elif self.cfg.trainer.diffusion.type == 'demix':
            from trainers.demixing_diffusion_pytorch import GaussianDiffusion, Trainer
        elif self.cfg.trainer.diffusion.type == 'denoise':
            from trainers.denoising_diffusion_pytorch import GaussianDiffusion, Trainer
        elif self.cfg.trainer.diffusion.type == 'resolution':
            from trainers.resolution_diffusion_pytorch import GaussianDiffusion, Trainer
        else:
            raise NotImplementedError
        # Instantiate the model and optimizer
        self.model = Unet(self.cfg)          
        self.diffusion = GaussianDiffusion(self.model, self.cfg)
        
        if self.cfg.trainer.gpu is not None:
            torch.cuda.set_device(self.cfg.trainer.gpu)
            self.model.cuda(self.cfg.trainer.gpu)
            self.diffusion.cuda(self.cfg.trainer.gpu)
        else:
            LOG.error(
                f"No training on GPU possible on rank : {self.cfg.trainer.rank}, local_rank (gpu_id) : {self.cfg.trainer.gpu}"
            )
            raise NotImplementedError  

        self.diffusion = DistributedDataParallel(self.diffusion, device_ids=[self.cfg.trainer.gpu])
        dist.barrier()
        LOG.info("Initialization passed successfully.")

        ######## Looking for checkpoints
        LOG.info('Looking for existing checkpoints in output dir...')
        chk_path = self.cfg.trainer.checkpointpath
        checkpoint_dict = self.checkpoint_load(chk_path)
        if checkpoint_dict:
        # Load Model parameters
            LOG.info(f'Checkpoint found: {str(chk_path)}')
            # Load Model parameters
            try:
                try:
                    config = checkpoint_dict['config']
                    config = OmegaConf.create(config)
                    self.cfg = config
                except:
                    pass
                self.model = Unet(self.cfg)  
                self.diffusion = GaussianDiffusion(self.model,self.cfg)
                self.diffusion.cuda(self.cfg.trainer.gpu)
                self.diffusion = DistributedDataParallel(self.diffusion, device_ids=[self.cfg.trainer.gpu], find_unused_parameters=True)
                dist.barrier()
                self.diffusion.load_state_dict(checkpoint_dict["model_state_dict"])
            except:
                try:
                    self.model = Unet(self.cfg) 
                    self.model.cuda(self.cfg.trainer.gpu)
                    self.model = DistributedDataParallel(self.model, device_ids=[self.cfg.trainer.gpu], find_unused_parameters=True)
                    self.model.load_state_dict(checkpoint_dict["model_state_dict"])
                    self.diffusion = GaussianDiffusion(self.model,self.cfg,)
                except:
                    pass
        self.trainer = Trainer(self.diffusion,self.cfg, self.writer)
        dist.barrier()
    def train(self) -> None:
        LOG.info(f"Starting on node {self.cfg.trainer.rank}, gpu {self.cfg.trainer.gpu}")
        self.trainer.train()
        dist.barrier()
        if self.cfg.trainer.rank == 0:
            self.checkpoint_dump(checkpoint_path = self.cfg.trainer.checkpointpath, epoch=epoch)
        LOG.info(f"{self.cfg.trainer.rank}:{self.cfg.trainer.gpu} training finished")
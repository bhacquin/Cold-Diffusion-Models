from abc import abstractmethod
import logging
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional, Union
from omegaconf import DictConfig, OmegaConf, open_dict
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.cuda import init
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
# import torchaudio
import hashlib
import subprocess
import submitit
from tqdm import tqdm
import time

from omegaconf import DictConfig, open_dict
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.tensorboard import SummaryWriter
from clearml import Task
import wandb
from utils_metrics.utils import fix_random_seeds

LOG = logging.getLogger(__name__)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_local_distributed_mode(cfg: DictConfig) -> DictConfig:
    with open_dict(cfg):  # add to config
        # launched with torch.distributed.launch locally
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            cfg.generator.rank = int(os.environ["RANK"])
            cfg.generator.world_size = int(os.environ["WORLD_SIZE"])
            cfg.generator.gpu = int(os.environ["LOCAL_RANK"])
            cfg.trainer.rank = int(os.environ["RANK"])
            cfg.trainer.world_size = int(os.environ["WORLD_SIZE"])
            cfg.trainer.gpu = int(os.environ["LOCAL_RANK"])
        elif torch.cuda.is_available():
            LOG.info("Will run the code on one GPU.")
            # Naive launch
            # we manually add MASTER_ADDR and MASTER_PORT to env variables
            cfg.generator.rank, cfg.generator.gpu, cfg.generator.world_size = 0, 0, 1
            cfg.trainer.rank, cfg.trainer.gpu, cfg.trainer.world_size = 0, 0, 1
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = "29500"
        else:
            LOG.error("Does not support training without GPU.")
            sys.exit(1)

    # LOG.info(f'Init process group: {cfg.trainer}')
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=cfg.generator.world_size,
        rank=cfg.generator.rank,
    )

    torch.cuda.set_device(cfg.generator.gpu)
    # LOG.info('| distributed init (rank {}): {}'.format(cfg.trainer.rank, cfg.trainer.gpu))
    dist.barrier()
    return cfg




class BaseGenerator(object):
    model = None
    writer = None

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg


    def setup_local(self) -> None:
        # torchaudio.set_audio_backend("sox_io")
        self.cfg = init_local_distributed_mode(self.cfg)


    def setup_platform(self) -> None:
        fix_random_seeds(self.cfg.generator.seed)
        if self.cfg.generator.platform == "local":
            LOG.info(f"Generating platform : {self.cfg.generator.platform}")
            self.setup_local()
        elif self.cfg.generator.platform == "slurm":
            LOG.info(f"Generating platform : {self.cfg.generator.platform}")
            self.setup_slurm()
        else:
            raise NotImplementedError("Unknown platform (valid value are local or slurm)")


    def setup_tracker(self,) -> None:
        LOG.info(f"Generator: {self.cfg.generator.rank}, gpu: {self.cfg.generator.gpu}")
        self.writer = None
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        if self.cfg.generator.rank == self.cfg.generator.desired_gpu:
            if self.cfg.generator.use_clearml:
                task = Task.init(project_name="Diffusion", task_name=self.cfg.generator.ml_exp_name)
            if self.cfg.generator.use_wandb:
                wandb.init(project="Diffusion",entity=self.cfg.generator.wandb_entity, sync_tensorboard=True, dir = self.cfg.generator.results_folder)
                wandb.run.name = self.cfg.generator.ml_exp_name
                # wandb.run.save()
            self.writer = SummaryWriter()

    def setup_slurm(self) -> None:
        # torchaudio.set_audio_backend("soundfile")
        # torchaudio.set_audio_backend("sox_io")
        job_env = submitit.JobEnvironment()
        with open_dict(self.cfg):
            self.cfg.generator.job_id = job_env.job_id
            self.cfg.generator.gpu = job_env.local_rank
            self.cfg.generator.rank = job_env.global_rank
            self.cfg.generator.world_size = job_env.num_tasks
        LOG.info(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")
        LOG.error(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")
        ### Master address & Port
        if "MASTER_PORT" in os.environ:
            pass  # use MASTER_PORT in the environment variable
        else:
            # from PyTorch Lightning
            default_port = os.environ.get("SLURM_JOB_ID")
            job_id = default_port
            if default_port:
                # use the last 4 numbers in the job id as the id
                default_port = default_port[-4:]
                # all ports should be in the 10k+ range
                default_port = int(default_port) + 15000
            else:
                default_port = 12910
            os.environ["MASTER_PORT"] = str(default_port)

        # Master Addr
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(self.cfg.generator.world_size)  # str(ntasks)
        os.environ["LOCAL_RANK"] = str(self.cfg.generator.gpu)  # str(proc_id % num_gpus)
        os.environ["RANK"] = str(self.cfg.generator.rank)  # str(proc_id)
        print("WORLD SIZE :", self.cfg.generator.world_size)
        print("LOCAL_RANK :", self.cfg.generator.gpu)
        print("RANK :", self.cfg.generator.rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=self.cfg.generator.world_size,
            rank=self.cfg.generator.rank,
        )
        torch.cuda.set_device(self.cfg.generator.gpu)
        dist.barrier()
    
    @abstractmethod
    def setup_generator(self) -> None:
        pass

    def __call__(self) -> None:
        self.setup_platform()
        self.setup_tracker()
        self.setup_generator()
        self.run()

    @abstractmethod
    def generate(self) -> None:
        pass

    @abstractmethod
    def generate_all(self) -> None:
        pass


    def run(self) -> None:
        self.generate_all()
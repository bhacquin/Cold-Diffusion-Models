import pathlib
import urllib
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import math
import torch
from omegaconf import DictConfig, OmegaConf, open_dict



def get_output_dir(cfg, job_id=None) -> Path:
    out_dir = Path(cfg.trainer.output_dir)
    exp_name = str(cfg.trainer.ml_exp_name)
    folder_name = exp_name+'_'+str(job_id)
    p = Path(out_dir).expanduser()
    if job_id is not None:
        # p = p / str(job_id)
        p = p / folder_name
    p.mkdir(parents=True, exist_ok=True)
    return p


def add_key_value_to_conf(cfg: DictConfig, key: Any, value: Any) -> DictConfig:
    with open_dict(cfg):
        cfg[key] = value
    return cfg


def fix_random_seeds(seed: int = 31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
import logging
import hydra
import submitit
import os
from omegaconf import DictConfig, open_dict
from torch.distributed.elastic.multiprocessing.errors import record
from generators.generator import Generator
from utils_metrics.utils import get_output_dir
# import torch.distributed as dist
LOG = logging.getLogger(__name__)

@record
@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> int:
    generator = Generator(cfg)
    with open_dict(cfg):
        cfg.generator.results_folder = str(get_output_dir(cfg, cfg.generator.sync_key))
    if cfg.generator.platform == "local":
        # LOG.info(f"Output directory {cfg.trainer.output_dir}/{cfg.trainer.sync_key}")
        generator.setup_platform()
        generator.setup_tracker()
        generator.setup_generator()
        generator.run()
        return 0
    LOG.info(f"Current working directory: {os.getcwd()}")


    # Mode SLURM
    executor = submitit.AutoExecutor(folder=cfg.generator.results_folder, slurm_max_num_timeout=30)
    executor.update_parameters(
        mem_gb=cfg.generator.slurm.mem,
        gpus_per_node=cfg.generator.slurm.gpus_per_node,
        tasks_per_node=cfg.generator.slurm.gpus_per_node,  # one task per GPU
        cpus_per_task=cfg.generator.slurm.cpus_per_task,
        nodes=cfg.generator.slurm.nodes,
        timeout_min=int(cfg.generator.slurm.timeout) * 60,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_partition=cfg.generator.slurm.partition,
        slurm_qos=cfg.generator.slurm.qos,
        slurm_gres=f"gpu:{cfg.generator.slurm.gpus_per_node}"
        # slurm_signal_delay_s=120,
        # **kwargs
    )

    executor.update_parameters(name=cfg.generator.name)

    slurm_additional_parameters = {
        'requeue': True
    }

    if cfg.generator.slurm.account:
        slurm_additional_parameters['account'] = cfg.generator.slurm.account
    if cfg.generator.slurm.reservation:
        slurm_additional_parameters['reservation'] = cfg.generator.slurm.reservation

    executor.update_parameters(slurm_additional_parameters=slurm_additional_parameters)

    job = executor.submit(generator)
    LOG.info(f"Submitted job_id: {job.job_id}")
    return job


if __name__ == "__main__":
    main()

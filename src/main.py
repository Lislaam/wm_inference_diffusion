import os
from pathlib import Path
from typing import List, Union

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import torch
import yaml
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp

from trainer import Trainer
from train_in_wm import TrainInWM
from inference_time_planning import WMInference
from utils import skip_if_run_is_over


OmegaConf.register_new_resolver("eval", eval)

def sync_atari_env():
    project_root = Path(get_original_cwd())
    trainer_path = project_root / "config" / "trainer.yaml"
    atari_path   = project_root / "config" / "atari.yaml"

    # Load trainer.yaml
    with open(trainer_path, "r") as f:
        trainer_cfg = yaml.safe_load(f)

    # Load atari.yaml
    with open(atari_path, "r") as f:
        atari_cfg = yaml.safe_load(f)

    env_type = trainer_cfg.get("initialization", {}).get("env_type")
    if env_type:
        atari_cfg.setdefault("train", {})["id"] = f"{env_type}NoFrameskip-v4"
        with open(atari_path, "w") as f:
            yaml.safe_dump(atari_cfg, f, sort_keys=False)
        print(f"✅ Synced: train.id = {env_type}NoFrameskip-v4 in atari.yaml")
    else:
        print("⚠️ env_type not found in trainer.yaml")


@hydra.main(config_path="../config", config_name="trainer", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_visible_cuda_devices(cfg.common.devices)
    world_size = torch.cuda.device_count()
    root_dir = Path(hydra.utils.get_original_cwd())
    if cfg.training.agent_in_wm == True:
        run_in_wm(cfg, root_dir)
    elif world_size < 2:
        run(cfg, root_dir)
    else:
        mp.spawn(main_ddp, args=(world_size, cfg, root_dir), nprocs=world_size)


def main_ddp(rank: int, world_size: int, cfg: DictConfig, root_dir: Path) -> None:
    setup_ddp(rank, world_size)
    run(cfg, root_dir)
    destroy_process_group()


@skip_if_run_is_over
def run(cfg: DictConfig, root_dir: Path) -> None:
    trainer = Trainer(cfg, root_dir)
    trainer.run()

@skip_if_run_is_over
def run_in_wm(cfg: DictConfig, root_dir: Path) -> None:
    trainer = WMInference(cfg, root_dir)
    trainer.run()


def setup_ddp(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "6006"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def setup_visible_cuda_devices(devices: Union[str, int, List[int]]) -> None:
    if isinstance(devices, str):
        if devices == "cpu":
            devices = []
        else:
            assert devices == "all"
            return
    elif isinstance(devices, int):
        devices = [devices]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))


if __name__ == "__main__":
    sync_atari_env()
    main()

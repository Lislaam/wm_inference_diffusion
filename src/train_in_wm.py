from functools import partial
from pathlib import Path
import shutil
import time
from typing import List, Optional, Tuple

from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import wandb

from agent import Agent
from coroutines.collector import make_collector, NumToCollect
from data import BatchSampler, collate_segments_to_batch, Dataset, DatasetTraverser
from envs import make_atari_env, WorldModelEnv
from utils import (
    broadcast_if_needed,
    build_ddp_wrapper,
    CommonTools,
    configure_opt,
    count_parameters,
    get_lr_sched,
    keep_agent_copies_every,
    Logs,
    process_confusion_matrices_if_any_and_compute_classification_metrics,
    save_info_for_import_script,
    save_with_backup,
    set_seed,
    StateDictMixin,
    try_until_no_except,
    wandb_log,
)

class TrainInWM(StateDictMixin):
    def __init__(self, cfg: DictConfig, root_dir: Path) -> None:
        torch.backends.cuda.matmul.allow_tf32 = True
        OmegaConf.resolve(cfg)
        self._cfg = cfg
        self._rank = dist.get_rank() if dist.is_initialized() else 0

        set_seed(torch.seed() % 10 ** 9)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu", self._rank)
        if self._device.type == "cuda":
            torch.cuda.set_device(self._rank)

        try_until_no_except(
            partial(wandb.init, config=OmegaConf.to_container(cfg, resolve=True), reinit=True, resume=True, **cfg.wandb)
        )

        self._path_ckpt_dir = Path("trained_models")
        self._path_state_ckpt = self._path_ckpt_dir / "actor_critic_state.pt"
        self._path_ckpt_dir.mkdir(exist_ok=True, parents=True)

        # Datasets for initialising world model
        dataset_path = Path(cfg.static_dataset.path or "dataset")
        self.train_dataset = Dataset(dataset_path / "train", "train_dataset", cache_in_ram=True)
        self.test_dataset = Dataset(dataset_path / "test", "test_dataset", cache_in_ram=True)
        self.train_dataset.load_from_default_path()
        self.test_dataset.load_from_default_path()

        # Real atari env just to get num_actions. We don't use it to train because we have pretrained world model.
        test_env = make_atari_env(num_envs=cfg.collection.test.num_envs, device=self._device, **cfg.env.test)
        num_actions = int(test_env.num_actions)

        # Create a fresh agent with untrained actor_critic
        self.agent = Agent(instantiate(cfg.agent, num_actions=num_actions)).to(self._device)

        # Create models
        cfg.initialization.load_actor_critic = False # Train actor_critic from scratch
        self.agent = Agent(instantiate(cfg.agent, num_actions=num_actions)).to(self._device)
        self._agent = build_ddp_wrapper(**self.agent._modules) if dist.is_initialized() else self.agent
        if cfg.initialization.path_to_ckpt is not None:
            self.agent.load(**cfg.initialization)

        # Freeze world model parameters
        for param in self.agent.denoiser.parameters():
            param.requires_grad = False
        for param in self.agent.rew_end_model.parameters():
            param.requires_grad = False

        # Optimizer and scheduler for actor_critic only
        self.opt = {"actor_critic": configure_opt(self.agent.actor_critic, **cfg.actor_critic.optimizer)}
        self.lr_sched = {
            "actor_critic": get_lr_sched(self.opt["actor_critic"], cfg.actor_critic.training.lr_warmup_steps)
        }

        # Collect for imagination's initialization
        n = 1000
        dataset = Dataset(Path(f"dataset/BankHeist_1000"))
        dataset.load_from_default_path()
        if len(dataset) == 0:
            print(f"Collecting {n} steps in real environment for world model initialization.")
            collector = make_collector(test_env, self.agent.actor_critic, dataset, epsilon=0)
            collector.send(NumToCollect(steps=n))
            dataset.save_to_default_path()

        # DataLoader for RL training inside world model
        bs = BatchSampler(dataset, 0, 1, 1, cfg.agent.denoiser.inner_model.num_steps_conditioning, None, False)
        dl = DataLoader(dataset, batch_sampler=bs, collate_fn=collate_segments_to_batch)

        wm_env_cfg = instantiate(cfg.world_model_env)
        wm_env = WorldModelEnv(self.agent.denoiser, self.agent.rew_end_model, dl, wm_env_cfg)

        if cfg.training.compile_wm:
            wm_env.predict_next_obs = torch.compile(wm_env.predict_next_obs, mode="reduce-overhead")
            wm_env.predict_rew_end = torch.compile(wm_env.predict_rew_end, mode="reduce-overhead")

        # Setup agent training inside world model
        sigma_cfg = instantiate(cfg.denoiser.sigma_distribution)
        loss_cfg = instantiate(cfg.actor_critic.actor_critic_loss)
        self.agent.setup_training(sigma_cfg, loss_cfg, wm_env)

        self.epoch = 0
        self._num_epochs = cfg.training.num_final_epochs

    def run(self) -> None:
        while self.epoch < self._num_epochs:
            self.epoch += 1
            start_time = time.time()
            print(f"\nEpoch {self.epoch} / {self._num_epochs}\n")

            train_logs = self.train_actor_critic()
            test_logs = self.test_actor_critic()

            if test_logs:
                wandb.log({f"epoch": self.epoch})

            wandb.log({"duration": (time.time() - start_time) / 3600})
            self.save_checkpoint()

    def train_actor_critic(self) -> Logs:
        self.agent.actor_critic.train()
        opt = self.opt["actor_critic"]
        sched = self.lr_sched["actor_critic"]
        cfg = self._cfg.actor_critic.training

        to_log = []

        for _ in trange(cfg.steps_per_epoch, desc="Training actor_critic"):
            loss, metrics = self.agent.actor_critic()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.actor_critic.parameters(), cfg.max_grad_norm)
            opt.step()
            opt.zero_grad()
            sched.step()

            metrics["mean_action_probs_hist"] = wandb.Histogram(metrics["mean_action_probs"])
            wandb.log({f"train/actor_critic/{k}": v for k, v in metrics.items()})
            to_log.append(metrics)

        return to_log


    @torch.no_grad()
    def test_actor_critic(self) -> Logs:
        self.agent.actor_critic.eval()
        cfg = self._cfg.actor_critic.training

        to_log = []

        for _ in trange(cfg.steps_per_epoch, desc="Testing actor_critic"):
            _, metrics = self.agent.actor_critic()

            metrics["mean_action_probs"] = wandb.Histogram(metrics["mean_action_probs"])
            wandb.log({f"test/actor_critic/{k}": v for k, v in metrics.items()})
            to_log.append(metrics)

        return to_log

    def save_checkpoint(self) -> None:
        torch.save(self.agent.actor_critic.state_dict(), self._path_state_ckpt)

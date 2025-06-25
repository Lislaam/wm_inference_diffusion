from functools import partial
from pathlib import Path
import shutil
import time
import math
from typing import List, Optional, Tuple

from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import torch.distributed as dist
from torch.distributions.categorical import Categorical
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

        self._path_ckpt_dir = Path("/homes/53/fpinto/diamond/trained_models/")
        self._path_state_ckpt = self._path_ckpt_dir / "actor_critic_state.pt"
        self._path_ckpt_dir.mkdir(exist_ok=True, parents=True)

        # Datasets for initialising world model
        dataset_path = Path(cfg.static_dataset.path or "dataset")
        self.train_dataset = Dataset(dataset_path / "train", "train_dataset", cache_in_ram=True)
        self.test_dataset = Dataset(dataset_path / "test", "test_dataset", cache_in_ram=True)
        self.train_dataset.load_from_default_path()
        self.test_dataset.load_from_default_path()

        # Need to eval agent in real environment
        self.test_env = make_atari_env(num_envs=cfg.collection.test.num_envs, device=self._device, **cfg.env.test)
        self.num_actions = int(self.test_env.num_actions)

        # Create models
        cfg.initialization.load_actor_critic = False # Train actor_critic from scratch
        self.agent = Agent(instantiate(cfg.agent, num_actions=self.num_actions)).to(self._device)

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
            self.collector = make_collector(self.test_env, self.agent.actor_critic, dataset, epsilon=0)
            self.collector.send(NumToCollect(steps=n))
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
        loss_cfg = instantiate(cfg.actor_critic.actor_critic_loss)
        self.agent.actor_critic.setup_training(wm_env, loss_cfg)

        self.epoch = 0
        self._num_epochs = cfg.training.num_final_epochs

    def run(self) -> None:
        wandb.watch(self.agent.actor_critic, log="all", log_freq=100)
        while self.epoch < self._num_epochs:
            self.epoch += 1
            start_time = time.time()
            print(f"\nEpoch {self.epoch} / {self._num_epochs}\n")

            train_logs = self.train_actor_critic()
            wandb.log({f"epoch": self.epoch})
            wandb.log({"duration": (time.time() - start_time) / 3600})
            if self._cfg.evaluation.should and self.epoch % self._cfg.evaluation.every == 0:
                test_logs = self.val_actor_critic()

            self.save_checkpoint()

        self.eval_actor_critic()
        self.save_checkpoint()
        return None

    def train_actor_critic(self) -> Logs:
        self.agent.actor_critic.train()
        opt = self.opt["actor_critic"]
        sched = self.lr_sched["actor_critic"]
        cfg = self._cfg.actor_critic.training

        to_log = []

        if self.epoch == 1:
            for step in trange(cfg.steps_first_epoch, desc="Training actor_critic"):
                loss, metrics = self.agent.actor_critic()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.actor_critic.parameters(), cfg.max_grad_norm)
                opt.step()
                opt.zero_grad()
                sched.step()

                if step % 50 == 0:
                    wandb.log({
                        f"train/actor_critic/mean_action_probs_hist": wandb.Histogram(metrics["mean_action_probs"]),
                        **{f"train/actor_critic/{k}": v for k, v in metrics.items() if k != "mean_action_probs"}
                    })
                    
                wandb.log({f"train/actor_critic/{k}": v for k, v in metrics.items() if k != "mean_action_probs"})
                to_log.append(metrics)

        else:
            for step in trange(cfg.steps_per_epoch, desc="Training actor_critic"):
                loss, metrics = self.agent.actor_critic()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.actor_critic.parameters(), cfg.max_grad_norm)
                opt.step()
                opt.zero_grad()
                sched.step()

            if step % 50 == 0:
                wandb.log({
                    f"train/actor_critic/mean_action_probs_hist": wandb.Histogram(metrics["mean_action_probs"]),
                    **{f"train/actor_critic/{k}": v for k, v in metrics.items() if k != "mean_action_probs"}
                })

            wandb.log({f"train/actor_critic/{k}": v for k, v in metrics.items() if k != "mean_action_probs"})
            to_log.append(metrics)

        return to_log


    @torch.no_grad()
    def val_actor_critic(self) -> Logs:
        self.agent.actor_critic.eval()
        cfg = self._cfg.actor_critic.training

        to_log = []

        for step in trange(cfg.steps_per_epoch, desc="Validating actor_critic"):
            _, metrics = self.agent.actor_critic()

            if step % 50 == 0:
                wandb.log({
                    f"val/actor_critic/mean_action_probs_hist": wandb.Histogram(metrics["mean_action_probs"]),
                    **{f"val/actor_critic/{k}": v for k, v in metrics.items() if k != "mean_action_probs"}
                })

            wandb.log({f"val/actor_critic/{k}": v for k, v in metrics.items() if k != "mean_action_probs"})
            to_log.append(metrics)

        return to_log
    
    @torch.no_grad()
    def eval_actor_critic(self) -> Logs:
        self.agent.actor_critic.eval()
        env = self.test_env
        obs = env.reset()[0]
        done = torch.zeros(env.num_envs, dtype=torch.bool, device=self._device)
        num_episodes = self._cfg.actor_critic.training.num_eval

        episode_rewards = torch.zeros(env.num_envs, device=self._device)
        completed_rewards = []

        all_probs = []

        # Init LSTM hidden state
        batch_size = env.num_envs
        hx = torch.zeros(batch_size, self.agent.actor_critic.lstm_dim, device=self._device)
        cx = torch.zeros(batch_size, self.agent.actor_critic.lstm_dim, device=self._device)

        step_count = 0  # For logging steps

        while len(completed_rewards) < num_episodes:
            out = self.agent.actor_critic.predict_act_value(obs, (hx, cx))
            logits, _, (hx, cx) = out

            dist = Categorical(logits=logits)
            actions = dist.sample()
            probs = dist.probs.detach().cpu()
            entropies = dist.entropy().detach().cpu()

            all_probs.append(probs)

            obs, rewards, terminated, truncated, infos = env.step(actions)
            done = terminated | truncated
            episode_rewards += rewards

            # Per-step logging
            wandb.log({
                "eval/actor_critic/step_reward": rewards.mean().item(),
                "eval/actor_critic/step_policy_entropy": entropies.mean().item() / math.log(2),
                "eval/actor_critic/step_action_distribution": wandb.Histogram(probs.mean(dim=0).numpy()),
            })

            step_count += 1

            for i, d in enumerate(done):
                if d:
                    completed_rewards.append(episode_rewards[i].item())
                    episode_rewards[i] = 0.0
                    hx[i] = 0
                    cx[i] = 0

        # Final summary stats
        mean_return = sum(completed_rewards[:num_episodes]) / num_episodes
        std_return = torch.tensor(completed_rewards[:num_episodes]).std().item()

        all_probs = torch.cat(all_probs, dim=0)
        mean_probs = all_probs.mean(dim=0)
        entropy = Categorical(probs=all_probs).entropy().mean() / math.log(2)

        metrics = {
            "eval/actor_critic/return_mean": mean_return,
            "eval/actor_critic/return_std": std_return,
            "eval/actor_critic/policy_entropy": entropy.item(),
            "eval/actor_critic/mean_action_probs": wandb.Histogram(mean_probs.numpy()),
        }

        wandb.log(metrics)
        return [metrics]


    def save_checkpoint(self) -> None:
        self._path_state_ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.agent.actor_critic.state_dict(), self._path_state_ckpt)

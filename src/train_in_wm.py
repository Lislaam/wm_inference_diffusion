from functools import partial
from pathlib import Path
import shutil
import time
from typing import List, Optional, Tuple
import math

from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.distributions.categorical import Categorical
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
        self._world_size = dist.get_world_size() if dist.is_initialized() else 1

        # Pick a random seed
        set_seed(torch.seed() % 10 ** 9)

        # Device
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu", self._rank)
        print(f"Starting on {self._device}")
        self._use_cuda = self._device.type == "cuda"
        if self._use_cuda:
            torch.cuda.set_device(self._rank)  # fix compilation error on multi-gpu nodes

        # Init wandb
        if self._rank == 0:
            try_until_no_except(
                partial(wandb.init, config=OmegaConf.to_container(cfg, resolve=True), reinit=True, resume=True, **cfg.wandb)
            )

        # Flags
        self._is_static_dataset = cfg.static_dataset.path is not None
        self._is_model_free = cfg.training.model_free

        # Checkpointing
        self._path_ckpt_dir = Path("trained_models")
        self._path_state_ckpt = self._path_ckpt_dir / "state.pt"
        self._keep_agent_copies = partial(
            keep_agent_copies_every,
            every=cfg.checkpointing.save_agent_every,
            path_ckpt_dir=self._path_ckpt_dir,
            num_to_keep=cfg.checkpointing.num_to_keep,
        )
        self._save_info_for_import_script = partial(
            save_info_for_import_script, run_name=cfg.wandb.name, path_ckpt_dir=self._path_ckpt_dir
        )

        # First time, init files hierarchy
        if not cfg.common.resume and self._rank == 0:
            self._path_ckpt_dir.mkdir(exist_ok=False, parents=False)
            path_config = Path("config") / "trainer.yaml"
            path_config.parent.mkdir(exist_ok=False, parents=False)
            shutil.move(".hydra/config.yaml", path_config)
            wandb.save(str(path_config))
            shutil.copytree(src=root_dir / "src", dst="./src")
            shutil.copytree(src=root_dir / "scripts", dst="./scripts")

        # Datasets
        num_workers = cfg.training.num_workers_data_loaders
        use_manager = cfg.training.cache_in_ram and (num_workers > 0)
        p = Path(cfg.static_dataset.path) if self._is_static_dataset else Path("dataset")
        self.train_dataset = Dataset(p / "train", "train_dataset", cfg.training.cache_in_ram, use_manager)
        self.test_dataset = Dataset(p / "test", "test_dataset", cache_in_ram=True)
        self.train_dataset.load_from_default_path()
        self.test_dataset.load_from_default_path()

        if self._rank == 0:
            # train_env = make_atari_env(num_envs=cfg.collection.train.num_envs, device=self._device, **cfg.env.train)
            test_env = make_atari_env(num_envs=cfg.collection.test.num_envs, device=self._device, **cfg.env.test) # Want to eval in real environment
            num_actions = int(test_env.num_actions)
        else:
            num_actions = None
        num_actions, = broadcast_if_needed(num_actions)

        # Create models
        self.agent = Agent(instantiate(cfg.agent, num_actions=num_actions)).to(self._device)
        self._agent = build_ddp_wrapper(**self.agent._modules) if dist.is_initialized() else self.agent

        if cfg.initialization.path_to_ckpt is not None:
            self.agent.load(**cfg.initialization)

        ######################################################

        # Optimizers and LR schedulers

        def build_opt(name: str) -> torch.optim.AdamW:
            return configure_opt(getattr(self.agent, name), **getattr(cfg, name).optimizer)

        def build_lr_sched(name: str) -> torch.optim.lr_scheduler.LambdaLR:
            return get_lr_sched(self.opt.get(name), getattr(cfg, name).training.lr_warmup_steps)

        self._model_names = ["denoiser", "rew_end_model", "actor_critic"] # only training the actor critic
        self.opt = CommonTools(*map(build_opt, self._model_names))
        self.lr_sched = CommonTools(*map(build_lr_sched, self._model_names))

        # Data loaders

        make_data_loader = partial(
            DataLoader,
            dataset=self.train_dataset,
            collate_fn=collate_segments_to_batch,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            pin_memory=self._use_cuda,
            pin_memory_device=str(self._device) if self._use_cuda else "",
        )

        make_batch_sampler = partial(BatchSampler, self.train_dataset, self._rank, self._world_size)

        def get_sample_weights(sample_weights: List[float]) -> Optional[List[float]]:
            return None if (self._is_static_dataset and cfg.static_dataset.ignore_sample_weights) else sample_weights

        c = cfg.denoiser.training
        seq_length = cfg.agent.denoiser.inner_model.num_steps_conditioning + 1 + c.num_autoregressive_steps
        bs = make_batch_sampler(c.batch_size, seq_length, get_sample_weights(c.sample_weights))
        dl_denoiser_train = make_data_loader(batch_sampler=bs)
        dl_denoiser_test = DatasetTraverser(self.test_dataset, c.batch_size, seq_length)

        c = cfg.rew_end_model.training
        bs = make_batch_sampler(c.batch_size, c.seq_length, get_sample_weights(c.sample_weights), can_sample_beyond_end=True)
        dl_rew_end_model_train = make_data_loader(batch_sampler=bs)
        dl_rew_end_model_test = DatasetTraverser(self.test_dataset, c.batch_size, c.seq_length)

        self._data_loader_train = CommonTools(dl_denoiser_train, dl_rew_end_model_train, None)
        self._data_loader_test = CommonTools(dl_denoiser_test, dl_rew_end_model_test, None)

        # RL env

        # if self._is_model_free:
        #     rl_env = make_atari_env(num_envs=cfg.actor_critic.training.batch_size, device=self._device, **cfg.env.train)

        # else:
        c = cfg.actor_critic.training
        sl = cfg.agent.denoiser.inner_model.num_steps_conditioning
        bs = make_batch_sampler(c.batch_size, sl, get_sample_weights(c.sample_weights))
        dl_actor_critic = make_data_loader(batch_sampler=bs)
        wm_env_cfg = instantiate(cfg.world_model_env)
        rl_env = WorldModelEnv(self.agent.denoiser, self.agent.rew_end_model, dl_actor_critic, wm_env_cfg)

        if cfg.training.compile_wm:
            rl_env.predict_next_obs = torch.compile(rl_env.predict_next_obs, mode="reduce-overhead")
            rl_env.predict_rew_end = torch.compile(rl_env.predict_rew_end, mode="reduce-overhead")

        # Setup training
        sigma_distribution_cfg = instantiate(cfg.denoiser.sigma_distribution)
        actor_critic_loss_cfg = instantiate(cfg.actor_critic.actor_critic_loss)
        self.agent.setup_training(sigma_distribution_cfg, actor_critic_loss_cfg, rl_env)

        # Training state (things to be saved/restored)
        self.epoch = 0
        self.num_epochs_collect = None
        self.num_episodes_test = 0
        self.num_batch_train = CommonTools(0, 0, 0)
        self.num_batch_test = CommonTools(0, 0, 0)

        if cfg.common.resume:
            self.load_state_checkpoint()
        else:
            self.save_checkpoint()

        if self._rank == 0:
            for name in self._model_names:
                print(f"{count_parameters(getattr(self.agent, name))} parameters in {name}")
            print(self.train_dataset)
            print(self.test_dataset)

    def run(self) -> None:
        to_log = []

        if self.epoch == 0:
            if self._is_model_free or self._is_static_dataset:
                self.num_epochs_collect = 0
            else:
                raise ValueError(
                    "Initial collection is required for training with a static dataset or model-free training. "
                    "Set `static_dataset.path` to None or `training.model_free` to True in the config."
                )

        num_epochs = self.num_epochs_collect + self._cfg.training.num_final_epochs

        while self.epoch < num_epochs:
            self.epoch += 1
            start_time = time.time()

            if self._rank == 0:
                print(f"\nEpoch {self.epoch} / {num_epochs}\n")

            sd_train_dataset, = broadcast_if_needed(self.train_dataset.state_dict())  # update dataset for ranks > 0
            self.train_dataset.load_state_dict(sd_train_dataset)
            
            if self._cfg.training.should:
                to_log += self.train_agent()

            # Evaluation
            should_test = self._rank == 0 and self._cfg.evaluation.should and (self.epoch % self._cfg.evaluation.every == 0)

            if should_test and not self._is_model_free:
                to_log += self.test_agent()

            # Logging
            to_log.append({"duration": (time.time() - start_time) / 3600})
            if self._rank == 0:
                wandb_log(to_log, self.epoch)
            to_log = []

            # Checkpointing
            self.save_checkpoint()
            
            if dist.is_initialized():
                dist.barrier()


    def train_agent(self) -> Logs:
        self.agent.train()
        self.agent.zero_grad()
        to_log = []
        model_names = ["actor_critic"] # only training the actor critic
        for name in model_names:
            cfg = getattr(self._cfg, name).training
            if self.epoch > cfg.start_after_epochs:
                steps = cfg.steps_first_epoch if self.epoch == 1 else cfg.steps_per_epoch
                to_log += self.train_component(name, steps)
        return to_log

    @torch.no_grad()
    def test_agent(self) -> Logs:
        self.agent.eval()
        to_log = []
        to_log += self.eval_ac_in_real()

        return to_log

    def train_component(self, name: str, steps: int) -> Logs:
        cfg = getattr(self._cfg, name).training
        model = getattr(self._agent, name)
        opt = self.opt.get(name)
        lr_sched = self.lr_sched.get(name)
        data_loader = self._data_loader_train.get(name)

        model.train()
        opt.zero_grad()
        data_iterator = iter(data_loader) if data_loader is not None else None
        to_log = []

        num_steps = cfg.grad_acc_steps * steps

        for i in trange(num_steps, desc=f"Training {name}", disable=self._rank > 0):
            batch = next(data_iterator).to(self._device) if data_iterator is not None else None
            loss, metrics = model(batch) if batch is not None else model()
            loss.backward()

            num_batch = self.num_batch_train.get(name)
            metrics[f"num_batch_train_{name}"] = num_batch
            self.num_batch_train.set(name, num_batch + 1)

            if (i + 1) % cfg.grad_acc_steps == 0:
                if cfg.max_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    metrics["grad_norm_before_clip"] = grad_norm

                opt.step()
                opt.zero_grad()

                if lr_sched is not None:
                    metrics["lr"] = lr_sched.get_last_lr()[0]
                    lr_sched.step()

            to_log.append(metrics)

            if i % 50 == 0:
                wandb.log({
                    f"actor_critic/train/mean_action_probs_hist": wandb.Histogram(metrics["mean_action_probs"])})

        process_confusion_matrices_if_any_and_compute_classification_metrics(to_log)
        to_log = [{f"{name}/train/{k}": v for k, v in d.items()} for d in to_log]
        return to_log
    
    @torch.no_grad()
    def eval_ac_in_real(self) -> Logs:
        """
        Evaluate the actor-critic in the real environment, using the World Model part to forcast an n-step lookahead.
        """
        self.agent.actor_critic.eval()

        # Create the environments
        env = make_atari_env(num_envs=self._cfg.collection.test.num_envs, device=self._device, **self._cfg.env.test)

        # Initialize the real environment
        obs = env.reset()[0]
        done = torch.zeros(env.num_envs, dtype=torch.bool, device=self._device)
        num_episodes = self._cfg.actor_critic.training.num_eval

        episode_rewards = []

        all_probs = []

        # Init LSTM hidden state
        batch_size = env.num_envs
        hx = torch.zeros(batch_size, self.agent.actor_critic.lstm_dim, device=self._device)
        cx = torch.zeros(batch_size, self.agent.actor_critic.lstm_dim, device=self._device)

        for i in trange(num_episodes, desc=f"Evaluating actor-critic"):
            if not done:
                out = self.agent.actor_critic.predict_act_value(obs, (hx, cx))
                logits, _, (hx, cx) = out

                dist = Categorical(logits=logits)
                actions = dist.sample()
                probs = dist.probs.detach().cpu()
                entropies = dist.entropy().detach().cpu() # One entry

                all_probs.append(probs)

                obs, rewards, terminated, truncated, infos = env.step(actions)
                done = terminated | truncated
                episode_rewards.append(rewards)

                # Per-step logging
                wandb.log({
                    "actor_critic/eval/step_reward": rewards.item(),
                    "actor_critic/eval/cumulative_reward": sum(episode_rewards),
                    "actor_critic/eval/policy_entropy": entropies.item() / math.log(2),
                    "actor_critic/eval/mean_action_distribution": wandb.Histogram(probs.mean(dim=0).numpy()), # ACtually probs.mean is the same as probs at dim 0
                })

        # Final summary stats
        mean_return = sum(episode_rewards[:num_episodes]) / num_episodes
        std_return = torch.tensor(episode_rewards[:num_episodes]).std().item()

        all_probs = torch.cat(all_probs, dim=0)
        mean_probs = all_probs.mean(dim=0)
        entropy = Categorical(probs=all_probs).entropy().mean() / math.log(2)

        metrics = {
            "actor_critic/eval/return_mean": mean_return,
            "actor_critic/eval/return_std": std_return,
            "actor_critic/eval/policy_entropy": entropy.item(),
            "actor_critic/eval/mean_action_probs": wandb.Histogram(mean_probs.numpy()),
        }

        wandb.log(metrics)
        return None


    def save_checkpoint(self) -> None:
        self._path_state_ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.agent.actor_critic.state_dict(), self._path_state_ckpt)

    def load_state_checkpoint(self) -> None:
        self.load_state_dict(torch.load(self._path_state_ckpt, map_location=self._device))

    def save_checkpoint(self) -> None:
        if self._rank == 0:
            save_with_backup(self.state_dict(), self._path_state_ckpt)
            self._keep_agent_copies(self.agent.state_dict(), self.epoch)
            self._save_info_for_import_script(self.epoch)

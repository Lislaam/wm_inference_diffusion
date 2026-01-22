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
from PIL import Image, ImageDraw, ImageFont

from tqdm import tqdm, trange
import wandb

from agent import Agent
from coroutines.collector import make_collector, NumToCollect
from data import BatchSampler, collate_segments_to_batch, Dataset, DatasetTraverser
from envs import make_atari_env, WorldModelEnv
from planning_rollout import multistep_planning

from utils import (
    broadcast_if_needed,
    build_ddp_wrapper,
    CommonTools,
    configure_opt,
    count_parameters,
    get_lr_sched,
    Logs,
    set_seed,
    StateDictMixin,
    try_until_no_except,
    wandb_log,
)


class WMInference(StateDictMixin):
    def __init__(self, cfg: DictConfig, root_dir: Path) -> None:
        torch.backends.cuda.matmul.allow_tf32 = True
        OmegaConf.resolve(cfg)
        self._cfg = cfg
        self._rank = dist.get_rank() if dist.is_initialized() else 0
        self._world_size = dist.get_world_size() if dist.is_initialized() else 1

        # Pick a random seed
        self.seed_number = self._cfg.common.seed
        self.seed = set_seed(self.seed_number)
            # torch.seed() % 10 ** 9)

        # Device
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu", self._rank)
        print(f"Starting on {self._device}")
        self._use_cuda = self._device.type == "cuda"
        if self._use_cuda:
            torch.cuda.set_device(self._rank)  # fix compilation error on multi-gpu nodes

        # Init wandb
        if self._rank == 0:
            try_until_no_except(
                partial(wandb.init, config=OmegaConf.to_container(cfg, resolve=True), reinit=True, resume=False, **cfg.wandb)
            )

        # Flags
        self._is_static_dataset = cfg.static_dataset.path is not None
        self._is_model_free = cfg.training.model_free

        # Checkpointing
        self._path_ckpt_dir = Path("trained_models")
        self._path_state_ckpt = self._path_ckpt_dir / "state.pt"

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
        self.train_dataset.load_from_default_path()

        # Real environment
        self.env = make_atari_env(num_envs=self._cfg.collection.test.num_envs, seed=self.seed, device=self._device, **self._cfg.env.test)
        num_actions = int(self.env.num_actions)
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

        self._model_names = ["denoiser", "rew_end_model", "actor_critic"]
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
        seq_length = cfg.agent.denoiser.inner_model.num_steps_conditioning + 1 + c.num_autoregressive_steps # 5 or 6
        bs = make_batch_sampler(c.batch_size, seq_length, get_sample_weights(c.sample_weights))
        dl_denoiser_train = make_data_loader(batch_sampler=bs)

        c = cfg.rew_end_model.training
        bs = make_batch_sampler(c.batch_size, c.seq_length, get_sample_weights(c.sample_weights), can_sample_beyond_end=True)
        dl_rew_end_model_train = make_data_loader(batch_sampler=bs)

        self._data_loader_train = CommonTools(dl_denoiser_train, dl_rew_end_model_train, None)

        # RL env
        c = cfg.actor_critic.training
        sl = cfg.agent.denoiser.inner_model.num_steps_conditioning
        bs = make_batch_sampler(c.batch_size, sl, get_sample_weights(c.sample_weights))
        dl_actor_critic = make_data_loader(batch_sampler=bs)
        wm_env_cfg = instantiate(cfg.world_model_env)

        self.rl_env = WorldModelEnv(self.agent.denoiser, self.agent.rew_end_model, dl_actor_critic, wm_env_cfg)

        if cfg.training.compile_wm:
            self.rl_env.predict_next_obs = torch.compile(self.rl_env.predict_next_obs, mode="reduce-overhead")
            self.rl_env.predict_rew_end = torch.compile(self.rl_env.predict_rew_end, mode="reduce-overhead")

        # Training state (things to be saved/restored)
        self.epoch = 0
        self.num_epochs_collect = None
        self.num_episodes_test = 0

        if self._rank == 0:
            for name in self._model_names:
                print(f"{count_parameters(getattr(self.agent, name))} parameters in {name}")

    def run(self) -> None:
        # So I can log and compare these graphs in wandb
        wandb.define_metric("eval_step")
        wandb.define_metric("actor_critic/eval/step_reward", step_metric="eval_step")
        wandb.define_metric("actor_critic/eval/planned_step_reward", step_metric="eval_step")
        wandb.define_metric("actor_critic/eval/cumulative_reward", step_metric="eval_step")
        wandb.define_metric("actor_critic/eval/planned_cumulative_reward", step_metric="eval_step")
        wandb.define_metric("actor_critic/eval/value", step_metric="eval_step")
        wandb.define_metric("actor_critic/eval/planned_value", step_metric="eval_step")
        wandb.define_metric("actor_critic/eval/td_error", step_metric="eval_step")
        wandb.define_metric("actor_critic/eval/planned_td_error", step_metric="eval_step")
        wandb.define_metric("actor_critic/eval/planning_flag", step_metric="eval_step")
        wandb.define_metric("unplanned_obs_sequence", step_metric="eval_step")
        wandb.define_metric("obs_sequence", step_metric="eval_step")
        wandb.define_metric("wm_obs_sequence", step_metric="eval_step")
        wandb.define_metric("meta_planning_depth", step_metric="eval_step")
        wandb.define_metric("step_time", step_metric="eval_step")

        # Evaluation
        # start_time = time.time()
        # self.eval_plain()
        # if self._rank == 0:
        #     wandb.log({"duration_plain": (time.time() - start_time)})

        # Same env with WM planning
        start_time = time.time()
        self.eval_with_planning()
        # Logging
        if self._rank == 0:
            wandb.log({"duration_with_planning": (time.time() - start_time)})

        if dist.is_initialized():
            dist.barrier()

        return None


    @torch.no_grad()
    def eval_plain(self) -> Logs:
        """
        Evaluate the actor-critic in the real environment.
        """
        self.agent.actor_critic.eval()

        env = self.env
        # Initialize the real environment
        obs = env.reset(seed=self.seed)[0]
        done = torch.zeros(env.num_envs, dtype=torch.bool, device=self._device)
        num_episodes = self._cfg.actor_critic.training.num_eval

        episode_rewards = []
        episode_values = []
        episode_td_errors = []
        entropies =[]
        all_probs = []

        # Init LSTM hidden state
        batch_size = env.num_envs
        hx = torch.zeros(batch_size, self.agent.actor_critic.lstm_dim, device=self._device)
        cx = torch.zeros(batch_size, self.agent.actor_critic.lstm_dim, device=self._device)

        step = 0

        for i in trange(num_episodes, desc=f"Evaluating actor-critic"):
            if not done:
                out = self.agent.actor_critic.predict_act_value(obs, (hx, cx))
                logits, value, (hx, cx) = out

                dist = Categorical(logits=logits)
                action = dist.sample() # Not perfectly deterministic like the argmax # .probs.argmax(dim=-1)
                probs = dist.probs.detach().cpu()
                entropy = dist.entropy().detach().cpu().item() / math.log(2)

                value_t = value.clone().detach()

                obs, rewards, terminated, truncated, infos = env.step(action)
                done = terminated | truncated

                with torch.no_grad():
                    _, value_tp1, _ = self.agent.actor_critic.predict_act_value(obs, (hx, cx))

                td_error = rewards + self._cfg.actor_critic.actor_critic_loss.gamma * (1 - done.float()) * value_tp1 - value_t
                td_error_mean = td_error.mean().item()

                episode_rewards.append(rewards)
                episode_values.append(value)
                episode_td_errors.append(td_error_mean)

                entropies.append(entropy)
                all_probs.append(probs)

                # Plotting real images to wandb
                obs_plot = obs
                # Convert to numpy and permute to HWC format
                frames = []
                for frame in obs_plot:
                    img = frame.detach().cpu().numpy()  # [3, 64, 64]
                    img = np.transpose(img, (1, 2, 0))  # [64, 64, 3]
                    
                    # Convert to uint8 if necessary
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                    
                    frames.append(img)

                # Horizontally stack frames: [64, 64 * 4, 3]
                grid = np.concatenate(frames, axis=1)

                # Per-step logging
                wandb.log({
                    "eval_step": step,
                    "actor_critic/eval/value": value,
                    "actor_critic/eval/td_error": td_error_mean,
                    "actor_critic/eval/step_reward": rewards.item(),
                    "actor_critic/eval/cumulative_reward": sum(episode_rewards),
                    "actor_critic/eval/policy_entropy": entropy,
                    "actor_critic/eval/mean_action_distribution": wandb.Histogram(probs.mean(dim=0).numpy()),
                    "unplanned_obs_sequence": wandb.Image(Image.fromarray(grid)),
                })

                step += 1

        # Final summary stats
        mean_return = sum(episode_rewards[:num_episodes]) / num_episodes
        std_return = torch.tensor(episode_rewards[:num_episodes]).std().item()
        mean_entropy = torch.tensor(entropies).mean().item()
        std_entropy = torch.tensor(entropies).std().item()
        mean_value = torch.tensor(episode_values).mean().item()
        std_value = torch.tensor(episode_values).std().item()
        mean_td_error = torch.tensor(episode_td_errors).mean().item()
        std_td_error = torch.tensor(episode_td_errors).std().item()

        all_probs = torch.cat(all_probs, dim=0)

        wandb.log({
            "actor_critic/eval/return_mean": mean_return,
            "actor_critic/eval/return_std": std_return,
            "actor_critic/eval/entropy_mean": mean_entropy,
            "actor_critic/eval/entropy_std": std_entropy,
            "actor_critic/eval/value_mean": mean_value,
            "actor_critic/eval/value_std": std_value,
            "actor_critic/eval/td_error_mean": mean_td_error,
            "actor_critic/eval/td_error_std": std_td_error,
        })

        return None
    
    @torch.no_grad()
    def eval_with_planning(self) -> Logs:
        """
        Evaluate the actor-critic in the real environment, using the World Model part to forecast an n-step lookahead.
        """
        self.agent.actor_critic.eval()

        env = self.env
        world_model_env = self.rl_env
        obs = env.reset(seed=self.seed)[0]
        done = torch.zeros(env.num_envs, dtype=torch.bool, device=self._device)

        world_model_env.reset_no_data()  # builds internal buffers
        num_episodes = self._cfg.actor_critic.training.num_eval

        episode_rewards = []
        episode_td_errors = []
        entropies = []
        all_probs = []

        self.agent.hx = torch.zeros(env.num_envs, self.agent.actor_critic.lstm_dim, device=self._device)
        self.agent.cx = torch.zeros(env.num_envs, self.agent.actor_critic.lstm_dim, device=self._device)

        step = 0
        planning_flag = 0
        plan_count = 0
        for i in trange(num_episodes, desc="Evaluating actor-critic with planning"):
            if not done:
                start_time = time.time()
                logits, value, (self.agent.hx, self.agent.cx) = self.agent.actor_critic.predict_act_value(obs, (self.agent.hx, self.agent.cx))
                dist = Categorical(logits=logits)
                actions = dist.sample()
                probs = dist.probs.detach().cpu()
                entropy = dist.entropy().detach().cpu().item() / math.log(2)

                use_real_step = ((i < self._cfg.agent.denoiser.inner_model.num_steps_conditioning) 
                                 or (np.random.uniform() < self._cfg.evaluation.planning_percentage) 
                                 or (self._cfg.evaluation.planning_steps == 0))

                if use_real_step:
                    planning_flag = 0
                    depth = 0
                    obs, rewards, terminated, truncated, infos = env.step(actions)
                    done = terminated | truncated
                    episode_rewards.append(rewards)

                    # Update WM buffers and hiddens with real action and obs following step() logic
                    world_model_env.act_buffer[:, -1] = actions
                    predicted_obs, _ = world_model_env.sampler.sample(world_model_env.obs_buffer, world_model_env.act_buffer) # Added to test
                    test_sanity_rew, _, (world_model_env.hx_rew_end, world_model_env.cx_rew_end) = world_model_env.rew_end_model.predict_rew_end(
                        world_model_env.obs_buffer[:, -1:],
                        world_model_env.act_buffer[:, -1:],
                        obs.unsqueeze(1),
                        (world_model_env.hx_rew_end, world_model_env.cx_rew_end),
                    )
                    rew_model_reward = Categorical(logits=test_sanity_rew).sample().squeeze(1) - 1.0
                    world_model_env.obs_buffer = world_model_env.obs_buffer.roll(-1, dims=1)
                    world_model_env.act_buffer = world_model_env.act_buffer.roll(-1, dims=1)
                    world_model_env.obs_buffer[:, -1] = obs # Changed from obs to predicted_obs to test

                    entropies.append(entropy)
                    all_probs.append(probs)

                else: # Planning step 
                    # Now we test out every action inside the world model and select the best one
                    # Cannot step() with every action as this will update the buffers.
                    # Copy some of the WM step() code and use it to predict obs and rewards
                    planning_flag = 100
                    plan_count += 1

                    plan = multistep_planning(self.agent, world_model_env, self.env.num_actions, self._cfg)
                    best_action, candidate_actions, action_predicted_rews, wm_predicted_obs, entropy, depth = plan

                    obs, rewards, terminated, truncated, infos = env.step(best_action)
                    done = terminated | truncated

                    # Update WM buffers and hiddens with the best action
                    world_model_env.act_buffer[:, -1] = best_action
                    predicted_obs, _ = world_model_env.sampler.sample(world_model_env.obs_buffer, world_model_env.act_buffer) # Added to test
                    test_sanity_rew, _, (world_model_env.hx_rew_end, world_model_env.cx_rew_end) = world_model_env.rew_end_model.predict_rew_end(
                        world_model_env.obs_buffer[:, -1:],
                        world_model_env.act_buffer[:, -1:],
                        obs.unsqueeze(1),
                        (world_model_env.hx_rew_end, world_model_env.cx_rew_end),
                    )
                    rew_model_reward = Categorical(logits=test_sanity_rew).sample().squeeze(1) - 1.0
                    world_model_env.obs_buffer = world_model_env.obs_buffer.roll(-1, dims=1)
                    world_model_env.act_buffer = world_model_env.act_buffer.roll(-1, dims=1)
                    world_model_env.obs_buffer[:, -1] = obs

                    episode_rewards.append(rewards)
                    entropies.append(entropy)

                    ###############################################

                    # Constants
                    font = ImageFont.load_default()

                    # Convert and annotate each frame
                    grid_rows = []
                    for row_idx in range(len(wm_predicted_obs[0])):  # for each of the n rows
                        row_images = []
                        for col_idx in range(len(wm_predicted_obs)):  # for each of the a columns
                            frame = wm_predicted_obs[col_idx][row_idx]  # [3, 64, 64]
                            img = frame.detach().cpu().numpy()
                            img = np.transpose(img, (1, 2, 0))  # [64, 64, 3]

                            # Convert to uint8
                            if img.max() <= 1.0:
                                img = (img * 255).astype(np.uint8)
                            else:
                                img = img.astype(np.uint8)

                            pil_img = Image.fromarray(img)
                            draw = ImageDraw.Draw(pil_img)

                            # Get reward and color
                            rew = action_predicted_rews[col_idx, row_idx]
                            if rew == 1:
                                color = (0, 255, 0)
                            elif rew == -1:
                                color = (0, 128, 255)
                            else:
                                color = (255, 255, 255)

                            draw.text((2, 2), f"R={rew:.0f}", fill=color, font=font)
                            row_images.append(np.array(pil_img))

                        # Horizontally stack the row
                        row_strip = np.concatenate(row_images, axis=1)  # [64, 64*a, 3]
                        grid_rows.append(row_strip)
                    # Vertically stack the rows to form final grid
                    final_grid = np.concatenate(grid_rows, axis=0)  # [64*n, 64*a, 3]
                    # Log to wandb
                    wandb.log({"wm_grid_rewards": wandb.Image(Image.fromarray(final_grid))})
                    wandb.log({"actor_critic/eval/candidate_actions": wandb.Histogram(candidate_actions)})

                # Plotting WM images to wandb
                wm_obs_plot = world_model_env.obs_buffer[0,:,:,:,:]  # [4, 3, 64, 64]
                # Convert to numpy and permute to HWC format
                frames = []
                for frame in wm_obs_plot:
                    img = frame.detach().cpu().numpy()  # [3, 64, 64]
                    img = np.transpose(img, (1, 2, 0))  # [64, 64, 3]
                    
                    # Convert to uint8 if necessary
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                    
                    frames.append(img)
                # Horizontally stack frames: [64, 64 * 4, 3]
                wm_grid = np.concatenate(frames, axis=1)

                # Plotting real images to wandb
                obs_plot = obs
                # Convert to numpy and permute to HWC format
                frames = []
                for frame in obs_plot:
                    img = frame.detach().cpu().numpy()  # [3, 64, 64]
                    img = np.transpose(img, (1, 2, 0))  # [64, 64, 3]
                    
                    # Convert to uint8 if necessary
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                    
                    frames.append(img)
                # Horizontally stack frames: [64, 64 * 4, 3]
                grid = np.concatenate(frames, axis=1)

                logits_next, value_next, _ = self.agent.actor_critic.predict_act_value(obs, (self.agent.hx, self.agent.cx))
                td_error = (rewards + self._cfg.actor_critic.actor_critic_loss.gamma * value_next - value).abs()
                episode_td_errors.append(td_error.item())

                # Log the step
                wandb.log({
                    "eval_step": step,
                    "planning_flag": planning_flag,
                    "actor_critic/eval/planned_value": value,
                    "actor_critic/eval/planned_td_error": td_error.item(),
                    "actor_critic/eval/planned_step_reward": rewards.item(),
                    "actor_critic/eval/planned_rew_model_reward": rew_model_reward.item(),
                    "actor_critic/eval/planned_cumulative_reward": sum(episode_rewards),
                    "actor_critic/eval/planned_mean_action_distribution": wandb.Histogram(probs.numpy()),
                    "actor_critic/eval/planned_entropy": entropy,
                    "obs_sequence": wandb.Image(Image.fromarray(grid)),           # from real obs
                    "wm_obs_sequence": wandb.Image(Image.fromarray(wm_grid)),     # from world model buffer
                    "meta_planning_depth": depth,
                    "step_time": time.time() - start_time
                })

                step += 1

        mean_return = sum(episode_rewards[:num_episodes]) / num_episodes
        std_return = torch.tensor(episode_rewards[:num_episodes]).std().item()
        mean_entropy = torch.tensor(entropies).mean().item()
        std_entropy = torch.tensor(entropies).std().item()
        mean_td_error = torch.tensor(episode_td_errors).mean().item()
        std_td_error = torch.tensor(episode_td_errors).std().item()
        all_probs = torch.cat(all_probs, dim=0)

        wandb.log({
            "actor_critic/eval/planned_return_mean": mean_return,
            "actor_critic/eval/planned_return_std": std_return,
            "actor_critic/eval/planned_entropy_mean": mean_entropy,
            "actor_critic/eval/planned_entropy_std": std_entropy,
            "actor_critic/eval/planned_td_error_mean": mean_td_error,
            "actor_critic/eval/planned_td_error_std": std_td_error,
            "actor_critic/eval/num_planning_steps": plan_count
        })

        return None
    

    def load_state_checkpoint(self) -> None:
        self.load_state_dict(torch.load(self._path_state_ckpt, map_location=self._device))
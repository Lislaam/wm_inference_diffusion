from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

import ale_py
import gymnasium
from gymnasium.vector import AsyncVectorEnv
import numpy as np
import torch
from torch import Tensor
from procgen import ProcgenEnv

from .atari_preprocessing import AtariPreprocessing


def make_atari_env(
    id: str,
    num_envs: int,
    device: torch.device,
    done_on_life_loss: bool,
    size: int,
    max_episode_steps: Optional[int],
    seed: Optional[int] = None,
) -> TorchEnv:
    def env_fn(rank=0):
        def _thunk():
            env = gymnasium.make(
                id,
                full_action_space=False,
                frameskip=1,
                render_mode="rgb_array",
                max_episode_steps=max_episode_steps,
            )

            # Set deterministic seed here
            env.reset(seed=seed + rank if seed is not None else None)

            # Set deterministic Atari preprocessing
            env = AtariPreprocessing(
                env=env,
                noop_max=0,  # << set to 0 to disable random no-ops
                frame_skip=4,
                screen_size=size,
            )
            return env
        return _thunk
    
    env = AsyncVectorEnv([env_fn(rank) for rank in range(num_envs)])
    
    # def env_fn():
    #     env = gymnasium.make(
    #         id,
    #         full_action_space=False,
    #         frameskip=1,
    #         render_mode="rgb_array",
    #         max_episode_steps=max_episode_steps,
    #     )
    #     env = AtariPreprocessing(
    #         env=env,
    #         noop_max=30,
    #         frame_skip=4,
    #         screen_size=size,
    #     )
    #     return env

    # env = AsyncVectorEnv([env_fn for _ in range(num_envs)])

    # The AsyncVectorEnv resets the env on termination, which means that it will
    # reset the environment if we use the default AtariPreprocessing of gymnasium with
    # terminate_on_life_loss=True (which means that we will only see the first life).
    # Hence a separate wrapper for life_loss, coming after the AsyncVectorEnv.

    if done_on_life_loss:
        env = DoneOnLifeLoss(env)

    env = TorchEnv(env, device)

    return env


def make_procgen_env(
    id: str,
    num_envs: int,
    device: torch.device,
    size: int,
    distribution_mode: str = "easy",
    num_levels: int = 200,
    start_level: int = 0,
    max_episode_steps: Optional[int] = None,
    seed: Optional[int] = None,
    **_: Any,
) -> TorchProcgenEnv:
    env_name = _extract_procgen_env_name(id)
    env = ProcgenEnv(
        num_envs=num_envs,
        env_name=env_name,
        num_levels=num_levels,
        start_level=start_level,
        distribution_mode=distribution_mode,
        rand_seed=0 if seed is None else seed,
    )
    return TorchProcgenEnv(env, device=device, size=size, max_episode_steps=max_episode_steps)


def make_env(name: str, **kwargs: Any) -> gymnasium.Wrapper:
    if name == "atari":
        return make_atari_env(**kwargs)
    if name == "procgen":
        return make_procgen_env(**kwargs)
    raise ValueError(f"Unknown env family '{name}'. Expected 'atari' or 'procgen'.")


class DoneOnLifeLoss(gymnasium.Wrapper):
    def __init__(self, env: AsyncVectorEnv) -> None:
        super().__init__(env)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        obs, rew, end, trunc, info = self.env.step(actions)
        life_loss = info["life_loss"]
        if life_loss.any():
            end[life_loss] = True
            info["final_observation"] = obs
        return obs, rew, end, trunc, info


class TorchEnv(gymnasium.Wrapper):
    def __init__(self, env: gymnasium.Env, device: torch.device) -> None:
        super().__init__(env)
        self.device = device
        self.num_envs = env.observation_space.shape[0]
        self.num_actions = env.unwrapped.single_action_space.n
        b, h, w, c = env.observation_space.shape
        self.observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(b, c, h, w))

    def reset(self, *args, **kwargs) -> Tuple[Tensor, Dict[str, Any]]:
        obs, info = self.env.reset(*args, **kwargs)
        return self._to_tensor(obs), info

    def step(self, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]:
        obs, rew, end, trunc, info = self.env.step(actions.cpu().numpy())
        dead = np.logical_or(end, trunc)
        if dead.any():
            info["final_observation"] = self._to_tensor(np.stack(info["final_observation"][dead]))
        obs, rew, end, trunc = (self._to_tensor(x) for x in (obs, rew, end, trunc))
        return obs, rew, end, trunc, info

    def _to_tensor(self, x: Tensor) -> Tensor:
        if x.ndim == 4:
            return torch.tensor(x, device=self.device).div(255).mul(2).sub(1).permute(0, 3, 1, 2).contiguous()
        elif x.dtype is np.dtype("bool"):
            return torch.tensor(x, dtype=torch.uint8, device=self.device)
        else:
            return torch.tensor(x, dtype=torch.float32, device=self.device)


class TorchProcgenEnv(gymnasium.Wrapper):
    def __init__(
        self,
        env: ProcgenEnv,
        device: torch.device,
        size: int,
        max_episode_steps: Optional[int],
    ) -> None:
        super().__init__(env)
        self.device = device
        self.num_envs = env.num_envs
        self.num_actions = env.action_space.n
        self.size = size
        self.max_episode_steps = max_episode_steps
        self.ep_len = np.zeros(self.num_envs, dtype=np.int32)

        rgb_space = env.observation_space.spaces["rgb"]
        h, w, c = rgb_space.shape
        self.observation_space = gymnasium.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_envs, c, h, w),
            dtype=np.float32,
        )
        self.action_space = env.action_space

    def reset(self, *args, **kwargs) -> Tuple[Tensor, Dict[str, Any]]:
        self.ep_len.fill(0)
        obs = self.env.reset()
        return self._to_tensor(self._extract_rgb(obs)), {}

    def step(self, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]:
        obs, rew, done, info = self.env.step(actions.cpu().numpy())
        obs_rgb = self._extract_rgb(obs)

        self.ep_len += 1
        terminated = done.astype(bool)
        if self.max_episode_steps is None:
            truncated = np.zeros_like(terminated, dtype=bool)
        else:
            truncated = self.ep_len >= self.max_episode_steps

        dead = np.logical_or(terminated, truncated)
        out_info: Dict[str, Any] = {}
        if dead.any():
            out_info["final_observation"] = self._to_tensor(obs_rgb[dead])
            self.ep_len[dead] = 0

        obs_t = self._to_tensor(obs_rgb)
        rew_t = torch.tensor(rew, dtype=torch.float32, device=self.device)
        term_t = torch.tensor(terminated, dtype=torch.uint8, device=self.device)
        trunc_t = torch.tensor(truncated, dtype=torch.uint8, device=self.device)
        return obs_t, rew_t, term_t, trunc_t, out_info

    def _extract_rgb(self, obs: Any) -> np.ndarray:
        return obs["rgb"] if isinstance(obs, dict) else obs

    def _to_tensor(self, x: np.ndarray) -> Tensor:
        if x.ndim == 4:
            return torch.tensor(x, device=self.device).div(255).mul(2).sub(1).permute(0, 3, 1, 2).contiguous()
        if x.dtype is np.dtype("bool"):
            return torch.tensor(x, dtype=torch.uint8, device=self.device)
        return torch.tensor(x, dtype=torch.float32, device=self.device)


def _extract_procgen_env_name(id_: str) -> str:
    # Supported forms: "coinrun", "procgen-coinrun-v0", "procgen:procgen-coinrun-v0"
    normalized = id_.split(":", 1)[-1]
    if normalized.startswith("procgen-") and normalized.endswith("-v0"):
        return normalized[len("procgen-") : -len("-v0")]
    return normalized

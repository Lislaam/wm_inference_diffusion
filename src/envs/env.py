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
) -> TorchEnv:
    def env_fn():
        env = gymnasium.make(
            id,
            full_action_space=False,
            frameskip=1,
            render_mode="rgb_array",
            max_episode_steps=max_episode_steps,
        )
        env = AtariPreprocessing(
            env=env,
            noop_max=30,
            frame_skip=4,
            screen_size=size,
        )
        return env

    env = AsyncVectorEnv([env_fn for _ in range(num_envs)])

    # The AsyncVectorEnv resets the env on termination, which means that it will
    # reset the environment if we use the default AtariPreprocessing of gymnasium with
    # terminate_on_life_loss=True (which means that we will only see the first life).
    # Hence a separate wrapper for life_loss, coming after the AsyncVectorEnv.

    if done_on_life_loss:
        env = DoneOnLifeLoss(env)

    env = TorchEnv(env, device)

    return env


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


    

def make_procgen_env(
    id: str,
    num_envs: int,
    device: torch.device,
    size: int,  # e.g., 64 for 64x64 images
    distribution_mode: str = "easy",
    render_mode: Optional[str] = None,
    num_levels: int = 200,
    start_level: int = 0,
    max_episode_steps: Optional[int] = 1000,
    seed: Optional[int] = None,
    done_on_life_loss: Optional[bool] = None,
    **_: Any,
) -> "TorchProcgenEnv":
    """
    Create a vectorized Procgen environment (num_envs copies), then wrap it to
    return PyTorch tensors in NCHW format, normalized to [-1, 1].
    """
    # Create a single ProcgenEnv that is already vectorized to `num_envs`.
    env_name = _extract_procgen_env_name(id)
    env = ProcgenEnv(
        num_envs=num_envs,
        env_name=env_name,
        distribution_mode=distribution_mode,
        render_mode=render_mode,
        num_levels=num_levels,
        start_level=start_level,
        rand_seed=0 if seed is None else seed,
                )
    # if max_episode_steps is not None:
    #     from gym.wrappers import TimeLimit  # or gymnasium.wrappers.TimeLimit if using Gymnasium
    #     # Check if environment is vectorized via an 'envs' attribute.
    #     if hasattr(env, "envs"):
    #         # Patch each sub-environment with a dummy spec if missing.
    #         for sub_env in env.envs:
    #             if not hasattr(sub_env, "spec"):
    #                 sub_env.spec = None
    #         # Wrap each sub-environment.
    #         env.envs = [TimeLimit(sub_env, max_episode_steps=max_episode_steps) for sub_env in env.envs]
    #     else:
    #         # For non-vectorized env, patch spec if missing.
    #         if not hasattr(env, "spec"):
    #             env.spec = None
    #         env = TimeLimit(env, max_episode_steps=max_episode_steps)
    

    # Wrap it in a TorchProcgenEnv to handle PyTorch transformation.
    return TorchProcgenEnv(env, device, size=size)


def _extract_procgen_env_name(id_: str) -> str:
    """
    Supported forms:
      - coinrun
      - procgen-coinrun-v0
      - procgen:procgen-coinrun-v0
    """
    normalized = id_.split(":", 1)[-1]
    if normalized.startswith("procgen-") and normalized.endswith("-v0"):
        normalized = normalized[len("procgen-") : -len("-v0")]
    return normalized.lower()


class TorchProcgenEnv(gymnasium.Wrapper):
    """
    Minimal wrapper that:
      1. Uses a vectorized ProcgenEnv with `num_envs`.
      2. Converts observations from [N, H, W, C] to [N, C, H, W].
      3. Normalizes pixels from [0, 255] to [-1, 1].
      4. Returns actions, rewards, etc. as PyTorch tensors on the specified device.
    """

    def __init__(self, env: ProcgenEnv, device: torch.device, size: int):
        super().__init__(env)
        self.device = device
        self.num_envs = env.num_envs
        self.num_actions = env.action_space.n
        self.size = size

        original_space = env.observation_space
        rgb_space = original_space.spaces["rgb"]  # This is the actual Box
        h, w, c = rgb_space.shape

        self.observation_space = gymnasium.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_envs, c, h, w),
            dtype=np.float32,
        )
        self.action_space = env.action_space

    def reset(self, seed=None, options=None, **kwargs):
        # Gymnasium 0.26+ needs seed and options to be passed as kwargs, remoe them from kwargs
        obs = self.env.reset(**kwargs)
        return self._to_tensor(obs['rgb']), {}

    def step(self, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]:
        result = self.env.step(actions.cpu().numpy())
        if len(result) == 4:
            # Gymnasium 0.26+ procgen envs return (obs, rew, done, info)
            obs, rew, done, info = result
            terminated = done
            truncated = np.zeros_like(terminated, dtype=bool)  # or just np.array([False]*N)
        elif len(result) == 5:
            obs, rew, terminated, truncated, info = result
        else:
            raise ValueError(f"Expected 4 or 5 return items, got {len(result)}")

        dead = np.logical_or(terminated, truncated)
        info = {}
        if dead.any():
            # Extract the "rgb" key from obs (assuming it's a dictionary)
            final_obs = obs['rgb'][dead]
            final_obs_t = self._to_tensor(final_obs)
            info['final_observation'] = final_obs_t

        obs_t = self._to_tensor(obs["rgb"])
        rew_t = torch.tensor(rew, device=self.device, dtype=torch.float32)
        term_t = torch.tensor(terminated, device=self.device, dtype=torch.int64)
        trunc_t = torch.tensor(truncated, device=self.device, dtype=torch.int64)

        # Now return the standard 5-tuple as Gymnasium 0.26+ expects:
        return obs_t, rew_t, term_t, trunc_t, info

    def _to_tensor(self, x: Tensor) -> Tensor:
        if x.ndim == 4:
            return torch.tensor(x, device=self.device).div(255).mul(2).sub(1).permute(0, 3, 1, 2).contiguous()
        elif x.dtype is np.dtype("bool"):
            return torch.tensor(x, dtype=torch.uint8, device=self.device)
        else:
            return torch.tensor(x, dtype=torch.float32, device=self.device)

import torch
from pathlib import Path
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from envs import make_atari_env

from agent import Agent, AgentConfig


@hydra.main(config_path="../config", config_name="trainer", version_base="1.3")
def main(cfg: DictConfig):
    test_env = make_atari_env(num_envs=cfg.collection.test.num_envs, device='cuda', **cfg.env.test)
    num_actions = int(test_env.num_actions)

    # Create the agent
    agent = Agent(instantiate(cfg.agent, num_actions=num_actions)).cuda()

    # Load pretrained world model (skip actor-critic)
    print("Before load:")
    print("Denoiser mean:", next(agent.denoiser.parameters()).data.mean().item())
    print("RewEnd mean:", next(agent.rew_end_model.parameters()).data.mean().item())
    print("ActorCritic mean (random init):", next(agent.actor_critic.parameters()).data.mean().item())
    # If working, only the actor_critic mean will be the same before and after load

    agent.load(
        path_to_ckpt=Path("/homes/53/fpinto/diamond/trained_models/BankHeist.pt"),
        load_denoiser=True,
        load_rew_end_model=True,
        load_actor_critic=False,
    )

    # Verify it worked
    print("After load:")
    print("Denoiser mean:", next(agent.denoiser.parameters()).data.mean().item())
    print("✅ Denoiser mean:", next(agent.denoiser.parameters()).data.mean().item())
    print("✅ RewEnd mean:", next(agent.rew_end_model.parameters()).data.mean().item())
    print("🟡 ActorCritic mean (random init):", next(agent.actor_critic.parameters()).data.mean().item())


if __name__ == "__main__":
    main()
# This will only work for the BankHeist env for now

from envs.world_model_env import WorldModelEnv, WorldModelEnvConfig
from agent import Agent, AgentConfig
from utils import load_config, save_config

# Load the configuration for the agent and environment


# Get the diffusion and reward model from agent (no need actor-critic)
agent = Agent(agent_config, env_config)

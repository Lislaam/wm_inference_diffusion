import math
from hydra.utils import instantiate
import numpy as np
import torch
from torch.distributions.categorical import Categorical


def multistep_planning(agent, world_model_env, num_actions, num_step_forcast, gamma, mode='reward') -> torch.Tensor:
    """
    Perform multistep planning using the world model environment.
    Args:
        agent: The agent to use for planning.
        world_model_env: The world model environment to use for planning.
        num_actions: Number of actions to consider for planning.
        num_step_forcast: Number of steps to forecast in the future.
    Returns:
        best_action: The best next step to take based on the planning.
    """
    assert mode in ['reward', 'value', 'td'], "Mode must be either 'reward' or 'value' or 'td'"

    candidate_actions = []
    action_predicted_rews = np.zeros((num_actions, num_step_forcast)) # Store predicted rewards for each action to log
    action_predicted_values = np.zeros((num_actions, num_step_forcast))
    action_predicted_tds = np.zeros((num_actions, num_step_forcast-1))
    wm_predicted_obs = [[] for _ in range(num_actions)]  # Store predicted observations for plotting

    # Backup buffers for planning
    obs_buffer_base = world_model_env.obs_buffer.clone()
    act_buffer_base = world_model_env.act_buffer.clone()
    wm_hx_base = world_model_env.hx_rew_end.clone()
    wm_cx_base = world_model_env.cx_rew_end.clone()
    hx_base = agent.hx.clone()
    cx_base = agent.cx.clone()
    
    for a in range(num_actions):
        # Propose an action a, and estimate rollout reward in imagination using real actor-critic
        # Following WM step() logic but not updating buffers and hiddens
        # Copy buffers to avoid modifying the original ones
        obs_buffer = obs_buffer_base.clone()
        act_buffer = act_buffer_base.clone()
        wm_hx = wm_hx_base.clone()
        wm_cx = wm_cx_base.clone()
        agent_hx = hx_base.clone()
        agent_cx = cx_base.clone()

        act_buffer[:, -1] = a  # candidate action

        for i in range(num_step_forcast):
            rews = [] # Store rewards for multiple samples
            next_obs, _ = world_model_env.sampler.sample(obs_buffer, act_buffer)
            logits_rew, _, (wm_hx, wm_cx) = world_model_env.rew_end_model.predict_rew_end(
                obs_buffer[:, -1:], act_buffer[:, -1:], next_obs.unsqueeze(1),
                (wm_hx, wm_cx)
            )
            for _ in range (10):  # Sample multiple times to get a good estimate
                rews.append((Categorical(logits=logits_rew).sample().squeeze(1) - 1.0).item())
                    # in {-1, 0, 1}
            # probs = torch.softmax(logits_rew, dim=-1).cpu()
            # print(probs)

            obs_buffer = obs_buffer.roll(-1, dims=1)
            act_buffer = act_buffer.roll(-1, dims=1)
            obs_buffer[:, -1] = next_obs  # predicted next obs

            # Get actor-critic to choose next action
            logits, value, (agent_hx, agent_cx) = agent.actor_critic.predict_act_value(
                next_obs, (agent_hx, agent_cx) # Squeeze obs to fit
            )
            dist = Categorical(logits=logits)
            entropy = dist.entropy().detach().cpu().item() / math.log(2) # May need to watch this
            act_buffer[:, -1] = dist.sample() # At the last step this will be reset unused

            # Store predicted obs and rewards etc for logging
            wm_predicted_obs[a].append(next_obs.squeeze())
            action_predicted_rews[a, i] = max(rews)
            action_predicted_values[a, i] = value.detach().cpu().item()
            if i > 0:  # TD error only makes sense from the second step
                action_predicted_tds[a, i-1] = (max(rews) + gamma * value - last_value).abs()
            
            last_value = value.detach().cpu().item()

    if mode == 'reward':
        best_score = max([np.sum(action_predicted_rews[a,:]) for a in range(num_actions)])
        candidate_actions = [a for a in range(num_actions) if np.sum(action_predicted_rews[a,:]) == best_score]
    elif mode == 'value':
        best_score = max([np.sum(action_predicted_values[a,:]) for a in range(num_actions)])
        candidate_actions = [a for a in range(num_actions) if np.sum(action_predicted_values[a,:]) == best_score]
    elif mode == 'td':
        best_score = min([np.sum(action_predicted_tds[a,:]) for a in range(num_actions)]) # MINIMISE the td error for best action
        candidate_actions = [a for a in range(num_actions) if np.sum(action_predicted_tds[a,:]) == best_score]

    best_action = torch.tensor([np.random.choice(np.array(candidate_actions))])  # Randomly select one of the best actions
    print(f"Best action selected: {best_action} from {candidate_actions}")

    return best_action, np.array(candidate_actions), action_predicted_rews, wm_predicted_obs
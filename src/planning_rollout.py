import math
from hydra.utils import instantiate
import numpy as np
import torch
from torch.distributions.categorical import Categorical


def multistep_planning(agent, world_model_env, num_actions, cfg):
    """
    Perform multistep planning using the world model environment.
    Pads all aborted or high-entropy trajectories to avoid indexing errors.
    """
    assert cfg.evaluation.planning_mode in ['reward', 'value', 'td'], "Invalid planning mode"

    # Ensure at least 2 steps so indexing downstream never fails
    min_steps = max(cfg.evaluation.planning_steps, 2)

    # Logging arrays
    action_predicted_rews = np.zeros((num_actions, min_steps), dtype=np.float32)
    action_predicted_values = np.zeros((num_actions, min_steps), dtype=np.float32)
    action_predicted_tds = np.zeros((num_actions, min_steps - 1), dtype=np.float32)
    wm_predicted_obs = [[] for _ in range(num_actions)]
    depths = np.zeros(num_actions)
    latest_entropies = np.zeros(num_actions)

    # Backup buffers
    obs_buffer_base = world_model_env.obs_buffer.clone()
    act_buffer_base = world_model_env.act_buffer.clone()
    wm_hx_base = world_model_env.hx_rew_end.clone()
    wm_cx_base = world_model_env.cx_rew_end.clone()
    hx_base = agent.hx.clone()
    cx_base = agent.cx.clone()

    max_depth = 0
    candidate_actions = []

    while not candidate_actions:
        # --- reset per-attempt logging so failed attempts don't accumulate entries ---
        action_predicted_rews.fill(0)
        action_predicted_values.fill(0)
        action_predicted_tds.fill(0)
        wm_predicted_obs = [[] for _ in range(num_actions)]

        rollout_valids = [False] * num_actions

        for a in range(num_actions):
            # Reset buffers/hiddens
            obs_buffer = obs_buffer_base.clone()
            act_buffer = act_buffer_base.clone()
            wm_hx, wm_cx = wm_hx_base.clone(), wm_cx_base.clone()
            agent_hx, agent_cx = hx_base.clone(), cx_base.clone()
            act_buffer[:, -1] = a

            collected = 0
            rollout_valid = True
            last_value = 0.0

            for i in range(cfg.evaluation.planning_steps):
                rews = []

                # Sample next obs
                next_obs, _ = world_model_env.sampler.sample(obs_buffer, act_buffer)

                # Reward-end model
                logits_rew, _, (wm_hx, wm_cx) = world_model_env.rew_end_model.predict_rew_end(
                    obs_buffer[:, -1:], act_buffer[:, -1:], next_obs.unsqueeze(1), (wm_hx, wm_cx)
                )
                for _ in range(10):
                    rews.append((Categorical(logits=logits_rew).sample().squeeze(1) - 1.0).item())

                # Roll buffers
                obs_buffer = obs_buffer.roll(-1, dims=1)
                act_buffer = act_buffer.roll(-1, dims=1)
                obs_buffer[:, -1] = next_obs

                # Actor-critic
                logits, value, (agent_hx, agent_cx) = agent.actor_critic.predict_act_value(next_obs, (agent_hx, agent_cx))
                dist = Categorical(logits=logits)
                entropy = dist.entropy().detach().cpu().item() / math.log(2)
                latest_entropies[a] = entropy

                need_inner = (
                    cfg.evaluation.inner_planning_steps != 0 and
                    depths[a] < max_depth and
                    max_depth < cfg.evaluation.planning_depth
                )

                if entropy > cfg.evaluation.entropy_threshold:
                    if need_inner:
                        inner_update = inner_planning(
                            agent, world_model_env, num_actions,
                            obs_buffer, act_buffer,
                            wm_hx.clone(), wm_cx.clone(),
                            agent_hx.clone(), agent_cx.clone(),
                            cfg, depths[a]+1, max_depth=max_depth,
                            remaining_steps=cfg.evaluation.planning_steps - i - 1
                        )
                        if inner_update[1] < cfg.evaluation.entropy_threshold:
                            act_buffer[:, -1], latest_entropies[a], depths[a] = inner_update
                        else: # Inner selection action was too high entropy
                            rollout_valid = False
                            break
                    else:
                        rollout_valid = False
                        break
                else:
                    act_buffer[:, -1] = dist.sample()

                # Logging
                wm_predicted_obs[a].append(next_obs.squeeze())
                action_predicted_rews[a, i] = max(rews)
                action_predicted_values[a, i] = value.detach().cpu().item()
                if i > 0:
                    action_predicted_tds[a, i-1] = (
                        max(rews) + cfg.actor_critic.actor_critic_loss.gamma * value - last_value
                    ).abs()
                last_value = value.detach().cpu().item()
                collected += 1

            # Pad any remaining steps if rollout terminated early
            if collected < cfg.evaluation.planning_steps:
                remaining = cfg.evaluation.planning_steps - collected
                dummy_obs = torch.zeros_like(next_obs.squeeze())
                wm_predicted_obs[a].extend([dummy_obs.clone() for _ in range(remaining)])
                for j in range(collected, cfg.evaluation.planning_steps):
                    action_predicted_rews[a, j] = 0.0
                    action_predicted_values[a, j] = 0.0
                    if j < cfg.evaluation.planning_steps - 1:
                        action_predicted_tds[a, j] = 0.0

            rollout_valids[a] = rollout_valid

        # If all rollouts failed, increase depth and retry
        if not any(rollout_valids):
            if max_depth >= cfg.evaluation.planning_depth:
                # Fallback: pick lowest-entropy action
                entropies = {a: latest_entropies[a] for a in range(num_actions)}
                best_action = torch.tensor([min(entropies, key=entropies.get)])
                print(f"⚠️ All rollouts failed at max depth. "
                    f"Falling back to lowest-entropy action {best_action.item()} "
                    f"(entropy={entropies[best_action.item()]:.3f})")
                return best_action  # or break out of loop safely
            else:
                max_depth += 1
                continue

        # Select candidate actions among valid rollouts
        valid_idxs = [a for a, ok in enumerate(rollout_valids) if ok]

        if cfg.evaluation.planning_mode == 'reward':
            scores = {a: float(np.sum(action_predicted_rews[a, :])) for a in valid_idxs}
            best = max(scores.values())
            candidate_actions = [a for a, s in scores.items() if s == best]
        elif cfg.evaluation.planning_mode == 'value':
            scores = {a: float(np.sum(action_predicted_values[a, :])) for a in valid_idxs}
            best = max(scores.values())
            candidate_actions = [a for a, s in scores.items() if s == best]
        elif cfg.evaluation.planning_mode == 'td':
            scores = {a: float(np.sum(action_predicted_tds[a, :])) for a in valid_idxs}
            best = min(scores.values())
            candidate_actions = [a for a, s in scores.items() if s == best]

    best_action = torch.tensor([np.random.choice(np.array(candidate_actions))])
    print(f"Best action selected: {best_action} from {candidate_actions}")

    return (best_action, np.array(candidate_actions),
            action_predicted_rews, wm_predicted_obs,
            latest_entropies[best_action.item()],
            depths[best_action.item()])


def inner_planning(agent, world_model_env, num_actions, obs_buffer, act_buffer, wm_hx, wm_cx, agent_hx, agent_cx,
                   cfg, depth, max_depth, remaining_steps):
    """
    Perform planning *within* a rollout when entropy is high.
    Returns the selected next action based on imagination from current buffer state.
    This avoids overwriting or re-cloning buffers from real env.
    """
    ############ TD MODE IS NOT SUPPORTED YET ############
    action_predicted_rews = np.zeros((num_actions, cfg.evaluation.inner_planning_steps))
    action_predicted_values = np.zeros((num_actions, cfg.evaluation.inner_planning_steps))
    action_predicted_tds = np.zeros((num_actions, cfg.evaluation.inner_planning_steps - 1))
    initial_depth = depth # Change this for each action
    latest_entropies = np.zeros(num_actions)  # Store latest entropies for each action

    for a in range(num_actions):
        # Clone buffers at current internal planning point
        obs_buf = obs_buffer.clone()
        act_buf = act_buffer.clone()
        wm_hx_a = wm_hx.clone()
        wm_cx_a = wm_cx.clone()
        agent_hx_a = agent_hx.clone()
        agent_cx_a = agent_cx.clone()
        depth = initial_depth  # Reset depth for each action

        act_buffer[:, -1] = a  # candidate action

        for i in range(remaining_steps):
            rews = [] # Store rewards for multiple samples
            next_obs, _ = world_model_env.sampler.sample(obs_buf, act_buf)
            logits_rew, _, (wm_hx_a, wm_cx_a) = world_model_env.rew_end_model.predict_rew_end(
                obs_buf[:, -1:], act_buf[:, -1:], next_obs.unsqueeze(1),
                (wm_hx_a, wm_cx_a)
            )
            for _ in range (10):  # Sample multiple times to get a good estimate
                rews.append((Categorical(logits=logits_rew).sample().squeeze(1) - 1.0).item())
                    # in {-1, 0, 1}

            obs_buf = obs_buf.roll(-1, dims=1)
            act_buf = act_buf.roll(-1, dims=1)
            obs_buf[:, -1] = next_obs  # predicted next obs

            # Get actor-critic to choose next action
            logits, value, (agent_hx_a, agent_cx_a) = agent.actor_critic.predict_act_value(
                next_obs, (agent_hx_a, agent_cx_a) # Squeeze obs to fit
            )
            dist = Categorical(logits=logits)
            entropy = dist.entropy().detach().cpu().item() / math.log(2)
            latest_entropies[a] = entropy  # Store latest entropy for this action

            if entropy > cfg.evaluation.entropy_threshold and depth < max_depth and remaining_steps - i - 1 > 0:
                act_buf[:, -1], latest_entropies[a], depth = inner_planning(agent, world_model_env, num_actions, obs_buf, act_buf, 
                                                        wm_hx_a.clone(), wm_cx_a.clone(), agent_hx_a.clone(), agent_cx_a.clone(), 
                                                        cfg, depth + 1, max_depth=max_depth, 
                                                        remaining_steps=remaining_steps - i - 1)
            else:
                act_buf[:, -1] = dist.sample() # If this is too large entropy, main fn will probably discard action anyway

            action_predicted_rews[a, i] = max(rews)
            action_predicted_values[a, i] = value.detach().cpu().item()

    if cfg.evaluation.planning_mode == 'reward':
        best_score = max([np.sum(action_predicted_rews[a, :]) for a in range(num_actions)])
        candidate_actions = [a for a in range(num_actions) if np.sum(action_predicted_rews[a, :]) == best_score]
    elif cfg.evaluation.planning_mode == 'value':
        best_score = max([np.sum(action_predicted_values[a, :]) for a in range(num_actions)])
        candidate_actions = [a for a in range(num_actions) if np.sum(action_predicted_values[a, :]) == best_score]
    elif cfg.evaluation.planning_mode == 'td':
        best_score = min([np.sum(action_predicted_tds[a, :]) for a in range(num_actions)])
        candidate_actions = [a for a in range(num_actions) if np.sum(action_predicted_tds[a, :]) == best_score]

    best_action = torch.tensor([np.random.choice(np.array(candidate_actions))])  # Randomly select one of the best actions

    return torch.tensor([np.random.choice(np.array(candidate_actions))]), latest_entropies[best_action.item()], depth
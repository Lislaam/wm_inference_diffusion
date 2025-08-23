import math
from hydra.utils import instantiate
import numpy as np
import torch
from torch.distributions.categorical import Categorical


def _expected_reward_from_logits(logits_rew: torch.Tensor) -> float:
    """
    Your sampling did: Categorical(logits).sample() in {0,1,2} then -1 => {-1,0,1}.
    Compute E[reward] analytically to avoid 10 samples/step.
    """
    # probs shape: [B, 3] or [3]; allow either
    probs = torch.softmax(logits_rew, dim=-1)
    classes = torch.tensor([-1.0, 0.0, 1.0], device=probs.device, dtype=probs.dtype)
    exp_rew = (probs * classes).sum(dim=-1)
    # If batched, average; if scalar, return scalar
    return float(exp_rew.mean().item())


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
                # 1) next obs
                next_obs, _ = world_model_env.sampler.sample(obs_buffer, act_buffer)
                logits_rew, _, (wm_hx, wm_cx) = world_model_env.rew_end_model.predict_rew_end(
                    obs_buffer[:, -1:], act_buffer[:, -1:], next_obs.unsqueeze(1), (wm_hx, wm_cx)
                )

                # FAST expected reward (instead of 10 samples)
                exp_rew = _expected_reward_from_logits(logits_rew)

                # roll buffers
                obs_buffer = obs_buffer.roll(-1, dims=1)
                act_buffer = act_buffer.roll(-1, dims=1)
                obs_buffer[:, -1] = next_obs

                # 2) policy/value
                logits, value, (agent_hx, agent_cx) = agent.actor_critic.predict_act_value(next_obs, (agent_hx, agent_cx))
                dist = Categorical(logits=logits)
                entropy = dist.entropy().detach().cpu().item() / math.log(2)
                latest_entropies[a] = entropy

                remaining_steps = cfg.evaluation.planning_steps - (i + 1)
                can_inner = (
                    cfg.evaluation.inner_planning_steps != 0 and
                    depths[a] < max_depth and
                    max_depth < cfg.evaluation.planning_depth and
                    remaining_steps > 0  # only inner-plan if there's horizon left
                )

                if entropy > cfg.evaluation.entropy_threshold:
                    if can_inner:
                        act_sel, ent_sel, new_depth = inner_planning(
                            agent, world_model_env, num_actions,
                            obs_buffer, act_buffer,
                            wm_hx.clone(), wm_cx.clone(),
                            agent_hx.clone(), agent_cx.clone(),
                            cfg, depths[a] + 1, max_depth=max_depth,
                            remaining_steps=remaining_steps
                        )
                        # if even the inner plan can’t get below threshold, abort this action
                        if ent_sel >= cfg.evaluation.entropy_threshold:
                            rollout_valid = False
                            break
                        act_buffer[:, -1] = act_sel  # OK to adopt inner-selected action
                        depths[a] = new_depth
                        latest_entropies[a] = ent_sel
                    else:
                        # No useful inner-planning possible (no horizon left or depth budget) ⇒ abort
                        rollout_valid = False
                        break
                else:
                    act_buffer[:, -1] = dist.sample()

                # --- logging ---
                wm_predicted_obs[a].append(next_obs.squeeze())
                action_predicted_rews[a, i] = exp_rew
                action_predicted_values[a, i] = float(value.detach().cpu().item())
                if i > 0:
                    td = (exp_rew + cfg.actor_critic.actor_critic_loss.gamma * value - last_value).abs()
                    action_predicted_tds[a, i-1] = float(td.detach().cpu().item() if torch.is_tensor(td) else td)
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


@torch.inference_mode()
def inner_planning(agent, world_model_env, num_actions,
                   obs_buffer, act_buffer, wm_hx, wm_cx, agent_hx, agent_cx,
                   cfg, depth, max_depth, remaining_steps=None):
    """
    Pruned inner planning. Only useful if there is remaining horizon.
    Returns: (chosen_action_tensor, latest_entropy_for_chosen, new_depth)
    """
    assert cfg.evaluation.planning_mode in ['reward', 'value', 'td']

    # If caller doesn't pass remaining_steps, assume full inner horizon but cap by outer remainder.
    if remaining_steps is None:
        remaining_steps = cfg.evaluation.inner_planning_steps
    inner_steps_eff = min(cfg.evaluation.inner_planning_steps, max(0, int(remaining_steps)))

    # If no horizon left, don't expand — report high entropy so caller treats as invalid.
    if inner_steps_eff <= 0 or depth >= cfg.evaluation.planning_depth:
        # Return a no-op action with very high entropy to signal "invalid"
        return torch.tensor([0], device=act_buffer.device), float('inf'), depth

    # Pre-alloc (small) for inner horizon
    min_steps = max(inner_steps_eff, 2)
    action_predicted_rews = np.zeros((num_actions, min_steps), dtype=np.float32)
    action_predicted_values = np.zeros((num_actions, min_steps), dtype=np.float32)
    action_predicted_tds = np.zeros((num_actions, min_steps - 1), dtype=np.float32)
    latest_entropies = np.zeros(num_actions, dtype=np.float32)

    local_max_depth = depth
    candidate_actions = []

    while not candidate_actions:
        rollout_valids = [False] * num_actions

        for a in range(num_actions):
            # local clones
            obs_buf = obs_buffer.clone()
            act_buf = act_buffer.clone()
            wm_hx_a, wm_cx_a = wm_hx.clone(), wm_cx.clone()
            agent_hx_a, agent_cx_a = agent_hx.clone(), agent_cx.clone()
            act_buf[:, -1] = a

            collected = 0
            rollout_valid = True
            last_value = 0.0

            for i in range(inner_steps_eff):
                # 1) next obs
                next_obs, _ = world_model_env.sampler.sample(obs_buf, act_buf)
                logits_rew, _, (wm_hx_a, wm_cx_a) = world_model_env.rew_end_model.predict_rew_end(
                    obs_buf[:, -1:], act_buf[:, -1:], next_obs.unsqueeze(1), (wm_hx_a, wm_cx_a)
                )

                # === FAST: expected reward instead of 10x sampling ===
                exp_rew = _expected_reward_from_logits(logits_rew)
                # If you insist on sampling, replace the line above with:
                # exp_rew = np.mean([(Categorical(logits=logits_rew).sample().squeeze(1).item() - 1.0) for _ in range(10)])

                # slide local buffers
                obs_buf = obs_buf.roll(-1, dims=1)
                act_buf = act_buf.roll(-1, dims=1)
                obs_buf[:, -1] = next_obs

                # 2) policy/value
                logits, value, (agent_hx_a, agent_cx_a) = agent.actor_critic.predict_act_value(next_obs, (agent_hx_a, agent_cx_a))
                dist = Categorical(logits=logits)
                entropy = dist.entropy().detach().cpu().item() / math.log(2)
                latest_entropies[a] = entropy

                # Can we go deeper from here?
                can_recurse = (local_max_depth < max_depth) and (local_max_depth < cfg.evaluation.planning_depth) \
                              and (i < inner_steps_eff - 1)  # only recurse if there is further horizon

                if entropy > cfg.evaluation.entropy_threshold:
                    if can_recurse:
                        # recurse with horizon reduced by the step we've just taken
                        sub_action, sub_ent, new_depth = inner_planning(
                            agent, world_model_env, num_actions,
                            obs_buf.clone(), act_buf.clone(),
                            wm_hx_a.clone(), wm_cx_a.clone(),
                            agent_hx_a.clone(), agent_cx_a.clone(),
                            cfg, local_max_depth + 1, max_depth,
                            remaining_steps=inner_steps_eff - (i + 1)
                        )
                        latest_entropies[a] = float(sub_ent)
                        # Treat this branch as completed (we used recursion to resolve)
                        action_predicted_rews[a, i] = exp_rew
                        action_predicted_values[a, i] = float(value.detach().cpu().item())
                        if i > 0:
                            td = (exp_rew + cfg.actor_critic.actor_critic_loss.gamma * value - last_value).abs()
                            action_predicted_tds[a, i-1] = float(td.detach().cpu().item() if torch.is_tensor(td) else td)
                        collected += 1
                        rollout_valid = (sub_ent < cfg.evaluation.entropy_threshold)
                        break
                    else:
                        # Depth constraint prevents recursion (or no horizon left) — abort
                        rollout_valid = False
                        break
                else:
                    act_buf[:, -1] = dist.sample()

                # logging this step
                action_predicted_rews[a, i] = exp_rew
                action_predicted_values[a, i] = float(value.detach().cpu().item())
                if i > 0:
                    td = (exp_rew + cfg.actor_critic.actor_critic_loss.gamma * value - last_value).abs()
                    action_predicted_tds[a, i-1] = float(td.detach().cpu().item() if torch.is_tensor(td) else td)
                last_value = value.detach().cpu().item()
                collected += 1

            rollout_valids[a] = rollout_valid

            # pad remaining (small inner arrays)
            if collected < min_steps:
                action_predicted_rews[a, collected:] = 0.0
                action_predicted_values[a, collected:] = 0.0
                if collected < (min_steps - 1):
                    action_predicted_tds[a, collected:] = 0.0

        # If all invalid at this local depth, try to open depth a bit (but not above global limits)
        if not any(rollout_valids):
            if local_max_depth >= max_depth or local_max_depth >= cfg.evaluation.planning_depth:
                # fallback: lowest-entropy action
                chosen = int(np.argmin(latest_entropies))
                return torch.tensor([chosen], device=act_buffer.device), float(latest_entropies[chosen]), local_max_depth
            local_max_depth += 1
            continue

        # choose among valid only
        valid = [i for i, ok in enumerate(rollout_valids) if ok]
        if cfg.evaluation.planning_mode == 'reward':
            vals = {a: float(action_predicted_rews[a, :].sum()) for a in valid}
            best = max(vals.values())
            candidate_actions = [a for a, s in vals.items() if s == best]
        elif cfg.evaluation.planning_mode == 'value':
            vals = {a: float(action_predicted_values[a, :].sum()) for a in valid}
            best = max(vals.values())
            candidate_actions = [a for a, s in vals.items() if s == best]
        else:  # 'td'
            vals = {a: float(action_predicted_tds[a, :].sum()) for a in valid}
            best = min(vals.values())
            candidate_actions = [a for a, s in vals.items() if s == best]

    chosen = int(np.random.choice(np.array(candidate_actions)))
    return torch.tensor([chosen], device=act_buffer.device), float(latest_entropies[chosen]), local_max_depth
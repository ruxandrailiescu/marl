"""PPO core: GAE and the clipped-surrogate update (CleanRL formulation)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from nhmrs_marl import metrics


def compute_gae(rewards, values, dones, next_value, next_done, gamma, gae_lambda):
    """Generalized advantage estimation over a (T, B) rollout.

    ``dones[t]`` marks that the observation at step t starts a new episode, so
    step t-1 must not bootstrap from it. Truncations are handled by the caller
    folding gamma * V(s_real_next) into the reward of the truncated step.
    """
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = 0
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_non_terminal = 1.0 - next_done
            next_values = next_value
        else:
            next_non_terminal = 1.0 - dones[t + 1]
            next_values = values[t + 1]
        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
    return advantages


def ppo_update(agent, optimizer, batch: dict, args, rng: np.random.Generator, ret_rms=None) -> dict:
    """Run ``update_epochs`` of minibatch PPO on a flattened rollout.

    ``batch`` holds flat tensors: obs, critic_in, actions, log_probs,
    advantages, returns (unnormalized) and values (critic's own scale).
    Returns the last minibatch's losses plus mean pre-clip gradient norms.
    """
    actor_params, critic_params = agent.actor_params(), agent.critic_params()
    actor_gnorm_sum = critic_gnorm_sum = 0.0
    n_grad_steps = 0
    clipfracs = []

    b_inds = np.arange(args.batch_size)
    for _ in range(args.update_epochs):
        rng.shuffle(b_inds)
        for start in range(0, args.batch_size, args.minibatch_size):
            mb = b_inds[start:start + args.minibatch_size]
            _, new_log_prob, entropy, new_value = agent.get_action_and_value(
                batch["obs"][mb], batch["critic_in"][mb], batch["actions"][mb])
            log_ratio = new_log_prob - batch["log_probs"][mb]
            ratio = log_ratio.exp()

            with torch.no_grad():
                approx_kl = ((ratio - 1) - log_ratio).mean()
                clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

            mb_adv = batch["advantages"][mb]
            if args.norm_adv:
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

            pg_loss1 = mb_adv * ratio
            pg_loss2 = mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = -torch.min(pg_loss1, pg_loss2).mean()

            new_value = new_value.view(-1)
            mb_returns = batch["returns"][mb]
            if ret_rms is not None:
                mb_returns = ret_rms.normalize(mb_returns)
            if args.clip_vloss:
                mb_values = batch["values"][mb]
                v_loss_unclipped = (new_value - mb_returns) ** 2
                v_clipped = mb_values + torch.clamp(
                    new_value - mb_values, -args.clip_coef, args.clip_coef)
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
            else:
                v_loss = 0.5 * ((new_value - mb_returns) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

            optimizer.zero_grad()
            loss.backward()
            actor_gnorm_sum += metrics.grad_norm(actor_params)   # pre-clip
            critic_gnorm_sum += metrics.grad_norm(critic_params)
            n_grad_steps += 1
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

        if args.target_kl is not None and approx_kl > args.target_kl:
            break

    return {
        "policy_loss": pg_loss.item(),
        "value_loss": v_loss.item(),
        "entropy": entropy_loss.item(),
        "approx_kl": approx_kl.item(),
        "clipfrac": float(np.mean(clipfracs)),
        "actor_grad_norm": actor_gnorm_sum / max(1, n_grad_steps),
        "critic_grad_norm": critic_gnorm_sum / max(1, n_grad_steps),
    }

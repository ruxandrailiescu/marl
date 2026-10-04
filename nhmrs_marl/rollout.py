"""Rollout storage and collection for parameter-shared multi-agent PPO.

Truncation handling: the env never terminates early; it only truncates at
``max_steps``. Truncation is not a real terminal, so the value must be
bootstrapped, not zeroed. At a truncated step we add gamma * V(real_next)
into the reward and mark the next step done, which makes ordinary GAE produce
the correct bootstrap while still cutting advantage propagation across the
episode boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import torch

from nhmrs_marl.env import EnvBatch


@dataclass
class RolloutBuffer:
    """(num_steps, B, .) tensors for one rollout of B = num_envs * n_agents actors."""
    obs: torch.Tensor
    critic_in: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor

    @classmethod
    def zeros(cls, num_steps, num_actors, obs_dim, critic_in_dim, action_dim, device):
        shape = (num_steps, num_actors)
        return cls(
            obs=torch.zeros(shape + (obs_dim,), device=device),
            critic_in=torch.zeros(shape + (critic_in_dim,), device=device),
            actions=torch.zeros(shape + (action_dim,), device=device),
            log_probs=torch.zeros(shape, device=device),
            rewards=torch.zeros(shape, device=device),
            dones=torch.zeros(shape, device=device),
            values=torch.zeros(shape, device=device),
        )

    def flatten(self) -> dict:
        """Merge the step and actor axes: (T, B, .) -> (T*B, .)."""
        out = {}
        for f in fields(self):
            x = getattr(self, f.name)
            out[f.name] = x.reshape((-1,) + x.shape[2:])
        return out


class RolloutCollector:
    """Steps an ``EnvBatch`` with the current policy, carrying state across rollouts.

    ``critic_input`` maps (B, obs_dim) observations to the critic's input
    (identity for IPPO, the global state for MAPPO).
    """

    def __init__(self, envs: EnvBatch, agent, critic_input, gamma: float, ret_rms, device):
        self.envs = envs
        self.agent = agent
        self.critic_input = critic_input
        self.gamma = gamma
        self.ret_rms = ret_rms
        self.device = device
        self.action_low = torch.tensor(envs.action_low, dtype=torch.float32, device=device)
        self.action_high = torch.tensor(envs.action_high, dtype=torch.float32, device=device)

        self.global_step = 0
        self.ep_return = np.zeros(envs.num_actors, dtype=np.float64)  # raw reward, for logging
        self._set_next(envs.observe(), torch.zeros(envs.num_actors, device=device))

    def _tensor(self, x) -> torch.Tensor:
        return torch.tensor(x, dtype=torch.float32, device=self.device)

    def _set_next(self, obs_np: np.ndarray, done: torch.Tensor):
        self.next_obs = self._tensor(obs_np)
        self.next_critic_in = self._tensor(self.critic_input(obs_np))
        self.next_done = done

    @torch.no_grad()
    def value(self, critic_in: torch.Tensor) -> torch.Tensor:
        """Critic value on the return scale (denormalized if returns are normalized)."""
        v = self.agent.get_value(critic_in).flatten()
        return self.ret_rms.denormalize(v) if self.ret_rms is not None else v

    def collect(self, buf: RolloutBuffer) -> list[tuple[int, float]]:
        """Fill ``buf`` with one rollout; return completed (global_step, mean per-agent return)."""
        envs = self.envs
        episodes = []
        for step in range(buf.obs.shape[0]):
            self.global_step += envs.num_envs
            buf.obs[step] = self.next_obs
            buf.critic_in[step] = self.next_critic_in
            buf.dones[step] = self.next_done

            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(
                    self.next_obs, self.next_critic_in)
                buf.values[step] = value.flatten()
            buf.actions[step] = action
            buf.log_probs[step] = logprob

            clipped = torch.max(torch.min(action, self.action_high), self.action_low).cpu().numpy()
            reward_B, trunc_B, real_next, truncated_envs = envs.step(clipped)

            # accumulate raw episodic return before the bootstrap is folded in
            self.ep_return += reward_B
            buf.rewards[step] = self._tensor(reward_B)

            if truncated_envs:
                boot_v = self.value(self._tensor(self.critic_input(real_next)))
                buf.rewards[step] = buf.rewards[step] + self.gamma * boot_v * self._tensor(trunc_B)
                for i in truncated_envs:
                    rows = envs.rows(i)
                    episodes.append((self.global_step, float(self.ep_return[rows].mean())))
                    self.ep_return[rows] = 0.0
                    envs.reset_env(i)

            self._set_next(envs.observe(), self._tensor(trunc_B))
        return episodes

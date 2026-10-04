"""Shared actor-critic used by both IPPO and MAPPO.

The actor always acts on an agent's local observation. The critic's input is
the only structural difference between the algorithms: the same local obs for
IPPO (``critic_in_dim == obs_dim``) or the centralized global state for MAPPO
(``critic_in_dim == n_agents * obs_dim``).

Attribute names (``critic``, ``actor_mean``, ``actor_logstd``) are the
checkpoint format; renaming them breaks loading of existing ``agent_*.pt``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, critic_in_dim: int, action_dim: int):
        super().__init__()
        # built before the actor: the init order fixes the seeded weights
        self.critic = nn.Sequential(
            layer_init(nn.Linear(critic_in_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def actor_params(self) -> list[nn.Parameter]:
        return list(self.actor_mean.parameters()) + [self.actor_logstd]

    def critic_params(self) -> list[nn.Parameter]:
        return list(self.critic.parameters())

    def get_value(self, critic_in):
        return self.critic(critic_in)

    def get_action_and_value(self, obs, critic_in, action=None):
        mean = self.actor_mean(obs)
        logstd = self.actor_logstd.expand_as(mean)
        dist = Normal(mean, torch.exp(logstd))
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action).sum(1)
        entropy = dist.entropy().sum(1)
        return action, logprob, entropy, self.critic(critic_in)

    @classmethod
    def from_checkpoint(cls, path, obs_dim: int, action_dim: int, device="cpu") -> "ActorCritic":
        """Load an IPPO or MAPPO checkpoint; the critic width is read from the file."""
        state_dict = torch.load(path, map_location=device)
        critic_in_dim = state_dict["critic.0.weight"].shape[1]
        net = cls(obs_dim, critic_in_dim, action_dim)
        net.load_state_dict(state_dict)
        return net.to(device)

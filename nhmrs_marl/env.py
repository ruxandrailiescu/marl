"""NHMRS ``simple_assignment_v0`` construction and dict <-> array adapters.

PettingZoo returns per-agent dicts; we stack them into arrays in the fixed
``env.possible_agents`` order. Observations are read straight from
``env.world`` via ``scenario.observation`` rather than from step()'s returned
dict: the env clears ``env.agents`` (and returns an empty obs dict) on the
truncation step, but we still need the real post-step observation to
bootstrap the value there.
"""

from __future__ import annotations

import numpy as np
from nhmrs import simple_assignment_v0
from nhmrs.simple_assignment.simple_assignment import Scenario


def env_kwargs(args) -> dict:
    """The env-shape fields of a training config, as ``make_env`` kwargs."""
    return dict(reward_mode=args.reward_mode, n_agents=args.n_agents,
                n_landmarks=args.n_landmarks, max_steps=args.max_steps)


def make_env(reward_mode: str, n_agents: int, n_landmarks: int, max_steps: int,
             render_mode=None):
    # observe_relative=True is part of the policy's input contract: training,
    # eval and inference must all build the env this way.
    scenario = Scenario(reward_mode=reward_mode, observe_relative=True)
    return simple_assignment_v0.env(
        scenario=scenario,
        render_mode=render_mode,
        max_steps=max_steps,
        n_agents=n_agents,
        n_landmarks=n_landmarks,
    )


def world_obs_array(env, agent_order) -> np.ndarray:
    """Current per-agent observations -> (N, obs_dim)."""
    name_to_agent = {ag.name: ag for ag in env.world.agents}
    return np.stack([
        env.scenario.observation(name_to_agent[name], env.world)
        for name in agent_order
    ]).astype(np.float32)


def array_to_action_dict(a_NA: np.ndarray, agent_order) -> dict:
    """(N, action_dim) array -> {agent_name: action_vec}"""
    return {name: a_NA[i] for i, name in enumerate(agent_order)}


def rewards_to_array(reward_dict, agent_order) -> np.ndarray:
    """{agent_name: float} -> (N,) in fixed order"""
    return np.array([reward_dict[name] for name in agent_order], dtype=np.float32)


def global_state(obs_B: np.ndarray, n_agents: int) -> np.ndarray:
    """(E*N, obs_dim) per-agent obs -> (E*N, N*obs_dim) centralized state.

    Each env's global state is the concatenation of its N local observations,
    and every agent in that env is handed the same state row.
    """
    per_env = obs_B.reshape(-1, n_agents * obs_B.shape[1])   # (E, N*obs_dim)
    return np.repeat(per_env, n_agents, axis=0)


class EnvBatch:
    """``num_envs`` independent env copies with all agents stacked on one axis.

    Arrays are (B, .) with B = num_envs * n_agents; env ``i`` owns rows
    ``[i*N, (i+1)*N)``. With parameter sharing this makes N agents in E envs
    look like a vector env of B actors.
    """

    def __init__(self, num_envs: int, seed: int, **env_kwargs):
        self.num_envs = num_envs
        self.envs = [make_env(**env_kwargs) for _ in range(num_envs)]
        for i, env in enumerate(self.envs):
            env.reset(seed=seed + i)
        self.agent_order = self.envs[0].possible_agents[:]
        self.n_agents = len(self.agent_order)
        self.num_actors = num_envs * self.n_agents
        space = self.envs[0].action_space(self.agent_order[0])
        self.action_low, self.action_high = space.low, space.high
        self.obs_dim = self.envs[0].observation_space(self.agent_order[0]).shape[0]
        self.action_dim = space.shape[0]

    def rows(self, i: int) -> slice:
        """Rows of the (B, .) arrays that belong to env ``i``."""
        return slice(i * self.n_agents, (i + 1) * self.n_agents)

    def observe(self) -> np.ndarray:
        """(B, obs_dim) current observations."""
        return np.concatenate([world_obs_array(e, self.agent_order) for e in self.envs], axis=0)

    def step(self, actions_B: np.ndarray):
        """Step every env with already-clipped (B, action_dim) actions.

        Returns per-actor rewards (B,), per-actor truncation flags (B,), the
        post-step observations (B, obs_dim) and the indices of truncated envs.
        Truncated envs are NOT reset here; call ``reset_env`` once their
        post-step observation has been used for bootstrapping.
        """
        rewards = np.zeros(self.num_actors, dtype=np.float32)
        truncs = np.zeros(self.num_actors, dtype=np.float32)
        next_obs = np.zeros((self.num_actors, self.obs_dim), dtype=np.float32)
        truncated_envs = []
        for i, env in enumerate(self.envs):
            rows = self.rows(i)
            _, rew, _, truncated, _ = env.step(array_to_action_dict(actions_B[rows], self.agent_order))
            rewards[rows] = rewards_to_array(rew, self.agent_order)
            next_obs[rows] = world_obs_array(env, self.agent_order)
            if all(truncated.values()):
                truncs[rows] = 1.0
                truncated_envs.append(i)
        return rewards, truncs, next_obs, truncated_envs

    def reset_env(self, i: int):
        self.envs[i].reset()

    def close(self):
        for env in self.envs:
            env.close()

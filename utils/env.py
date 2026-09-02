import numpy as np

from classes.cli_args import Args

from nhmrs import simple_assignment_v0
from nhmrs.simple_assignment.simple_assignment import Scenario


def make_env(args: Args):
    scenario = Scenario(reward_mode=args.reward_mode, observe_relative=True)
    return simple_assignment_v0.env(
        scenario=scenario,
        render_mode=None,
        max_steps=args.max_steps,
        n_agents=args.n_agents,
        n_landmarks=args.n_landmarks,
    )


def world_obs_array(env, agent_order) -> np.ndarray:
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


def collect_obs(agent_order, envs):
    return np.concatenate([world_obs_array(e, agent_order) for e in envs], axis=0)  # (B, obs_dim)
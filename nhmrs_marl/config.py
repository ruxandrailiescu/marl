"""Training configuration: one dataclass, exposed 1:1 as ``--kebab-case`` CLI flags."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

ALGOS = ("ippo", "mappo")
# nhmrs Scenario silently treats any other reward_mode as 'balanced'
REWARD_MODES = ("spread", "simple", "balanced", "patrol")


@dataclass
class Args:
    # algorithm
    algo: str = "ippo"            # 'ippo' (local critic) | 'mappo' (centralized critic)

    # experiment / reproducibility
    exp_name: str = ""            # run-name prefix (defaults to algo)
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True

    # environment
    reward_mode: str = "simple"   # 'spread' | 'simple' | 'balanced' | 'patrol'
    n_agents: int = 3
    n_landmarks: int = 3
    max_steps: int = 100
    num_envs: int = 25            # independent env copies (batch = num_envs * n_agents)

    # PPO
    total_timesteps: int = 2_000_000
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    num_steps: int = 100          # rollout length per env (defaults to one episode)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 2
    update_epochs: int = 10
    norm_adv: bool = True
    norm_returns: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 10.0
    target_kl: float | None = None

    # logging / eval / plotting
    eval_points: int = 40         # target number of evenly spaced greedy evals
    n_eval_episodes: int = 5      # greedy episodes averaged per eval point
    plot_interval: int = 10       # iterations between PNG refreshes
    eval_radius: float = 0.15     # "landmark covered" distance threshold

    # seed-aggregation outputs
    results_dir: str = ""         # defaults to results_<algo>
    config_name: str = ""         # subdir under results_dir (defaults to reward_mode)

    # derived (filled by finalize)
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


DERIVED = ("batch_size", "minibatch_size", "num_iterations")


def _str2bool(x) -> bool:
    return str(x).lower() in ("1", "true", "yes")


def parse_args(argv=None) -> Args:
    a = Args()
    p = argparse.ArgumentParser(description="IPPO / MAPPO for NHMRS simple_assignment_v0")
    for f, default in vars(a).items():
        if f in DERIVED:
            continue
        flag = f"--{f.replace('_', '-')}"
        if f == "algo":
            p.add_argument(flag, default=default, choices=ALGOS)
        elif f == "reward_mode":
            p.add_argument(flag, default=default, choices=REWARD_MODES)
        elif default is None:  # only target_kl is optional
            p.add_argument(flag, default=None, type=float)
        elif isinstance(default, bool):
            p.add_argument(flag, default=default, type=_str2bool)
        else:
            p.add_argument(flag, default=default, type=type(default))
    ns = p.parse_args(argv)
    for f in vars(a):
        if f not in DERIVED:
            setattr(a, f, getattr(ns, f))  # argparse maps --foo-bar -> foo_bar
    return finalize(a)


def finalize(a: Args) -> Args:
    """Fill name defaults and the batch sizes derived from the rollout shape."""
    a.exp_name = a.exp_name or a.algo
    a.results_dir = a.results_dir or f"results_{a.algo}"
    rollout_actors = a.num_envs * a.n_agents
    a.batch_size = a.num_steps * rollout_actors
    a.minibatch_size = a.batch_size // a.num_minibatches
    a.num_iterations = a.total_timesteps // (a.num_steps * a.num_envs)
    return a


# =============================================================================
# Registered experiments (configs/experiments.json)
# =============================================================================

EXPERIMENTS_JSON = Path(__file__).resolve().parents[1] / "configs" / "experiments.json"
# per-experiment keys that describe the experiment rather than set an Args field
META_KEYS = ("exp_id", "label", "n_seeds")


def load_experiments(path=EXPERIMENTS_JSON) -> dict[int, dict]:
    """All registered experiments keyed by exp_id; every non-meta key must be an Args field."""
    with open(path) as f:
        entries = json.load(f)
    valid = set(vars(Args())) - set(DERIVED)
    experiments = {}
    for e in entries:
        unknown = set(e) - valid - set(META_KEYS)
        if unknown:
            raise ValueError(f"exp {e.get('exp_id')} in {path}: unknown keys {sorted(unknown)}")
        experiments[e["exp_id"]] = e
    return experiments


def experiment_flags(exp: dict) -> list[str]:
    """CLI flags for the trainer that reproduce a registered experiment.

    The experiment is named ``exp_<id>`` for both the run dirs and the results
    subdir, i.e. runs/exp_<id>_seed_<s> and <results_dir>/exp_<id>.
    """
    name = f"exp_{exp['exp_id']}"
    flags = ["--exp-name", name, "--config-name", name]
    for k, v in exp.items():
        if k not in META_KEYS:
            flags += [f"--{k.replace('_', '-')}", str(v)]
    return flags

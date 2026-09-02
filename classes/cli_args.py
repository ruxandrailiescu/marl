import argparse
from dataclasses import dataclass


@dataclass
class Args:
    # experiment / reproducibility
    exp_name: str = ""
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
    num_minibatches: int = 1
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
    eval_radius: float = 0.15     # "landmark covered" / collision distance threshold

    # seed-aggregation outputs
    results_dir: str = "results_ippo"
    config_name: str = ""         # subdir under results_dir (defaults to reward_mode)

    # derived (filled in main)
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def parse_args() -> Args:
    a = Args()
    derived = ("batch_size", "minibatch_size", "num_iterations")
    p = argparse.ArgumentParser(description="IPPO for NHMRS simple_assignment_v0")
    for f, default in vars(a).items():
        if f in derived:
            continue
        flag = f"--{f.replace('_', '-')}"
        if default is None:
            p.add_argument(flag, default=None)
        elif isinstance(default, bool):
            p.add_argument(flag, default=default,
                           type=lambda x: str(x).lower() in ("1", "true", "yes"))
        else:
            p.add_argument(flag, default=default, type=type(default))
    ns = p.parse_args()
    for f in vars(a):
        if f not in derived:
            setattr(a, f, getattr(ns, f))  # argparse maps --foo-bar -> foo_bar
    return a
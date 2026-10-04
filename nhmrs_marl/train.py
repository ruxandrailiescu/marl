"""IPPO / MAPPO trainer for the NHMRS ``simple_assignment_v0`` environment.

Both algorithms use parameter sharing: the agents are homogeneous, so one
actor-critic network is shared and the N agents of each of E env copies are
treated as B = E * N entries in the batch dimension. The actor acts on each
agent's local observation. The critic is the only difference:

- IPPO: decentralized critic, valuing each agent's own observation.
- MAPPO: centralized critic, valuing the global state (all N local
  observations concatenated), shared by every agent in the same env. Rewards
  stay per-agent, so each agent gets its own advantage A_i = GAE(r_i, V(s));
  the shared critic regresses to their mean (the team-average return).

Actions are clipped to the env's Box bounds (unicycle: [v, omega] in
[-2, 2] x [-pi, pi]) before stepping. See ``rollout.py`` for truncation
bootstrapping.

Outputs:
  runs/<exp_name>_seed_<seed>/              tensorboard events, episode_returns.png,
                                            agent_<algo>.pt
  <results_dir>/<config_name>/seed_<seed>.npz   greedy-eval curves for aggregation

Usage:
    python -m nhmrs_marl.train --algo mappo --reward-mode spread --seed 1
"""

from __future__ import annotations

import glob
import os
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from nhmrs_marl import metrics
from nhmrs_marl.config import Args, parse_args
from nhmrs_marl.env import EnvBatch, env_kwargs, global_state
from nhmrs_marl.networks import ActorCritic
from nhmrs_marl.normalization import RunningMeanStd
from nhmrs_marl.ppo import compute_gae, ppo_update
from nhmrs_marl.rollout import RolloutBuffer, RolloutCollector

EVAL_KEYS = ("team_return", "coverage", "collision_rate", "mean_final_dist")


def critic_input_fn(algo: str, n_agents: int):
    """(B, obs_dim) observations -> the critic's input for ``algo``."""
    if algo == "mappo":
        return lambda obs_B: global_state(obs_B, n_agents)
    return lambda obs_B: obs_B


def train(args: Args):
    eval_every = max(1, args.num_iterations // args.eval_points)  # eval cadence in iterations
    config_dir = os.path.join(args.results_dir, args.config_name or args.reward_mode)
    algo_label = args.algo.upper()

    run_name = f"{args.exp_name}_seed_{args.seed}"
    run_dir = os.path.join("runs", run_name)
    # a second event file in the same dir gets merged into one jumbled curve
    if glob.glob(os.path.join(run_dir, "events.out.tfevents.*")):
        raise SystemExit(f"{run_dir} already holds a run; delete it or pick another --exp-name/--seed")
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(run_dir)

    # reproducibility
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device} | algo: {args.algo} | run: {run_name}")

    envs = EnvBatch(args.num_envs, args.seed, **env_kwargs(args))
    obs_dim, action_dim = envs.obs_dim, envs.action_dim
    critic_in_dim = obs_dim * envs.n_agents if args.algo == "mappo" else obs_dim
    print(f"agents={args.n_agents} landmarks={args.n_landmarks} "
          f"obs_dim={obs_dim} critic_in_dim={critic_in_dim} action_dim={action_dim} "
          f"rollout_actors={envs.num_actors} rollout_length={args.batch_size} "
          f"iters={args.num_iterations}")

    agent = ActorCritic(obs_dim, critic_in_dim, action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    ret_rms = RunningMeanStd(device=device) if args.norm_returns else None

    buf = RolloutBuffer.zeros(args.num_steps, envs.num_actors, obs_dim, critic_in_dim,
                              action_dim, device)
    collector = RolloutCollector(envs, agent, critic_input_fn(args.algo, envs.n_agents),
                                 args.gamma, ret_rms, device)

    start_time = time.time()
    # completed training episodes (mean per-agent raw return)
    hist_steps: list[int] = []
    hist_return: list[float] = []
    # per-iteration training return (fixed grid, identical across seeds -> stackable)
    train_steps: list[int] = []
    train_return: list[float] = []
    # per-seed greedy-eval curves (fixed timestep grid, saved for aggregation)
    eval_steps: list[int] = []
    eval_series: dict[str, list[float]] = {k: [] for k in EVAL_KEYS}

    def save_plot():
        metrics.plot_episode_returns(
            hist_steps, hist_return, os.path.join(run_dir, "episode_returns.png"),
            title=f"{algo_label} on simple_assignment_v0 ({args.reward_mode})")

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        episodes = collector.collect(buf)
        global_step = collector.global_step
        for step, ep_ret in episodes:
            hist_steps.append(step)
            hist_return.append(ep_ret)
            writer.add_scalar("charts/episodic_return", ep_ret, step)

        # advantages and return targets, on the unnormalized return scale
        with torch.no_grad():
            next_value = collector.value(collector.next_critic_in)
            values = ret_rms.denormalize(buf.values) if ret_rms is not None else buf.values
            advantages = compute_gae(buf.rewards, values, buf.dones, next_value,
                                     collector.next_done, args.gamma, args.gae_lambda)
            returns = advantages + values
            if ret_rms is not None:
                ret_rms.update(returns.reshape(-1))

        batch = buf.flatten()
        batch["advantages"] = advantages.reshape(-1)
        batch["returns"] = returns.reshape(-1)

        before = metrics.snapshot_before_update(agent, batch["obs"], batch["critic_in"])
        stats = ppo_update(agent, optimizer, batch, args, rng, ret_rms)
        stats.update(metrics.update_sizes(agent, before, batch["obs"], batch["critic_in"]))

        if episodes:
            train_steps.append(global_step)
            train_return.append(float(np.mean([r for _, r in episodes])))

        # logging
        sps = int(global_step / (time.time() - start_time))
        recent = np.mean(hist_return[-args.num_envs:]) if hist_return else float("nan")
        print(f"iter {iteration:04d}/{args.num_iterations} | step {global_step:>9d} | "
              f"ep_ret {recent:8.2f} | v_loss {stats['value_loss']:7.3f} | "
              f"pg_loss {stats['policy_loss']:7.4f} | kl {stats['approx_kl']:.4f} | SPS {sps}")
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        for k in ("value_loss", "policy_loss", "entropy", "approx_kl", "clipfrac"):
            writer.add_scalar(f"losses/{k}", stats[k], global_step)
        for k in ("policy_param_update", "policy_output_kl", "critic_output_update",
                  "actor_grad_norm", "critic_grad_norm"):
            writer.add_scalar(f"diagnostics/{k}", stats[k], global_step)

        if iteration % eval_every == 0 or iteration == args.num_iterations:
            eval_metrics = metrics.greedy_eval(agent, args, device, n_episodes=args.n_eval_episodes)
            eval_steps.append(global_step)
            eval_series["team_return"].append(eval_metrics["episodic_return"])
            for k in EVAL_KEYS[1:]:
                eval_series[k].append(eval_metrics[k])
            print("  eval: " + " ".join(f"{k}={v:.3f}" for k, v in eval_metrics.items()))
            for k, v in eval_metrics.items():
                writer.add_scalar(f"eval/{k}", v, global_step)

        if iteration % args.plot_interval == 0:
            save_plot()

    # finish
    save_plot()
    metrics.save_seed_metrics(config_dir, args.seed, eval_steps, eval_series,
                              extra={"train_steps": train_steps, "train_return": train_return})
    torch.save(agent.state_dict(), os.path.join(run_dir, f"agent_{args.algo}.pt"))
    envs.close()
    writer.close()
    print(f"saved per-seed eval metrics to {os.path.join(config_dir, f'seed_{args.seed}.npz')}")


def main():
    train(parse_args())


if __name__ == "__main__":
    main()

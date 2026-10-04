"""Minimal inference / visualization for an IPPO or MAPPO policy on simple_assignment_v0.

Training saves a single shared actor-critic (parameter sharing across the N
homogeneous agents) as ``runs/<run_name>/agent_ippo.pt`` or ``agent_mappo.pt``.
Which one a checkpoint holds is detected from its critic input width (obs_dim
for IPPO, N * obs_dim for MAPPO). At inference only the actor is needed, and we
act greedily by taking the actor mean (no sampling), clipped to the action
bounds -- exactly the greedy rollout used during eval.

The env MUST be built the same way as training (``observe_relative=True`` and the
same agent/landmark counts) so the observation layout matches the trained policy.

Usage:
    python scripts/infer.py runs/<run_name>/agent_ippo.pt
    python scripts/infer.py runs/<run_name>/agent_mappo.pt
    python scripts/infer.py <ckpt> --render-mode rgb_array --save-video out.mp4
    python scripts/infer.py <ckpt> --render-mode none        # no window (debug)
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

# reuse the exact network + env adapters from training (no drift)
from nhmrs_marl.env import array_to_action_dict, make_env, rewards_to_array, world_obs_array
from nhmrs_marl.networks import ActorCritic


def parse_args():
    p = argparse.ArgumentParser(description="Visualize a trained IPPO or MAPPO policy")
    p.add_argument("checkpoint", help="path to agent_ippo.pt or agent_mappo.pt")
    p.add_argument("--reward-mode", default="spread")
    p.add_argument("--n-agents", type=int, default=3)
    p.add_argument("--n-landmarks", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--render-mode", default="human", choices=["human", "rgb_array", "none"])
    p.add_argument("--save-video", default=None, help="path for rgb_array frames (needs imageio)")
    return p.parse_args()


def main():
    args = parse_args()
    render_mode = None if args.render_mode == "none" else args.render_mode

    env = make_env(args.reward_mode, args.n_agents, args.n_landmarks, args.max_steps,
                   render_mode=render_mode)
    env.reset(seed=args.seed)

    agent_order = env.possible_agents[:]
    obs_dim = env.observation_space(agent_order[0]).shape[0]
    action_dim = env.action_space(agent_order[0]).shape[0]
    a_low = env.action_space(agent_order[0]).low
    a_high = env.action_space(agent_order[0]).high

    # load the shared actor-critic, eval mode, greedy (mean) actions.
    # Only the actor is used at inference; the critic width tells IPPO
    # (local obs) and MAPPO (global state) checkpoints apart.
    device = torch.device("cpu")
    policy = ActorCritic.from_checkpoint(args.checkpoint, obs_dim, action_dim, device).eval()
    critic_in_dim = policy.critic[0].in_features
    if critic_in_dim == obs_dim:
        algo = "IPPO"
    elif critic_in_dim == obs_dim * args.n_agents:
        algo = "MAPPO"
    else:
        raise SystemExit(f"checkpoint critic input {critic_in_dim} matches neither IPPO "
                         f"({obs_dim}) nor MAPPO ({obs_dim * args.n_agents}); "
                         f"check --n-agents / --n-landmarks")
    print(f"loaded {algo} {args.checkpoint} | agents={args.n_agents} "
          f"obs_dim={obs_dim} action_dim={action_dim}")

    frames = []
    for ep in range(args.episodes):
        env.reset()
        ep_return = 0.0
        for _ in range(args.max_steps):
            obs = torch.tensor(world_obs_array(env, agent_order), dtype=torch.float32, device=device)
            with torch.no_grad():
                mean = policy.actor_mean(obs).cpu().numpy()
            action = np.clip(mean, a_low, a_high)
            _, reward, _, truncated, _ = env.step(array_to_action_dict(action, agent_order))
            ep_return += float(np.mean(rewards_to_array(reward, agent_order)))

            if render_mode is not None:
                frame = env.render()
                if render_mode == "rgb_array" and args.save_video:
                    frames.append(frame)
                if render_mode == "human" and args.fps:
                    time.sleep(1.0 / args.fps)

            if all(truncated.values()):
                break
        print(f"episode {ep + 1:02d}/{args.episodes} | return {ep_return:8.2f}")

    env.close()

    if args.save_video and frames:
        import imageio
        imageio.mimsave(args.save_video, frames, fps=args.fps)
        print(f"saved video to {args.save_video} ({len(frames)} frames)")


if __name__ == "__main__":
    main()

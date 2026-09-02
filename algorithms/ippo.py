import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from classes.agent import Agent
from classes.running_mean_std import RunningMeanStd
from classes.cli_args import parse_args
from utils import metrics
from utils.env import array_to_action_dict, collect_obs, make_env, rewards_to_array, world_obs_array
from utils.metrics import save_plot


def main():
    args = parse_args()
    B = args.num_envs * args.n_agents                       # rollout "actors"
    args.batch_size = args.num_steps * B
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // (args.num_steps * args.num_envs)
    eval_every = max(1, args.num_iterations // args.eval_points)  # eval cadence in iterations
    config_dir = os.path.join(args.results_dir, args.config_name or args.reward_mode)

    run_name = f"{args.exp_name}_seed_{args.seed}"
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(run_dir) 

    # reproducibility
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device} | run: {run_name}")

    # build envs
    envs = [make_env(args) for _ in range(args.num_envs)]
    for i, env in enumerate(envs):
        env.reset(seed=args.seed + i)
    agent_order = envs[0].possible_agents[:]                # fixed agent order
    obs_dim = envs[0].observation_space(agent_order[0]).shape[0]
    action_dim = envs[0].action_space(agent_order[0]).shape[0]
    a_low, a_high = envs[0].action_space(agent_order[0]).low, envs[0].action_space(agent_order[0]).high
    action_low = torch.tensor(a_low, dtype=torch.float32, device=device)
    action_high = torch.tensor(a_high, dtype=torch.float32, device=device)
    print(f"agents={args.n_agents} landmarks={args.n_landmarks} "
          f"obs_dim={obs_dim} action_dim={action_dim} rollout_actors={B} "
          f"rollout_length={args.batch_size} iters={args.num_iterations}")

    agent = Agent(obs_dim, action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    ret_rms = RunningMeanStd(device=device) if args.norm_returns else None

    # rollout storage: (num_steps, B, .)
    obs = torch.zeros((args.num_steps, B, obs_dim), device=device)
    actions = torch.zeros((args.num_steps, B, action_dim), device=device)
    log_probs = torch.zeros((args.num_steps, B), device=device)
    rewards = torch.zeros((args.num_steps, B), device=device)
    dones = torch.zeros((args.num_steps, B), device=device)
    values = torch.zeros((args.num_steps, B), device=device)

    global_step = 0
    start_time = time.time()
    next_obs = torch.tensor(collect_obs(agent_order, envs), dtype=torch.float32, device=device)
    next_done = torch.zeros(B, device=device)

    # per-actor running episode return (raw reward, for logging/plots)
    ep_return = np.zeros(B, dtype=np.float64)
    # plotting history
    hist_steps: list[int] = []
    hist_return: list[float] = []
    # per-seed greedy-eval curves (fixed timestep grid, saved for aggregation)
    eval_steps: list[int] = []
    eval_series: dict[str, list[float]] = {
        "team_return": [], "coverage": [], "collision_rate": [], "mean_final_dist": [],
    }
    # per-iteration training return (fixed grid, identical across seeds -> stackable)
    train_steps: list[int] = []
    train_return: list[float] = []

    for iteration in range(1, args.num_iterations + 1):
        ep_start = len(hist_return)                         # episodes completed before this iter
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        # rollout
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            log_probs[step] = logprob

            # clip to action bounds, then step every env
            clipped = torch.max(torch.min(action, action_high), action_low).cpu().numpy()
            reward_B = np.zeros(B, dtype=np.float32)
            trunc_B = np.zeros(B, dtype=np.float32)
            real_next = np.zeros((B, obs_dim), dtype=np.float32)
            for ei, env in enumerate(envs):
                sl = slice(ei * args.n_agents, (ei + 1) * args.n_agents)
                _, rew, _, truncated, _ = env.step(array_to_action_dict(clipped[sl], agent_order))
                reward_B[sl] = rewards_to_array(rew, agent_order)
                real_next[sl] = world_obs_array(env, agent_order)
                if all(truncated.values()):
                    trunc_B[sl] = 1.0

            # log + accumulate raw episodic return before any bootstrap mutation
            ep_return += reward_B
            rewards[step] = torch.tensor(reward_B, dtype=torch.float32, device=device)

            if trunc_B.any():
                rn = torch.tensor(real_next, dtype=torch.float32, device=device)
                with torch.no_grad():
                    boot_v = agent.get_value(rn).flatten()
                    if ret_rms is not None:
                        boot_v = ret_rms.denormalize(boot_v)
                mask = torch.tensor(trunc_B, dtype=torch.float32, device=device)
                rewards[step] = rewards[step] + args.gamma * boot_v * mask
                for ei, env in enumerate(envs):
                    sl = slice(ei * args.n_agents, (ei + 1) * args.n_agents)
                    if trunc_B[ei * args.n_agents] > 0:
                        # record completed-episode return, then reset
                        completed = float(ep_return[sl].mean())
                        hist_steps.append(global_step)
                        hist_return.append(completed)
                        if writer:
                            writer.add_scalar("charts/episodic_return", completed, global_step)
                        ep_return[sl] = 0.0
                        env.reset()

            next_done = torch.tensor(trunc_B, dtype=torch.float32, device=device)
            next_obs = torch.tensor(collect_obs(), dtype=torch.float32, device=device)

        # GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs).flatten()
            if ret_rms is not None:
                next_value = ret_rms.denormalize(next_value)
                val = ret_rms.denormalize(values)
            else:
                val = values
            advantages = torch.zeros_like(rewards)
            last_gae = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_non_terminal = 1.0 - next_done
                    next_values = next_value
                else:
                    next_non_terminal = 1.0 - dones[t + 1]
                    next_values = val[t + 1]
                delta = rewards[t] + args.gamma * next_values * next_non_terminal - val[t]
                advantages[t] = last_gae = (
                    delta + args.gamma * args.gae_lambda * next_non_terminal * last_gae
                )
            returns = advantages + val
            if ret_rms is not None:
                ret_rms.update(returns.reshape(-1))

        # flatten batch
        b_obs = obs.reshape(-1, obs_dim)
        b_log_probs = log_probs.reshape(-1)
        b_actions = actions.reshape(-1, action_dim)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # diagnostics: snapshot policy/critic before the update
        actor_params = list(agent.actor_mean.parameters()) + [agent.actor_logstd]
        critic_params = list(agent.critic.parameters())
        with torch.no_grad():
            old_actor_flat = metrics.flat_params(actor_params)
            old_mean = agent.actor_mean(b_obs)
            old_logstd = agent.actor_logstd.expand_as(old_mean)
            v_before = agent.get_value(b_obs).flatten()
        actor_gnorm_sum = critic_gnorm_sum = 0.0
        n_grad_steps = 0

        # PPO update
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for _ in range(args.update_epochs):
            rng.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb = b_inds[start:start + args.minibatch_size]
                _, new_log_prob, entropy, new_value = agent.get_action_and_value(
                    b_obs[mb], b_actions[mb])
                log_ratio = new_log_prob - b_log_probs[mb]
                ratio = log_ratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_adv = b_advantages[mb]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = mb_adv * ratio
                pg_loss2 = mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = -torch.min(pg_loss1, pg_loss2).mean()

                new_value = new_value.view(-1)
                mb_returns = b_returns[mb]
                if ret_rms is not None:
                    mb_returns = ret_rms.normalize(mb_returns)
                if args.clip_vloss:
                    v_loss_unclipped = (new_value - mb_returns) ** 2
                    v_clipped = b_values[mb] + torch.clamp(
                        new_value - b_values[mb], -args.clip_coef, args.clip_coef)
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

        # diagnostics: measure update sizes after the full update
        with torch.no_grad():
            new_mean = agent.actor_mean(b_obs)
            new_logstd = agent.actor_logstd.expand_as(new_mean)
            policy_output_kl = metrics.gaussian_kl(old_mean, old_logstd, new_mean, new_logstd).mean().item()
            policy_param_update = (metrics.flat_params(actor_params) - old_actor_flat).norm().item()
            critic_output_update = (agent.get_value(b_obs).flatten() - v_before).pow(2).mean().sqrt().item()
        actor_grad_norm = actor_gnorm_sum / max(1, n_grad_steps)
        critic_grad_norm = critic_gnorm_sum / max(1, n_grad_steps)

        # record mean training return over episodes completed this iteration
        if len(hist_return) > ep_start:
            train_steps.append(global_step)
            train_return.append(float(np.mean(hist_return[ep_start:])))

        # logging
        sps = int(global_step / (time.time() - start_time))
        recent = np.mean(hist_return[-args.num_envs:]) if hist_return else float("nan")
        print(f"iter {iteration:04d}/{args.num_iterations} | step {global_step:>9d} | "
              f"ep_ret {recent:8.2f} | v_loss {v_loss.item():7.3f} | "
              f"pg_loss {pg_loss.item():7.4f} | kl {approx_kl.item():.4f} | SPS {sps}")
        if writer:
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar("diagnostics/policy_param_update", policy_param_update, global_step)
            writer.add_scalar("diagnostics/policy_output_kl", policy_output_kl, global_step)
            writer.add_scalar("diagnostics/critic_output_update", critic_output_update, global_step)
            writer.add_scalar("diagnostics/actor_grad_norm", actor_grad_norm, global_step)
            writer.add_scalar("diagnostics/critic_grad_norm", critic_grad_norm, global_step)

        if iteration % eval_every == 0 or iteration == args.num_iterations:
            eval_metrics = metrics.greedy_eval(
                agent, lambda: make_env(args),
                (world_obs_array, array_to_action_dict, rewards_to_array),
                args, device, agent_order, a_low, a_high, n_episodes=args.n_eval_episodes)
            eval_steps.append(global_step)
            eval_series["team_return"].append(eval_metrics["episodic_return"])
            eval_series["coverage"].append(eval_metrics["coverage"])
            eval_series["collision_rate"].append(eval_metrics["collision_rate"])
            eval_series["mean_final_dist"].append(eval_metrics["mean_final_dist"])
            print("  eval: " + " ".join(f"{k}={v:.3f}" for k, v in eval_metrics.items()))
            if writer:
                for k, v in eval_metrics.items():
                    writer.add_scalar(f"eval/{k}", v, global_step)

        if iteration % args.plot_interval == 0:
            save_plot(hist_steps, hist_return, run_dir, "", args)

    # finish
    save_plot(hist_steps, hist_return, run_dir, "", args)
    metrics.save_seed_metrics(config_dir, args.seed, eval_steps, eval_series,
                              extra={"train_steps": train_steps, "train_return": train_return})
    torch.save(agent.state_dict(), os.path.join(run_dir, "agent_ippo.pt"))
    for env in envs:
        env.close()
    if writer:
        writer.close()
    print(f"saved per-seed eval metrics to {os.path.join(config_dir, f'seed_{args.seed}.npz')}")


if __name__ == "__main__":
    main()
"""Run IPPO or MAPPO over several seeds (one config) and plot mean +- std bands.

Each seed is a separate subprocess (crash-isolated), writing its own
``results_dir/config_name/seed_<seed>.npz``. After all seeds finish, the
per-seed curves are aggregated into mean +- std plots in the same directory.

Either run a registered experiment from configs/experiments.json (named
``exp_<id>`` for both runs/ and the results subdir):
    python scripts/run_seeds.py --exp-id 3
or an ad-hoc config, named by --config-name:
    python scripts/run_seeds.py --algo mappo --config-name trial -- --reward-mode spread
Trainer flags after ``--`` are forwarded and override the experiment's values:
    python scripts/run_seeds.py --exp-id 7 --n-seeds 2 -- --total-timesteps 500000
"""

import argparse
import os
import subprocess
import sys

from nhmrs_marl import metrics
from nhmrs_marl.config import ALGOS, experiment_flags, load_experiments, parse_args


def main():
    p = argparse.ArgumentParser(description="Multi-seed IPPO/MAPPO runner + aggregation")
    p.add_argument("--exp-id", type=int, help="registered experiment in configs/experiments.json")
    p.add_argument("--algo", choices=ALGOS, help="ad-hoc runs only")
    p.add_argument("--config-name", help="ad-hoc runs only: results subdir and run-name prefix")
    p.add_argument("--n-seeds", type=int, default=None, help="default: the experiment's n_seeds, else 5")
    p.add_argument("--base-seed", type=int, default=1)
    p.add_argument("--aggregate-only", action="store_true",
                   help="skip training; only re-aggregate existing seed files")
    p.add_argument("passthrough", nargs="*",
                   help="extra flags forwarded to the trainer (after --)")
    args = p.parse_args()

    if args.exp_id is not None:
        if args.algo or args.config_name:
            p.error("--algo/--config-name come from the experiment when --exp-id is given")
        exp = load_experiments()[args.exp_id]
        flags = experiment_flags(exp)
        n_seeds = args.n_seeds or exp["n_seeds"]
    else:
        if not (args.algo and args.config_name):
            p.error("give --exp-id, or both --algo and --config-name")
        flags = ["--algo", args.algo, "--exp-name", args.config_name,
                 "--config-name", args.config_name]
        n_seeds = args.n_seeds or 5
    flags += args.passthrough

    # resolve exactly as the trainer will, to find its output dir
    trainer_args = parse_args(flags)
    config_dir = os.path.join(trainer_args.results_dir, trainer_args.config_name)

    if not args.aggregate_only:
        for i in range(n_seeds):
            seed = args.base_seed + i
            cmd = [sys.executable, "-m", "nhmrs_marl.train", "--seed", str(seed)] + flags
            print(f"\n=== seed {seed} ({i + 1}/{n_seeds}) ===")
            print("running:", " ".join(cmd))
            subprocess.run(cmd, check=True)

    metrics.aggregate_and_plot(config_dir, algo=trainer_args.algo.upper())


if __name__ == "__main__":
    main()

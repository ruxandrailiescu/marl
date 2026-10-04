"""Overlay seed-aggregated IPPO/MAPPO configs on shared axes (mean +- std per metric).

Example (value-standardization ablation):
    python scripts/compare.py results_ippo/valstd_on results_ippo/valstd_off \
        --labels "value std on" "value std off"
Writes compare_*.png to --out-dir (default: <parent>/compare).
"""

import argparse
import os

from nhmrs_marl import metrics


def main():
    p = argparse.ArgumentParser(description="Compare seed-aggregated configs")
    p.add_argument("config_dirs", nargs="+", help="config directories (each with seed_*.npz)")
    p.add_argument("--labels", nargs="*", default=None, help="legend label per config dir")
    p.add_argument("--out-dir", default=None, help="output dir (default: <parent>/compare)")
    args = p.parse_args()

    if args.labels and len(args.labels) != len(args.config_dirs):
        p.error("--labels must have one label per config dir")

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.normpath(args.config_dirs[0])), "compare")
    metrics.compare_configs(args.config_dirs, out_dir, args.labels)


if __name__ == "__main__":
    main()

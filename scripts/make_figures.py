"""Seed-aggregated figures for registered experiments, read from TensorBoard events.

Reads the per-seed event files in ``<runs-dir>/exp_<id>_seed_<s>`` directly
(no dependence on the ``.npz`` pipeline) and renders:

  * overlays      -- one axis per ablation group, eval return of each member as
                     a mean +- 1 std band (headline figures).
  * grids         -- per ablation group, a 2x2 panel of eval metrics.
  * trainreturns  -- per ablation group, the EMA-smoothed training return.
  * configs       -- per experiment, its 5 eval metrics with faint per-seed
                     traces plus the smoothed training return (appendix figures).

Ablation groups live in configs/figures.json; experiments in
configs/experiments.json. Eval is logged on a fixed step grid identical across
seeds (fixed-length episodes), so seeds stack directly with no interpolation.
Figures are written as both vector PDF and PNG.

Usage:
    python scripts/make_figures.py --runs-dir ~/rl-runs/runs
    python scripts/make_figures.py --runs-dir R --style paper --group "IPPO vs MAPPO" --only grids
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from nhmrs_marl.config import load_experiments

FIGURES_JSON = Path(__file__).resolve().parents[1] / "configs" / "figures.json"
FIGURE_KINDS = ("overlays", "grids", "trainreturns", "configs")

EVAL_RETURN = ("eval/episodic_return", "Episode rewards")
GRID_METRICS = [  # 2x2 group grid; distinct success is only in the per-config grid
    EVAL_RETURN,
    ("eval/coverage", "Landmark coverage"),
    ("eval/collision_rate", "Collision rate"),
    ("eval/mean_final_dist", "Mean final distance"),
]
CONFIG_METRICS = GRID_METRICS + [("eval/distinct_success", "Distinct success rate")]
TRAIN_RETURN = "charts/episodic_return"


# =============================================================================
# Styles
# =============================================================================

_LEGEND_BOX = dict(frameon=True, handlelength=1.5, facecolor="white", edgecolor="none",
                   framealpha=0.9, borderpad=0.6, labelspacing=0.4)

STYLES = {
    # light: white panel, y-grid only, x in millions of steps, n in legend
    "default": dict(
        rc={
            "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 10,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6, "figure.dpi": 120,
        },
        palette=["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"],
        lw=2.2, band_alpha=0.18, x_scale=1e6, xlabel="Environment steps (1e6)",
        y_grid_only=True, show_n=True, legend=dict(frameon=False, loc="best"),
        overlay_size=(6.0, 4.2), grid_size=(11.0, 8.5), title_size=13,
    ),
    # seaborn-darkgrid / MAPPO (Yu et al. 2022) aesthetic: gray panel, white
    # gridlines behind the data, no spines, large fonts
    "paper": dict(
        rc={
            "font.size": 16, "axes.titlesize": 24, "axes.labelsize": 20, "legend.fontsize": 18,
            "xtick.labelsize": 16, "ytick.labelsize": 16,
            "axes.facecolor": "#E6E6E6", "axes.edgecolor": "white", "axes.linewidth": 0,
            "axes.axisbelow": True,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.spines.left": False, "axes.spines.bottom": False,
            "axes.grid": True, "grid.color": "white", "grid.linewidth": 1.2, "grid.alpha": 1.0,
            "xtick.major.size": 0, "ytick.major.size": 0, "figure.dpi": 120,
        },
        palette=["red", "blue", "green", "purple", "orange", "brown"],
        lw=2.8, band_alpha=0.2, x_scale=1.0, xlabel="Timesteps",
        y_grid_only=False, show_n=False, legend=dict(_LEGEND_BOX, loc="lower right"),
        overlay_size=(6.5, 5.2), grid_size=(13.5, 11.0), title_size=26,
    ),
}


# =============================================================================
# Data loading
# =============================================================================

@lru_cache(maxsize=None)
def load_run(run_dir: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """All scalars of one run: {tag: (steps, values)}. Each event file is read once."""
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    out = {}
    for tag in ea.Tags()["scalars"]:
        events = ea.Scalars(tag)
        out[tag] = (np.array([e.step for e in events], dtype=np.float64),
                    np.array([e.value for e in events], dtype=np.float64))
    return out


def seed_runs(runs_dir: str, exp_id: int) -> list[str]:
    """Run dirs of every seed of ``exp_<id>``, in seed order."""
    dirs = glob.glob(os.path.join(runs_dir, f"exp_{exp_id}_seed_*"))
    seed = lambda d: int(re.search(r"_seed_(\d+)$", d).group(1))
    return sorted((d for d in dirs if re.search(r"_seed_\d+$", d)), key=seed)


def aggregate_seeds(runs_dir: str, exp_id: int, tag: str):
    """Stack a tag across an exp's seeds -> (steps, mean, std, n_seeds), or None.

    Truncates to the shortest seed series (a safety net; within a config the
    grids are identical).
    """
    series = [load_run(d)[tag] for d in seed_runs(runs_dir, exp_id) if tag in load_run(d)]
    if not series:
        return None
    L = min(len(values) for _, values in series)
    arr = np.stack([values[:L] for _, values in series])  # (n_seeds, L)
    return series[0][0][:L], arr.mean(axis=0), arr.std(axis=0), len(series)


def ema(values: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """Exponential moving average for the noisy per-episode training curve."""
    out = np.empty_like(values, dtype=np.float64)
    acc = values[0]
    for i, v in enumerate(values):
        acc = alpha * acc + (1.0 - alpha) * v
        out[i] = acc
    return out


# =============================================================================
# Figures
# =============================================================================

class Figures:
    def __init__(self, runs_dir: str, out_dir: str, style: str):
        self.runs_dir = runs_dir
        self.out_dir = out_dir
        self.s = STYLES[style]
        plt.rcParams.update(self.s["rc"])

    # -- helpers --------------------------------------------------------------

    def member_color(self, member: dict, position: int) -> str:
        return member.get("color") or self.s["palette"][position % len(self.s["palette"])]

    def label(self, text: str, n: int) -> str:
        return f"{text} (n={n})" if self.s["show_n"] else text

    def band(self, ax, steps, mean, std, color, label, smooth=False):
        x = steps / self.s["x_scale"]
        if smooth:  # faint raw mean under the EMA
            ax.plot(x, mean, color=color, alpha=0.15, lw=1)
            mean = ema(mean)
        ax.plot(x, mean, color=color, lw=self.s["lw"], label=label)
        ax.fill_between(x, mean - std, mean + std, color=color,
                        alpha=self.s["band_alpha"], linewidth=0)

    def plot_member(self, ax, exp_id, tag, color, label, smooth=False):
        agg = aggregate_seeds(self.runs_dir, exp_id, tag)
        if agg is None:
            print(f"  [skip] exp{exp_id} has no '{tag}'")
            return
        steps, mean, std, n = agg
        self.band(ax, steps, mean, std, color, self.label(label, n), smooth=smooth)

    def members_with_runs(self, group: dict) -> list[tuple[int, dict]]:
        """(palette position, member) for members that have runs; reports the rest once."""
        present = []
        for i, m in enumerate(group["members"]):
            if seed_runs(self.runs_dir, m["exp_id"]):
                present.append((i, m))
            else:
                print(f"  [skip] exp{m['exp_id']}: no runs")
        if not present:
            print(f"  [skip] '{group['title']}': none of its experiments have runs")
        return present

    def style_axis(self, ax, ylabel, legend_loc=None):
        ax.set_xlabel(self.s["xlabel"])
        ax.set_ylabel(ylabel)
        if self.s["y_grid_only"]:
            ax.grid(axis="y", alpha=0.25)
            ax.grid(axis="x", visible=False)
        else:
            # raw timesteps with an automatic "1e7"-style offset
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.margins(x=0.01)
        if ax.get_legend_handles_labels()[0]:
            legend = dict(self.s["legend"])
            if legend_loc:
                legend["loc"] = legend_loc
            ax.legend(**legend)

    def save(self, fig, name: str):
        os.makedirs(self.out_dir, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(self.out_dir, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {name}.pdf/.png")

    # -- figure kinds ---------------------------------------------------------

    def overlay(self, group: dict, tag: str, ylabel: str, prefix: str, smooth=False):
        """One axis; each group member as a mean +- std band."""
        members = self.members_with_runs(group)
        if not members:
            return
        fig, ax = plt.subplots(figsize=self.s["overlay_size"], constrained_layout=True)
        for i, m in members:
            self.plot_member(ax, m["exp_id"], tag, self.member_color(m, i), m["label"], smooth)
        ax.set_title(group["title"])
        self.style_axis(ax, ylabel)
        self.save(fig, prefix + slug(group["title"]))

    def overlays(self, group: dict):
        self.overlay(group, *EVAL_RETURN, prefix="ablation_")

    def trainreturns(self, group: dict):
        self.overlay(group, TRAIN_RETURN, "Episode rewards", "ablation_trainreturn_", smooth=True)

    def grids(self, group: dict):
        """2x2 eval metrics; each member is one mean +- std band per panel."""
        members = self.members_with_runs(group)
        if not members:
            return
        fig, axes = plt.subplots(2, 2, figsize=self.s["grid_size"], constrained_layout=True)
        # generous padding so neighbouring axis labels/titles never overlap
        fig.set_constrained_layout_pads(w_pad=0.18, h_pad=0.18, wspace=0.10, hspace=0.10)
        for ax, (tag, ylabel) in zip(axes.ravel(), GRID_METRICS):
            for i, m in members:
                self.plot_member(ax, m["exp_id"], tag, self.member_color(m, i), m["label"])
            ax.set_title(ylabel)
            self.style_axis(ax, ylabel, legend_loc="best")
        fig.suptitle(group["title"], fontsize=self.s["title_size"])
        self.save(fig, "ablation_grid_" + slug(group["title"]))

    def config_grid(self, exp: dict):
        """2x3: 5 eval metrics with faint per-seed traces + smoothed training return."""
        exp_id = exp["exp_id"]
        runs = seed_runs(self.runs_dir, exp_id)
        if not runs:
            print(f"  [skip] exp{exp_id}: no runs in {self.runs_dir}")
            return
        color = self.s["palette"][(exp_id - 1) % len(self.s["palette"])]
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), constrained_layout=True)
        axes = axes.ravel()
        for ax, (tag, ylabel) in zip(axes, CONFIG_METRICS):
            for run in runs:
                if tag in load_run(run):
                    steps, values = load_run(run)[tag]
                    ax.plot(steps / self.s["x_scale"], values, color=color, alpha=0.12, lw=1)
            self.plot_member(ax, exp_id, tag, color, "mean")
            ax.set_title(ylabel)
            self.style_axis(ax, ylabel, legend_loc="best")
        self.plot_member(axes[5], exp_id, TRAIN_RETURN, color, "train return, EMA", smooth=True)
        axes[5].set_title("Training return (smoothed)")
        self.style_axis(axes[5], "Episode rewards", legend_loc="best")
        fig.suptitle(f"exp{exp_id} ({exp['algo']}): {exp['label']}", fontsize=self.s["title_size"])
        self.save(fig, f"config_grid_exp{exp_id}_{exp['label']}")


def slug(title: str) -> str:
    return title.lower().replace(" ", "_")


def main():
    p = argparse.ArgumentParser(description="Seed-aggregated figures from TensorBoard runs")
    p.add_argument("--runs-dir", required=True, help="directory holding exp_<id>_seed_<s> runs")
    p.add_argument("--out-dir", default="figures")
    p.add_argument("--style", choices=sorted(STYLES), default="paper")
    p.add_argument("--only", nargs="+", choices=FIGURE_KINDS, default=list(FIGURE_KINDS),
                   help="figure kinds to render (default: all)")
    p.add_argument("--group", action="append",
                   help="ablation group title from configs/figures.json (repeatable; default: all)")
    p.add_argument("--exp-id", type=int, action="append",
                   help="experiments for the per-config grids (repeatable; default: all)")
    args = p.parse_args()

    runs_dir = os.path.expanduser(args.runs_dir)
    if not os.path.isdir(runs_dir):
        p.error(f"--runs-dir {runs_dir} is not a directory")

    with open(FIGURES_JSON) as f:
        groups = json.load(f)["groups"]
    if args.group:
        unknown = set(args.group) - {g["title"] for g in groups}
        if unknown:
            p.error(f"unknown --group {sorted(unknown)}; see {FIGURES_JSON}")
        groups = [g for g in groups if g["title"] in args.group]
    experiments = load_experiments()
    exp_ids = args.exp_id or sorted(experiments)

    figs = Figures(runs_dir, args.out_dir, args.style)
    for kind in ("overlays", "grids", "trainreturns"):
        if kind in args.only:
            print(f"{kind}:")
            for group in groups:
                getattr(figs, kind)(group)
    if "configs" in args.only:
        print("configs:")
        for exp_id in exp_ids:
            figs.config_grid(experiments[exp_id])
    print(f"\nDone. Figures in {args.out_dir}")


if __name__ == "__main__":
    main()

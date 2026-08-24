#!/usr/bin/env python
"""Figure for the gradient-informed temperature sweep run by run_local_sweep_sampler_k_resid_mlp2.py.

Pulls the final-checkpoint benchmark summary of every `sampler-benchmark-k-*` run and
plots, against k:

  left   mean per-batch dL_recon for both samplers (uniform is flat in k, as it must be)
  right  the ratio dL_GI / dL_uniform, which is the quantity the claim rests on

dL_recon is the reconstruction error induced by one draw of the stochastic mask on a
frozen ComponentModel, so higher means the sampler found a mask that damages the
decomposition more. Both samplers see the same model and the same eval batches, and the
paired t-test across batches is logged per run.

Usage:
    source .venv/bin/activate
    python spd/scripts/plot_sampler_k_sweep.py --entity <wandb_entity>
"""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import wandb

RUN_PREFIX = "sampler-benchmark-k-"

UNIFORM_COLOUR = "#2a78d6"
GI_COLOUR = "#eb6834"
REFERENCE = "#8a8a84"
GRID_COLOUR = "#d8d8d4"
TEXT_PRIMARY = "#1a1a19"
TEXT_SECONDARY = "#5c5c57"


def fetch_k_sweep(entity: str, project: str) -> dict[float, np.ndarray]:
    """k -> array of shape [n_seeds, 4] holding (uniform dL, GI dL, t_stat, p_value)."""
    api = wandb.Api()
    per_k: dict[float, list[tuple[float, float, float, float]]] = defaultdict(list)
    for run in api.runs(f"{entity}/{project}"):
        if not run.name.startswith(RUN_PREFIX):
            continue
        k = float(run.name.split("_k-")[1].split("_seed")[0])
        summary = run.summary
        per_k[k].append(
            (
                summary["benchmark/uniform/mean_delta_l_recon"],
                summary["benchmark/gi/mean_delta_l_recon"],
                summary["benchmark/t_stat"],
                summary["benchmark/p_value"],
            )
        )
    assert per_k, f"no runs named {RUN_PREFIX}* in {entity}/{project}"
    return {k: np.array(v) for k, v in sorted(per_k.items())}


def _style_axis(ax: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=10)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=10)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=11, pad=8)
    ax.grid(True, color=GRID_COLOUR, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID_COLOUR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)


def plot_k_sweep(per_k: dict[float, np.ndarray], out_path: Path) -> None:
    ks = np.array(sorted(per_k))
    uniform = np.array([per_k[k][:, 0] for k in ks])
    gi = np.array([per_k[k][:, 1] for k in ks])
    ratio = gi / uniform

    fig, (ax_abs, ax_ratio) = plt.subplots(1, 2, figsize=(12, 4.6))

    for values, colour, label in [
        (uniform, UNIFORM_COLOUR, "Uniform"),
        (gi, GI_COLOUR, "Gradient-informed"),
    ]:
        mean, std = values.mean(axis=1), values.std(axis=1)
        ax_abs.plot(ks, mean, color=colour, linewidth=2.0, marker="o", markersize=5, label=label)
        ax_abs.fill_between(ks, mean - std, mean + std, color=colour, alpha=0.15, linewidth=0)
    _style_axis(
        ax_abs,
        "temperature $k$",
        r"mean $\Delta \mathcal{L}_{\mathrm{recon}}$ ($\uparrow$ harder mask)",
        "Reconstruction error induced by one mask draw",
    )
    ax_abs.legend(fontsize=9, frameon=False)

    mean, std = ratio.mean(axis=1), ratio.std(axis=1)
    ax_ratio.plot(ks, mean, color=GI_COLOUR, linewidth=2.0, marker="o", markersize=5)
    ax_ratio.fill_between(ks, mean - std, mean + std, color=GI_COLOUR, alpha=0.15, linewidth=0)
    ax_ratio.axhline(1.0, color=REFERENCE, linestyle="--", linewidth=1.0)
    ax_ratio.text(ks[0], 1.005, "uniform baseline", color=TEXT_SECONDARY, fontsize=9)
    ax_ratio.axvline(1.0, color=REFERENCE, linestyle=":", linewidth=1.4)
    ax_ratio.text(
        1.15,
        0.97,
        "$k=1$ (value used\nin training)",
        transform=ax_ratio.get_xaxis_transform(),
        color=TEXT_SECONDARY,
        fontsize=8,
        va="top",
    )
    _style_axis(
        ax_ratio,
        "temperature $k$",
        r"$\Delta \mathcal{L}_{\mathrm{GI}} / \Delta \mathcal{L}_{\mathrm{uniform}}$",
        "Gradient-informed advantage, saturating near $k=8$",
    )

    fig.suptitle(
        "Sampler benchmark on frozen resid_mlp2 (3 seeds, shaded $\\pm$1 std)",
        fontsize=12,
        color=TEXT_PRIMARY,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", type=str, required=True)
    parser.add_argument("--project", type=str, default="spd-gradient-informed-sampling")
    parser.add_argument(
        "--out", type=Path, default=Path("spd/scripts/plots/sweep_curves/sampler_k_sweep.png")
    )
    args = parser.parse_args()

    per_k = fetch_k_sweep(args.entity, args.project)

    print(f"{'k':>5}{'uniform':>13}{'GI':>13}{'ratio':>9}{'t':>9}{'p':>11}{'seeds':>7}")
    print("-" * 67)
    for k, arr in per_k.items():
        u, g, t, p = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        print(
            f"{k:>5g}{u.mean():>13.3e}{g.mean():>13.3e}"
            f"{(g / u).mean():>9.3f}{t.mean():>9.1f}{p.mean():>11.1e}{len(arr):>7}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot_k_sweep(per_k, args.out)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

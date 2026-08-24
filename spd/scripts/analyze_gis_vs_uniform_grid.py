#!/usr/bin/env python
"""Analyse the GIS-vs-uniform grid produced by run_local_grid_gis_vs_uniform_resid_mlp2.py.

Headline metric, per training-adversary budget:

    S = steps_uniform(target) / steps_gis(target)

where steps_X(target) is the first training step at which arm X's eval PGD loss
drops to `target`, and target is the worse arm's final loss (so both arms reach
it). S > 1 means GIS reached the same adversarial robustness in fewer training
steps.

CI-L0 is reported at those same steps as the control. Adversarial loss can be
lowered either by making the decomposition genuinely robust or by pushing ci
toward 1, which shrinks the (1 - ci) factor that is the adversary's only leverage
over the mask. Only the first is a result. If the two arms sit at materially
different CI-L0 when they hit the target, S is not interpretable on its own and
the Pareto panel is the honest read.

Usage:
    source .venv/bin/activate
    python spd/scripts/analyze_gis_vs_uniform_grid.py --entity <wandb_entity>
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import wandb

PGD_LOSS_KEY = "eval/loss/PGDReconLoss"
CI_L0_PREFIX = "eval/l0/0.1_"
RUN_NAME_PREFIX = "gis-grid_resid_mlp2_"

# Categorical slots 1 and 2 of the validated palette; assigned to the sampler
# identity and never reordered.
SAMPLER_COLOUR = {"continuous": "#2a78d6", "gradient_informed": "#eb6834"}
SAMPLER_LABEL = {"continuous": "Uniform", "gradient_informed": "Gradient-Informed"}

GRID_COLOUR = "#d8d8d4"
TEXT_PRIMARY = "#1a1a19"
TEXT_SECONDARY = "#5c5c57"


def fetch_runs(entity: str, project: str) -> dict[tuple[str, int, int], dict[str, np.ndarray]]:
    """Pull each grid run's PGD-loss and total-CI-L0 trajectory, keyed by arm."""
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")

    out: dict[tuple[str, int, int], dict[str, np.ndarray]] = {}
    for run in runs:
        if not run.name.startswith(RUN_NAME_PREFIX):
            continue
        suffix = run.name[len(RUN_NAME_PREFIX) :]
        sampling, pgd_tag, seed_tag = suffix.rsplit("_", 2)
        pgd_n_steps = int(pgd_tag.removeprefix("pgd-"))
        seed = int(seed_tag.removeprefix("seed-"))

        steps: list[int] = []
        pgd_losses: list[float] = []
        ci_l0s: list[float] = []
        for row in run.scan_history():
            if PGD_LOSS_KEY not in row or row[PGD_LOSS_KEY] is None:
                continue
            layer_l0s = [v for k, v in row.items() if k.startswith(CI_L0_PREFIX) and v is not None]
            if not layer_l0s:
                continue
            steps.append(row["_step"])
            pgd_losses.append(row[PGD_LOSS_KEY])
            ci_l0s.append(float(np.sum(layer_l0s)))

        assert steps, f"no eval points with both PGD loss and CI-L0 in run {run.name}"
        out[(sampling, pgd_n_steps, seed)] = {
            "step": np.array(steps),
            "pgd_loss": np.array(pgd_losses),
            "ci_l0": np.array(ci_l0s),
        }
    return out


def mean_over_seeds(
    runs: dict[tuple[str, int, int], dict[str, np.ndarray]], sampling: str, pgd_n_steps: int
) -> dict[str, np.ndarray]:
    """Average the trajectories of all seeds for one arm onto their shared step grid."""
    arms = [v for (s, n, _), v in runs.items() if s == sampling and n == pgd_n_steps]
    assert arms, f"no runs for sampling={sampling} pgd_n_steps={pgd_n_steps}"

    n_points = min(len(a["step"]) for a in arms)
    steps = arms[0]["step"][:n_points]
    assert all(np.array_equal(a["step"][:n_points], steps) for a in arms), (
        "seeds disagree on eval step grid"
    )

    return {
        "step": steps,
        "pgd_loss": np.mean([a["pgd_loss"][:n_points] for a in arms], axis=0),
        "pgd_loss_std": np.std([a["pgd_loss"][:n_points] for a in arms], axis=0),
        "ci_l0": np.mean([a["ci_l0"][:n_points] for a in arms], axis=0),
    }


def first_step_at_or_below(
    steps: np.ndarray, values: np.ndarray, target: float
) -> tuple[int, int] | None:
    """First (step, index) where `values` drops to `target`, using a running min.

    The running min keeps a single noisy eval point from being read as convergence.
    """
    running_min = np.minimum.accumulate(values)
    hits = np.flatnonzero(running_min <= target)
    if hits.size == 0:
        return None
    return int(steps[hits[0]]), int(hits[0])


@dataclass
class BudgetSummary:
    """Headline metric and control for one training-adversary budget."""

    pgd_n_steps: int
    target: float
    uniform_step: int
    gis_step: int
    speedup: float
    uniform_ci_l0: float
    gis_ci_l0: float
    uniform: dict[str, np.ndarray]
    gis: dict[str, np.ndarray]


def summarise_budget(
    runs: dict[tuple[str, int, int], dict[str, np.ndarray]], pgd_n_steps: int
) -> BudgetSummary:
    """Compute S and the CI-L0 control for one training-adversary budget."""
    uniform = mean_over_seeds(runs, "continuous", pgd_n_steps)
    gis = mean_over_seeds(runs, "gradient_informed", pgd_n_steps)

    # Target = the worse arm's final loss, so both arms are guaranteed to reach it.
    target = max(uniform["pgd_loss"][-1], gis["pgd_loss"][-1])

    uniform_hit = first_step_at_or_below(uniform["step"], uniform["pgd_loss"], target)
    gis_hit = first_step_at_or_below(gis["step"], gis["pgd_loss"], target)
    assert uniform_hit is not None and gis_hit is not None, (
        "both arms must reach a target defined as the worse arm's final loss"
    )

    uniform_step, uniform_idx = uniform_hit
    gis_step, gis_idx = gis_hit

    return BudgetSummary(
        pgd_n_steps=pgd_n_steps,
        target=float(target),
        uniform_step=uniform_step,
        gis_step=gis_step,
        speedup=uniform_step / max(gis_step, 1),
        uniform_ci_l0=float(uniform["ci_l0"][uniform_idx]),
        gis_ci_l0=float(gis["ci_l0"][gis_idx]),
        uniform=uniform,
        gis=gis,
    )


def _style_axis(ax: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=9)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=10, pad=8)
    ax.grid(True, color=GRID_COLOUR, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID_COLOUR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)


def plot_grid(summaries: list[BudgetSummary], out_path: str) -> None:
    """Three rows: adversarial loss over training, CI-L0 over training, and the Pareto."""
    n_cols = len(summaries)
    fig, axes = plt.subplots(3, n_cols, figsize=(6.5 * n_cols, 13), squeeze=False)

    for col, summary in enumerate(summaries):
        budget = summary.pgd_n_steps
        arms = [("continuous", summary.uniform), ("gradient_informed", summary.gis)]

        ax = axes[0][col]
        for sampling, data in arms:
            colour = SAMPLER_COLOUR[sampling]
            ax.plot(
                data["step"],
                data["pgd_loss"],
                color=colour,
                linewidth=2.0,
                marker="o",
                markersize=4,
                label=SAMPLER_LABEL[sampling],
            )
            ax.fill_between(
                data["step"],
                data["pgd_loss"] - data["pgd_loss_std"],
                data["pgd_loss"] + data["pgd_loss_std"],
                color=colour,
                alpha=0.12,
                linewidth=0,
            )
        ax.axhline(summary.target, color=TEXT_SECONDARY, linestyle="--", linewidth=1.0)
        ax.set_yscale("log")
        _style_axis(
            ax,
            "Training step",
            "Eval PGD recon loss (log, ↓ more robust)",
            f"Adversarial robustness — training PGD budget {budget}\n"
            f"S = {summary.speedup:.2f}×  (uniform {summary.uniform_step} "
            f"vs GIS {summary.gis_step} steps to target)",
        )
        ax.legend(fontsize=9, frameon=False)

        ax = axes[1][col]
        for sampling, data in arms:
            ax.plot(
                data["step"],
                data["ci_l0"],
                color=SAMPLER_COLOUR[sampling],
                linewidth=2.0,
                marker="o",
                markersize=4,
                label=SAMPLER_LABEL[sampling],
            )
        _style_axis(
            ax,
            "Training step",
            "Total CI-L0 (all layers)",
            f"Sparsity control — training PGD budget {budget}\n"
            f"at target: uniform {summary.uniform_ci_l0:.1f} vs GIS {summary.gis_ci_l0:.1f}",
        )
        ax.legend(fontsize=9, frameon=False)

        ax = axes[2][col]
        for sampling, data in arms:
            ax.plot(
                data["ci_l0"],
                data["pgd_loss"],
                color=SAMPLER_COLOUR[sampling],
                linewidth=2.0,
                marker="o",
                markersize=4,
                label=SAMPLER_LABEL[sampling],
            )
        ax.set_yscale("log")
        _style_axis(
            ax,
            "Total CI-L0 (→ less sparse)",
            "Eval PGD recon loss (log, ↓ more robust)",
            f"Pareto — training PGD budget {budget}\n"
            "lower-left is better; read S here if CI-L0 diverges",
        )
        ax.legend(fontsize=9, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", type=str, required=True, help="WandB entity")
    parser.add_argument("--project", type=str, default="spd-gradient-informed-sampling")
    parser.add_argument("--out", type=str, default="gis_vs_uniform_grid.png")
    args = parser.parse_args()

    runs = fetch_runs(entity=args.entity, project=args.project)
    assert runs, f"no runs named {RUN_NAME_PREFIX}* in {args.entity}/{args.project}"

    by_budget: dict[int, list[int]] = defaultdict(list)
    for _, pgd_n_steps, seed in runs:
        by_budget[pgd_n_steps].append(seed)
    print(f"Found {len(runs)} runs across budgets {sorted(by_budget)}")

    summaries = [summarise_budget(runs, budget) for budget in sorted(by_budget)]

    print(f"\n{'PGD budget':<12}{'target':<12}{'uniform':<10}{'GIS':<10}{'S':<8}{'CI-L0 (u/g)'}")
    print("-" * 68)
    for s in summaries:
        print(
            f"{s.pgd_n_steps:<12}{s.target:<12.6f}{s.uniform_step:<10}"
            f"{s.gis_step:<10}{s.speedup:<8.2f}"
            f"{s.uniform_ci_l0:.1f} / {s.gis_ci_l0:.1f}"
        )

    print("\nS > 1 means GIS reached the same adversarial robustness in fewer training steps.")
    print("If the CI-L0 columns differ materially, read the Pareto panel instead of S.")

    plot_grid(summaries, args.out)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

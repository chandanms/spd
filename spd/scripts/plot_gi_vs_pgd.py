#!/usr/bin/env python
"""Plot mean ± std training curves across seed runs from WandB sweeps.

Usage:
    python spd/scripts/plotting.py

Each group produces one figure with one subplot per metric.
Each condition within the group is plotted as a mean line with ±1 std shading.

Metrics are auto-discovered from the first run in each group, excluding figures
and charts. Override with METRICS to fix specific keys across all groups.

Edit the CONFIG section at the bottom to configure everything.
"""

import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

# (display_label, run_name_prefix, project)
ConditionSpec = tuple[str, str, str]
# (group_id, display_title, [ConditionSpec, ...])
GroupSpec = tuple[str, str, list[ConditionSpec]]

# Columns to skip when auto-discovering metrics
_SKIP_PATTERNS = re.compile(
    r"(grad_norm|figure|chart_table|bar_chart|causal_importance|ci_mean_per_component)"
)


def _is_plottable(col: str) -> bool:
    return (col.startswith("train/") or col.startswith("eval/")) and not _SKIP_PATTERNS.search(col)


def discover_metrics(entity: str, project: str, run_name_prefix: str) -> list[str]:
    """Return sorted list of plottable metric keys from the first matching seed run."""
    api = wandb.Api()
    runs = list(
        api.runs(
            f"{entity}/{project}",
            filters={"display_name": {"$regex": f"^{run_name_prefix}_seed-"}},
        )
    )
    assert runs, f"No runs found for prefix: {run_name_prefix}"
    for run in runs:
        df = run.history(samples=2, pandas=True)
        if df.empty:
            continue
        return sorted(c for c in df.columns if _is_plottable(c))
    return []


def fetch_run_histories(
    entity: str,
    project: str,
    run_name_prefix: str,
    metrics: list[str],
    samples: int,
) -> list[pd.DataFrame]:
    """Fetch history DataFrames (columns: _step + metrics) for all matching seed runs."""
    api = wandb.Api()
    runs = api.runs(
        f"{entity}/{project}",
        filters={"display_name": {"$regex": f"^{run_name_prefix}_seed-"}},
    )

    histories = []
    n_runs = 0
    for run in runs:
        n_runs += 1
        df = run.history(keys=metrics, samples=samples, pandas=True)
        if df.empty or "_step" not in df.columns:
            continue
        df = df.dropna(subset=["_step"])
        df["_step"] = df["_step"].astype(int)
        histories.append(df)

    print(f"  [{project}/{run_name_prefix}] {n_runs} runs")
    return histories


def align_and_aggregate(
    histories: list[pd.DataFrame],
    metrics: list[str],
) -> pd.DataFrame:
    """Interpolate each run onto a shared step grid, return mean/std columns per metric."""
    if not histories:
        return pd.DataFrame()

    all_steps = sorted({int(s) for df in histories for s in df["_step"].dropna()})
    grid = np.array(all_steps)

    interpolated: dict[str, list[np.ndarray]] = {m: [] for m in metrics}
    for df in histories:
        df_sorted = df.sort_values("_step")
        steps = df_sorted["_step"].values.astype(float)
        for metric in metrics:
            if metric not in df_sorted.columns:
                continue
            vals = df_sorted[metric].values.astype(float)
            mask = ~np.isnan(vals)
            if mask.sum() < 2:
                continue
            interpolated[metric].append(np.interp(grid, steps[mask], vals[mask]))

    result = pd.DataFrame({"_step": grid})
    for metric in metrics:
        arrays = interpolated[metric]
        if not arrays:
            result[f"{metric}__mean"] = np.nan
            result[f"{metric}__std"] = np.nan
        else:
            stacked = np.stack(arrays)
            result[f"{metric}__mean"] = stacked.mean(axis=0)
            result[f"{metric}__std"] = stacked.std(axis=0)
    return result


def plot_group(
    entity: str,
    group_id: str,
    title: str,
    conditions: list[ConditionSpec],
    metrics: list[str] | None,
    samples: int,
    output_dir: Path | None,
) -> None:
    """Fetch and plot one figure for a single group.

    Only metrics common to all conditions are plotted, so every condition has a
    line in every subplot.
    """
    # Pass 1: discover metrics per condition
    cond_discovered: dict[str, list[str]] = {}
    for cond_label, prefix, project in conditions:
        cond_metrics = metrics if metrics is not None else discover_metrics(entity, project, prefix)
        if cond_metrics:
            cond_discovered[cond_label] = cond_metrics

    if not cond_discovered:
        print(f"  No metrics found for group '{group_id}', skipping")
        return

    # Intersection so every condition has data in every subplot
    if metrics is not None:
        group_metrics = sorted(metrics)
    else:
        metric_sets = [set(m) for m in cond_discovered.values()]
        group_metrics = sorted(set.intersection(*metric_sets))

    if not group_metrics:
        print(f"  No common metrics for group '{group_id}', skipping")
        return

    print(f"  Intersection: {len(group_metrics)} metrics across {len(cond_discovered)} conditions")

    # Pass 2: fetch histories using intersection metrics only
    cond_aggs: dict[str, pd.DataFrame] = {}
    for cond_label, prefix, project in conditions:
        if cond_label not in cond_discovered:
            continue
        histories = fetch_run_histories(entity, project, prefix, group_metrics, samples)
        agg = align_and_aggregate(histories, group_metrics)
        cond_aggs[cond_label] = agg

    n_metrics = len(group_metrics)
    n_cols = min(4, n_metrics)
    n_rows = math.ceil(n_metrics / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), squeeze=False)

    for cond_label, agg in cond_aggs.items():
        if agg.empty:
            continue
        steps = agg["_step"].values
        for idx, metric in enumerate(group_metrics):
            row, col = divmod(idx, n_cols)
            ax = axes[row][col]
            mean_col = f"{metric}__mean"
            std_col = f"{metric}__std"
            if mean_col not in agg.columns or agg[mean_col].isna().all():
                continue
            mean = agg[mean_col].values
            std = agg[std_col].values
            (line,) = ax.plot(steps, mean, label=cond_label, linewidth=1.5)
            ax.fill_between(steps, mean - std, mean + std, alpha=0.15, color=line.get_color())
            ax.set_title(metric, fontsize=7)
            ax.set_xlabel("step", fontsize=7)
            ax.tick_params(labelsize=6)

    # Hide unused subplots
    for idx in range(n_metrics, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    n_legend_cols = min(4, len(labels))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=n_legend_cols,
        fontsize=9,
        bbox_transform=fig.transFigure,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{group_id}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved {path}")
    else:
        plt.show()

    plt.close(fig)


def plot_groups(
    entity: str,
    groups: list[GroupSpec],
    metrics: list[str] | None = None,
    samples: int = 500,
    output_dir: Path | None = None,
) -> None:
    """Plot one figure per group.

    Args:
        metrics: Fixed list of metric keys to plot for all groups. Pass None to
                 auto-discover and intersect per group (recommended — metrics differ by experiment).
    """
    for group_id, title, conditions in groups:
        print(f"\n[{group_id}]")
        plot_group(entity, group_id, title, conditions, metrics, samples, output_dir)


# ── CONFIG ────────────────────────────────────────────────────────────────────
ENTITY = "chandan-sreedhara"
DEFAULT_PROJECT = "spd-gradient-informed-sampling"

# Each group produces one figure. Each condition is a colored mean±std line.
# Condition format: (display_label, run_name_prefix, project)
GROUPS: list[GroupSpec] = [
    # (
    #     "tms_5-2",
    #     [
    #         ("pgd", "tms_5-2_pgd_revised", DEFAULT_PROJECT),
    #         ("continuous", "tms_5-2_continuous_sampling", DEFAULT_PROJECT),
    #         ("gradient_informed", "tms_5-2_gradient_informed_sampling", DEFAULT_PROJECT),
    #     ],
    # ),
    # (
    #     "tms_5-2-id",
    #     [
    #         ("pgd", "tms_5-2-id_pgd_revised", DEFAULT_PROJECT),
    #         ("continuous", "tms_5-2-id_continuous_sampling", DEFAULT_PROJECT),
    #         ("gradient_informed", "tms_5-2-id_gradient_informed_sampling", DEFAULT_PROJECT),
    #     ],
    # ),
    # (
    #     "tms_40-10",
    #     [
    #         ("pgd", "tms_40-10_pgd_revised", DEFAULT_PROJECT),
    #         ("continuous", "tms_40-10_continuous_sampling", DEFAULT_PROJECT),
    #         ("gradient_informed", "tms_40-10_gradient_informed_sampling", DEFAULT_PROJECT),
    #     ],
    # ),
    # (
    #     "tms_40-10-id",
    #     [
    #         ("pgd", "tms_40-10-id_pgd_revised", DEFAULT_PROJECT),
    #         ("continuous", "tms_40-10-id_continuous_sampling", DEFAULT_PROJECT),
    #         ("gradient_informed", "tms_40-10-id_gradient_informed_sampling", DEFAULT_PROJECT),
    #     ],
    # ),
    # (
    #     "resid_mlp2",
    #     [
    #         ("subset_loss_continuous", "resid_mlp2_continuous_subset", DEFAULT_PROJECT),
    #         ("layerwise_loss_continuous", "resid_mlp2_continuous", DEFAULT_PROJECT),
    #         ("subset_loss_pgd", "resid_mlp2_pgd", "spd"),
    #         (
    #             "subset_loss_gradient_informed",
    #             "resid_mlp2_subset_gradient_informed_2.0",
    #             DEFAULT_PROJECT,
    #         ),
    #     ],
    # ),
    (
        "resid_mlp2_pgd_vs_gradient-informed",
        "resid_mlp2: Comparison of hyperparameters in PGD and gradient informed sampling",
        [
            ("subset_loss_pgd", "resid_mlp2_pgd", "spd"),
            ("subset_loss_pgd_nsteps2.0", "resid_mlp2_pgd_nsteps_2.0", "spd"),
            ("subset_loss_pgd_nsteps3.0", "resid_mlp2_pgd_nsteps_3.0", DEFAULT_PROJECT),
            ("subset_loss_pgd_nsteps4.0", "resid_mlp2_pgd_nsteps_4.0", DEFAULT_PROJECT),
            ("subset_loss_pgd_nsteps5.0", "resid_mlp2_pgd_nsteps_5.0", DEFAULT_PROJECT),
            (
                "subset_loss_gradient_informed",
                "resid_mlp2_subset_gradient_informed_2.0",
                DEFAULT_PROJECT,
            ),
            (
                "subset_loss_gradient_informed_coeff3.0",
                "resid_mlp2_gradient_informed_3.0",
                DEFAULT_PROJECT,
            ),
            (
                "subset_loss_gradient_informed_coeff4.0",
                "resid_mlp2_gradient_informed_4.0",
                DEFAULT_PROJECT,
            ),
            (
                "subset_loss_gradient_informed_coeff5.0",
                "resid_mlp2_gradient_informed_5.0",
                DEFAULT_PROJECT,
            ),
            ("subset_loss_pgd_coeff3.0_nsteps1.0", "resid_mlp2_pgd_3.0", DEFAULT_PROJECT),
            # ("subset_loss_pgd_coeff4.0", "resid_mlp2_pgd_4.0", DEFAULT_PROJECT),
            ("subset_loss_pgd_coeff5.0_nsteps1.0", "resid_mlp2_pgd_5.0", DEFAULT_PROJECT),
        ],
    ),
]

# Metrics to plot. Set to None to auto-discover per group (recommended).
# Set to a list of keys to fix the same metrics for every group.
METRICS: list[str] | None = None

# Directory to save figures (one file per group), or None to show interactively
OUTPUT_DIR: Path | None = Path("plots/sweep_curves")
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    plot_groups(
        entity=ENTITY,
        groups=GROUPS,
        metrics=METRICS,
        output_dir=OUTPUT_DIR,
    )

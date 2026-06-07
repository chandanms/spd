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
# (group_label, [ConditionSpec, ...])
GroupSpec = tuple[str, list[ConditionSpec]]

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
    group_label: str,
    conditions: list[ConditionSpec],
    metrics: list[str] | None,
    samples: int,
    output_dir: Path | None,
) -> None:
    """Fetch and plot one figure for a single group.

    Each condition is fetched with its own metric keys to avoid empty responses
    when keys span metrics that are never co-logged. Subplots are created for
    the union of all metrics; conditions only appear in subplots where they have data.
    """
    # Per-condition: discover own metrics (or use fixed list) and fetch histories
    # cond_data: {cond_label: {metric: agg_df}}  — each cond has its own agg per metric
    cond_aggs: dict[str, pd.DataFrame] = {}
    union_metrics: set[str] = set()

    for cond_label, prefix, project in conditions:
        cond_metrics = metrics if metrics is not None else discover_metrics(entity, project, prefix)
        if not cond_metrics:
            continue
        union_metrics.update(cond_metrics)
        histories = fetch_run_histories(entity, project, prefix, cond_metrics, samples)
        agg = align_and_aggregate(histories, cond_metrics)
        cond_aggs[cond_label] = agg

    group_metrics = sorted(union_metrics)
    if not group_metrics:
        print(f"  No metrics found for group '{group_label}', skipping")
        return

    print(f"  Union: {len(group_metrics)} metrics across {len(cond_aggs)} conditions")

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
    fig.legend(handles, labels, loc="upper right", fontsize=9)
    fig.suptitle(group_label, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{group_label}.png"
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
                 auto-discover per group (recommended — metrics differ by experiment).
    """
    for group_label, conditions in groups:
        print(f"\n[{group_label}]")
        plot_group(entity, group_label, conditions, metrics, samples, output_dir)


# ── CONFIG ────────────────────────────────────────────────────────────────────
ENTITY = "chandan-sreedhara"
DEFAULT_PROJECT = "spd-gradient-informed-sampling"

# Each group produces one figure. Each condition is a colored mean±std line.
# Condition format: (display_label, run_name_prefix, project)
GROUPS: list[GroupSpec] = [
    (
        "tms_5-2",
        [
            ("continuous", "tms_5-2_continuous_sampling", DEFAULT_PROJECT),
            ("gradient", "tms_5-2_gradient_sampling", DEFAULT_PROJECT),
            ("gradient_informed", "tms_5-2_gradient_informed_sampling", DEFAULT_PROJECT),
        ],
    ),
    (
        "tms_5-2-id",
        [
            ("continuous", "tms_5-2-id_continuous_sampling", DEFAULT_PROJECT),
            ("gradient", "tms_5-2-id_gradient_sampling", DEFAULT_PROJECT),
            ("gradient_informed", "tms_5-2-id_gradient_informed_sampling", DEFAULT_PROJECT),
        ],
    ),
    (
        "tms_40-10",
        [
            ("continuous", "tms_40-10_continuous_sampling", DEFAULT_PROJECT),
            ("gradient", "tms_40-10_gradient_sampling", DEFAULT_PROJECT),
            ("gradient_informed", "tms_40-10_gradient_informed_sampling", DEFAULT_PROJECT),
        ],
    ),
    (
        "tms_40-10-id",
        [
            ("continuous", "tms_40-10-id_continuous_sampling", DEFAULT_PROJECT),
            ("gradient", "tms_40-10-id_gradient_sampling", DEFAULT_PROJECT),
            ("gradient_informed", "tms_40-10-id_gradient_informed_sampling", DEFAULT_PROJECT),
        ],
    ),
    (
        "resid_mlp2",
        [
            ("subset_loss_continuous", "resid_mlp2_pgd", "spd"),
            ("total_loss_gradient_informed", "resid_mlp2_gradient_informed", DEFAULT_PROJECT),
            (
                "subset_loss_gradient_informed",
                "resid_mlp2_subset_gradient_informed",
                DEFAULT_PROJECT,
            ),
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

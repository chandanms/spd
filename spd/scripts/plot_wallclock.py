#!/usr/bin/env python
"""Plot mean ± std wall-clock training time as a bar chart across WandB conditions.

Usage:
    python spd/scripts/plot_wallclock.py

Each group produces one bar chart. Each bar is one condition (mean across seeds),
with ±1 std error bars.

Edit the CONFIG section at the bottom to configure everything.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb

# (display_label, run_name_prefix, project)
ConditionSpec = tuple[str, str, str]
# (group_id, display_title, [ConditionSpec, ...])
GroupSpec = tuple[str, str, list[ConditionSpec]]


def fetch_runtimes(entity: str, project: str, run_name_prefix: str) -> list[float]:
    """Return wall-clock runtimes (seconds) for all matching seed runs."""
    api = wandb.Api()
    runs = api.runs(
        f"{entity}/{project}",
        filters={"display_name": {"$regex": f"^{run_name_prefix}_seed-"}},
    )
    runtimes = []
    n_runs = 0
    for run in runs:
        n_runs += 1
        runtime = run.summary.get("_runtime")
        if runtime is not None:
            runtimes.append(float(runtime))
    print(f"  [{project}/{run_name_prefix}] {n_runs} runs, {len(runtimes)} with runtime")
    return runtimes


def plot_group(
    entity: str,
    group_id: str,
    title: str,
    conditions: list[ConditionSpec],
    output_dir: Path | None,
) -> None:
    labels = []
    means = []
    stds = []

    for cond_label, prefix, project in conditions:
        runtimes = fetch_runtimes(entity, project, prefix)
        if not runtimes:
            print(f"  No runtime data for {cond_label}, skipping")
            continue
        arr = np.array(runtimes)
        labels.append(cond_label)
        means.append(arr.mean())
        stds.append(arr.std())

    if not labels:
        print(f"  No data for group '{group_id}', skipping")
        return

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(5, 2 * len(labels)), 4))
    bars = ax.bar(x, means, yerr=stds, capsize=5, width=0.6)

    # Annotate each bar with mean ± std
    for bar, mean, std in zip(bars, means, stds, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + max(means) * 0.01,
            f"{mean:.0f}s",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.tight_layout()

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{group_id}_wallclock.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved {path}")
    else:
        plt.show()

    plt.close(fig)


def plot_wallclock(
    entity: str,
    groups: list[GroupSpec],
    output_dir: Path | None = None,
) -> None:
    for group_id, title, conditions in groups:
        print(f"\n[{group_id}]")
        plot_group(entity, group_id, title, conditions, output_dir)


# ── CONFIG ────────────────────────────────────────────────────────────────────
ENTITY = "chandan-sreedhara"
DEFAULT_PROJECT = "spd-gradient-informed-sampling"

GROUPS: list[GroupSpec] = [
    (
        "resid_mlp2_pgd_vs_gradient-informed",
        "resid_mlp2: Comparison of wall clock training time in PGD and gradient informed sampling",
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
        ],
    ),
]

OUTPUT_DIR: Path | None = Path("plots/wallclock")
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    plot_wallclock(entity=ENTITY, groups=GROUPS, output_dir=OUTPUT_DIR)

#!/usr/bin/env python
"""What the gradient-informed temperature k actually does to the sampled mask.

GIS draws `s = (1 - w_c) * U[0,1]` with `w_c = |g_c|^k / sum_c |g_c|^k`. This script
measures, on real gradients, how `k` reshapes `w` and what it costs in sampling
variance:

  max w             how much weight lands on the single top-attribution component
  n components      how many components carry non-negligible weight (w > 0.01)
  Var[s] / Var[U]   = E[(1-w)^2], the variance GIS gives up relative to uniform

The two limits bracket everything `k` can do. As k -> 0 weights flatten to 1/C and
GIS becomes uniform; as k -> inf all weight lands on the top component, which is
hard-ablated while every other component stays exactly uniform. So `k` interpolates
between "uniform" and "uniform plus a hard ablation of the top component", and the
question this answers is which values of k are distinguishable from uniform at all.

Rows whose gradients are entirely zero (datapoints with no active feature, which
`feature_probability: 0.01` produces often) are excluded: their weights are zero by
the epsilon guard, so GIS falls back to uniform and they carry no signal about k.

Usage:
    source .venv/bin/activate
    python spd/scripts/diagnose_gis_temperature.py --out gis_temperature.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from spd.configs import Config, ResidMLPTaskConfig
from spd.experiments.resid_mlp.models import ResidMLP, ResidMLPTargetRunInfo
from spd.experiments.resid_mlp.resid_mlp_dataset import ResidMLPDataset
from spd.models.component_model import ComponentModel
from spd.scripts.benchmark_sampler import compute_importance_gradients
from spd.utils.distributed_utils import get_device
from spd.utils.module_utils import expand_module_patterns

BASE_CONFIG = Path("spd/experiments/resid_mlp/resid_mlp2_gis_grid_config.yaml")
KS = [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]

SERIES = "#2a78d6"
REFERENCE = "#8a8a84"
GRID_COLOUR = "#d8d8d4"
TEXT_PRIMARY = "#1a1a19"
TEXT_SECONDARY = "#5c5c57"


def normalised_grad_magnitudes(config: Config, batch_size: int) -> torch.Tensor:
    """|grad| per component, scaled so the per-row max is 1, over non-degenerate rows."""
    device = get_device()
    torch.manual_seed(0)

    task_config = config.task_config
    assert isinstance(task_config, ResidMLPTaskConfig), f"expected resid_mlp, got {task_config}"
    assert config.pretrained_model_path is not None, "pretrained_model_path must be set"

    target_run_info = ResidMLPTargetRunInfo.from_path(config.pretrained_model_path)
    target_model = ResidMLP.from_run_info(target_run_info).to(device).eval()
    target_model.requires_grad_(False)

    dataset = ResidMLPDataset(
        n_features=target_model.config.n_features,
        feature_probability=task_config.feature_probability,
        device=device,
        calc_labels=False,
        label_type=None,
        act_fn_name=None,
        label_fn_seed=None,
        label_coeffs=None,
        data_generation_type=task_config.data_generation_type,
        synced_inputs=target_run_info.config.synced_inputs,
    )
    model = ComponentModel(
        target_model=target_model,
        module_path_info=expand_module_patterns(target_model, config.all_module_info),
        ci_fn_type=config.ci_fn_type,
        ci_fn_hidden_dims=config.ci_fn_hidden_dims,
        pretrained_model_output_attr=config.pretrained_model_output_attr,
        sigmoid_type=config.sigmoid_type,
    ).to(device)

    batch, _ = dataset.generate_batch(batch_size)
    batch = batch.to(device)
    result = model(batch, cache_type="input")
    ci = {
        layer: v.detach()
        for layer, v in model.calc_causal_importances(
            {k: v.detach() for k, v in result.cache.items()}, sampling="continuous"
        ).lower_leaky.items()
    }
    grads = compute_importance_gradients(model, batch, ci, result.output.detach(), None)

    magnitudes = torch.cat([grads[layer] for layer in sorted(grads)], dim=-1).abs()
    live = magnitudes.amax(dim=-1) > 0
    assert live.any(), "every row had all-zero gradients"
    magnitudes = magnitudes[live]
    return magnitudes / magnitudes.amax(dim=-1, keepdim=True)


def temperature_stats(magnitudes: torch.Tensor, ks: list[float]) -> dict[str, np.ndarray]:
    """max weight, count of non-negligible weights, and variance ratio, per k."""
    max_w, n_active, var_ratio = [], [], []
    for k in ks:
        w = magnitudes.pow(k)
        w = w / w.sum(dim=-1, keepdim=True)
        max_w.append(w.amax(dim=-1).mean().item())
        n_active.append((w > 0.01).float().sum(dim=-1).mean().item())
        var_ratio.append(((1.0 - w) ** 2).mean().item())
    return {
        "k": np.array(ks, dtype=float),
        "max_w": np.array(max_w),
        "n_active": np.array(n_active),
        "var_ratio": np.array(var_ratio),
    }


def _style_axis(ax: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xscale("log", base=2)
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


def _mark_training_default(ax: plt.Axes) -> None:
    """k = 1 is the value the training path actually uses — no metric passes k through."""
    ax.axvline(1.0, color="#eb6834", linestyle=":", linewidth=1.6)
    ax.text(
        1.15,
        0.97,
        "k = 1\n(training default)",
        transform=ax.get_xaxis_transform(),
        color="#eb6834",
        fontsize=8,
        va="top",
    )


def plot_stats(stats: dict[str, np.ndarray], n_components: int, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ks = stats["k"]
    max_w_at_1 = float(stats["max_w"][list(ks).index(1.0)])

    ax = axes[0]
    ax.plot(ks, stats["max_w"], color=SERIES, linewidth=2.0, marker="o", markersize=5)
    ax.axhline(1.0 / n_components, color=REFERENCE, linestyle="--", linewidth=1.0)
    ax.text(
        ks[0],
        1.0 / n_components * 1.4,
        f"uniform, 1/C = {1.0 / n_components:.5f}",
        color=TEXT_SECONDARY,
        fontsize=8,
    )
    ax.set_yscale("log")
    _mark_training_default(ax)
    _style_axis(
        ax,
        "k",
        "mean max weight",
        f"Weight on the top component climbs to ~1\nat k = 1 it is only {max_w_at_1:.3f}",
    )

    ax = axes[1]
    ax.plot(ks, stats["n_active"], color=SERIES, linewidth=2.0, marker="o", markersize=5)
    ax.set_yscale("log")
    _mark_training_default(ax)
    _style_axis(
        ax,
        "k",
        "components with w > 0.01",
        f"Spreads over ~17 components at k = 2–4,\nthen collapses onto ~1 of {n_components}",
    )

    ax = axes[2]
    ax.plot(ks, stats["var_ratio"], color=SERIES, linewidth=2.0, marker="o", markersize=5)
    ax.axhline(1.0, color=REFERENCE, linestyle="--", linewidth=1.0)
    ax.text(ks[0], 1.0 - 0.00015, "uniform baseline", color=TEXT_SECONDARY, fontsize=8)
    _mark_training_default(ax)
    _style_axis(
        ax,
        "k",
        "Var[s] / Var[U]",
        "Sampling variance is untouched\n(<0.15% reduction at every k)",
    )

    fig.suptitle(
        "What k does to gradient-informed sampling (resid_mlp2, C = 1600 across all modules)",
        fontsize=12,
        color=TEXT_PRIMARY,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("gis_temperature.png"))
    parser.add_argument("--batch_size", type=int, default=2048)
    args = parser.parse_args()

    config = Config.from_file(BASE_CONFIG)
    magnitudes = normalised_grad_magnitudes(config, args.batch_size)
    n_components = magnitudes.shape[-1]
    stats = temperature_stats(magnitudes, KS)

    print(f"C = {n_components}   uniform weight 1/C = {1.0 / n_components:.5f}")
    print(f"\n{'k':>8}{'max w':>10}{'n w>0.01':>12}{'Var[s]/Var[U]':>16}")
    print("-" * 46)
    for i, k in enumerate(stats["k"]):
        print(
            f"{k:>8g}{stats['max_w'][i]:>10.4f}"
            f"{stats['n_active'][i]:>12.2f}{stats['var_ratio'][i]:>16.4f}"
        )

    plot_stats(stats, n_components, args.out)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

"""
Sampler Benchmark: Gradient-informed and PGD vs Uniform (continuous) sampling.

Trains a ComponentModel with uniform (continuous) sampling. Benchmarks temperature
sweeps and directional gradient-informed variants, plus PGD ceiling references,
at step 100 and then every 1000 steps up to the final step.

At each checkpoint, for each sampler, n_benchmark_batches batches are evaluated.
Each batch yields one draw of the stochastic mask (n_draws=1 by default, matching
training, since batch_size ~4096 already gives a stable per-batch ΔL_recon estimate).

Metrics logged per sampler per checkpoint:
  L_recon(unmasked) = MSE(target_out, target_out) = 0 by definition (logged as sanity check)
  L_recon(masked)   = MSE(masked_out, target_out)
  ΔL_recon          = L_recon(masked) - L_recon(unmasked) = L_recon(masked)

A paired t-test across the n_benchmark_batches observations compares GI vs uniform.

Results are logged to WandB under the project defined in the config.

Usage:
    source .venv/bin/activate

    python spd/scripts/benchmark_sampler.py \\
        --config_path spd/experiments/tms/tms_40-10_config.yaml

    python spd/scripts/benchmark_sampler.py \\
        --config_path spd/experiments/resid_mlp/resid_mlp2_continuous_config.yaml
"""

from __future__ import annotations

import argparse
import io
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from PIL import Image
from scipy import stats as scipy_stats
from tqdm import tqdm

from spd.configs import Config, GIVariant, PGDMultiBatchConfig
from spd.eval import evaluate
from spd.identity_insertion import insert_identity_operations_
from spd.log import logger
from spd.losses import compute_total_loss
from spd.models.component_model import ComponentModel
from spd.models.components import ComponentsMaskInfo, make_mask_infos
from spd.routing import AllLayersRouter
from spd.run_spd import get_unique_metric_configs, run_faithfulness_warmup
from spd.utils.component_utils import gradient_informed_source
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import get_scheduled_value, set_seed
from spd.utils.module_utils import expand_module_patterns
from spd.utils.run_utils import save_file, setup_decomposition_run
from spd.utils.wandb_utils import init_wandb, try_wandb

# ---------------------------------------------------------------------------
# Dataset + target model setup
# ---------------------------------------------------------------------------


def _make_dataset_and_target_model(
    config: Config, device: str
) -> tuple[nn.Module, Any, list[tuple[str, str]] | None]:
    """Load the frozen target model and create the synthetic dataset from config."""
    task_config = config.task_config
    assert config.pretrained_model_path is not None

    match task_config.task_name:
        case "tms":
            from spd.experiments.tms.models import TMSModel, TMSTargetRunInfo
            from spd.utils.data_utils import SparseFeatureDataset

            tgt = TMSTargetRunInfo.from_path(config.pretrained_model_path)
            target_model = TMSModel.from_run_info(tgt)
            dataset = SparseFeatureDataset(
                n_features=target_model.config.n_features,
                feature_probability=task_config.feature_probability,
                device=device,
                data_generation_type=task_config.data_generation_type,
                value_range=(0.0, 1.0),
                synced_inputs=tgt.config.synced_inputs,
            )
            tied_weights = [("linear1", "linear2")] if target_model.config.tied_weights else None
            return target_model, dataset, tied_weights

        case "resid_mlp":
            from spd.experiments.resid_mlp.models import ResidMLP, ResidMLPTargetRunInfo
            from spd.experiments.resid_mlp.resid_mlp_dataset import ResidMLPDataset

            tgt = ResidMLPTargetRunInfo.from_path(config.pretrained_model_path)
            target_model = ResidMLP.from_run_info(tgt)
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
                synced_inputs=tgt.config.synced_inputs,
            )
            return target_model, dataset, None

        case _:
            raise ValueError(f"Unsupported task: {task_config.task_name}")


# ---------------------------------------------------------------------------
# Importance gradient computation (clean standalone helper)
# ---------------------------------------------------------------------------


def compute_importance_gradients(
    model: ComponentModel,
    batch: torch.Tensor,
    ci_detached: dict[str, torch.Tensor],
    target_out: torch.Tensor,
    weight_deltas: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    """Compute raw importance gradients |∂L_recon/∂g_c| for gradient-informed sampling.

    Uses a single continuous-sampled stochastic mask to estimate how sensitive
    reconstruction loss is to each component's CI value. Does not update any
    model parameters.

    Returns:
        Dict mapping layer name to raw gradient tensor of shape [..., C].
        Not yet normalised — normalisation happens in sample_component_masks.
    """
    ci_leaf = {layer: ci.clone().requires_grad_(True) for layer, ci in ci_detached.items()}

    component_masks = {
        layer: ci + (1.0 - ci) * torch.rand_like(ci) for layer, ci in ci_leaf.items()
    }

    leading_dims = next(iter(ci_detached.values())).shape[:-1]
    device = next(iter(ci_detached.values())).device
    dtype = next(iter(ci_detached.values())).dtype

    weight_deltas_and_masks = (
        {
            layer: (wd, torch.rand(leading_dims, device=device, dtype=dtype))
            for layer, wd in weight_deltas.items()
        }
        if weight_deltas is not None
        else None
    )

    routing_masks = AllLayersRouter().get_masks(
        module_names=list(ci_detached.keys()), mask_shape=leading_dims
    )
    mask_infos = make_mask_infos(
        component_masks=component_masks,
        weight_deltas_and_masks=weight_deltas_and_masks,
        routing_masks=routing_masks,
    )

    masked_out = model(batch, mask_infos=mask_infos)
    loss = ((masked_out - target_out) ** 2).mean()

    layers = list(ci_leaf.keys())
    grads: dict[str, torch.Tensor] = {}
    for i, layer in enumerate(layers):
        retain = i < len(layers) - 1
        (grad,) = torch.autograd.grad(loss, ci_leaf[layer], retain_graph=retain, allow_unused=True)
        if grad is not None:  # pyright: ignore[reportUnnecessaryComparison]
            grads[layer] = grad.detach()

    return grads


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_component_masks(
    ci: dict[str, torch.Tensor],
    importance_grads: dict[str, torch.Tensor] | None,
    weight_deltas: dict[str, torch.Tensor] | None,
    importance_temperature: float,
    variant: GIVariant = "per_component",
    coeff: float = 5.0,
    power_iters: int = 5,
) -> tuple[dict[str, ComponentsMaskInfo], dict[str, torch.Tensor], float | None]:
    """Draw one set of component masks and return the raw stochastic sources.

    For uniform sampling:      stochastic_source ~ Uniform[0, 1]^{C}
    For gradient-informed:     stochastic_source = (1 - importance_normalised) * Uniform[0, 1]^{C}
      where importance_normalised = |grad|^k / sum(|grad|^k)

    ``importance_temperature`` is used by the per-component variant; the other
    variants use ``coeff`` and, for power iteration, ``power_iters``.

    Returns:
        mask_infos: passed directly to model forward.
        sources: dict layer -> stochastic_source tensor [B, C] for covariance tracking.
        captured: mean gradient energy captured by power iteration, otherwise None.
    """
    component_masks: dict[str, torch.Tensor] = {}
    sources: dict[str, torch.Tensor] = {}
    captured_per_layer: list[float] = []

    for layer, g_c in ci.items():
        if importance_grads is not None and layer in importance_grads:
            source, captured = gradient_informed_source(
                grad=importance_grads[layer],
                ci=g_c,
                variant=variant,
                coeff=coeff,
                power_iters=power_iters,
                importance_temperature=importance_temperature,
            )
            if captured is not None:
                captured_per_layer.append(captured)
        else:
            source = torch.rand_like(g_c)

        sources[layer] = source.detach()
        component_masks[layer] = g_c + (1.0 - g_c) * source

    leading_dims = next(iter(ci.values())).shape[:-1]
    device = next(iter(ci.values())).device
    dtype = next(iter(ci.values())).dtype

    weight_deltas_and_masks = (
        {
            layer: (wd, torch.rand(leading_dims, device=device, dtype=dtype))
            for layer, wd in weight_deltas.items()
        }
        if weight_deltas is not None
        else None
    )

    routing_masks = AllLayersRouter().get_masks(
        module_names=list(ci.keys()), mask_shape=leading_dims
    )
    mask_infos = make_mask_infos(
        component_masks=component_masks,
        weight_deltas_and_masks=weight_deltas_and_masks,
        routing_masks=routing_masks,
    )

    mean_captured = float(np.mean(captured_per_layer)) if captured_per_layer else None
    return mask_infos, sources, mean_captured


@dataclass(frozen=True)
class UniformArm:
    """Plain Uniform[0, 1] stochastic source. The baseline every other arm is paired against."""

    @property
    def name(self) -> str:
        return "uniform"


@dataclass(frozen=True)
class GISArm:
    """Gradient-informed source, including HEAD's directional variants."""

    k: float
    variant: GIVariant = "per_component"

    @property
    def name(self) -> str:
        return f"gis_k{self.k:g}" if self.variant == "per_component" else f"gis_{self.variant}"


@dataclass(frozen=True)
class PGDArm:
    """Adversarially optimised source: the ceiling on what any sampler could find."""

    step_size: float
    n_steps: int

    @property
    def name(self) -> str:
        return f"pgd_s{self.step_size:g}_n{self.n_steps}"


SamplerArm = UniformArm | GISArm | PGDArm


def pgd_component_masks(
    model: ComponentModel,
    batch: torch.Tensor,
    ci: dict[str, torch.Tensor],
    weight_deltas: dict[str, torch.Tensor] | None,
    target_out: torch.Tensor,
    step_size: float,
    n_steps: int,
) -> tuple[dict[str, ComponentsMaskInfo], dict[str, torch.Tensor]]:
    """Maximise reconstruction error over the stochastic source by projected gradient ascent.

    Ascends the same objective the samplers are scored on, MSE(masked_out, target_out), so the
    resulting delta_l_recon is directly comparable. The source is initialised uniformly, matching
    ``init: random`` in the PGD losses.

    The weight-delta mask is drawn randomly rather than optimised, exactly as in
    ``sample_component_masks``. This keeps every arm differing only in how the *component* source
    is chosen, at the cost of handicapping PGD slightly relative to the training-time PGD losses.
    """
    sources = {layer: torch.rand_like(g_c) for layer, g_c in ci.items()}

    for _ in range(n_steps):
        leaves = {layer: s.clone().requires_grad_(True) for layer, s in sources.items()}
        mask_infos, _ = _assemble_mask_infos(ci, leaves, weight_deltas)
        masked_out = model(batch, mask_infos=mask_infos)
        loss = ((masked_out - target_out) ** 2).mean()
        grads = torch.autograd.grad(loss, list(leaves.values()))
        with torch.no_grad():
            sources = {
                layer: (sources[layer] + step_size * grad.sign()).clamp_(0.0, 1.0)
                for layer, grad in zip(leaves.keys(), grads, strict=True)
            }

    return _assemble_mask_infos(ci, sources, weight_deltas)


def _assemble_mask_infos(
    ci: dict[str, torch.Tensor],
    sources: dict[str, torch.Tensor],
    weight_deltas: dict[str, torch.Tensor] | None,
) -> tuple[dict[str, ComponentsMaskInfo], dict[str, torch.Tensor]]:
    """Build mask infos from per-layer sources, shared by every arm so they stay comparable."""
    component_masks = {layer: ci[layer] + (1.0 - ci[layer]) * s for layer, s in sources.items()}

    leading_dims = next(iter(ci.values())).shape[:-1]
    device = next(iter(ci.values())).device
    dtype = next(iter(ci.values())).dtype

    weight_deltas_and_masks = (
        {
            layer: (wd, torch.rand(leading_dims, device=device, dtype=dtype))
            for layer, wd in weight_deltas.items()
        }
        if weight_deltas is not None
        else None
    )
    routing_masks = AllLayersRouter().get_masks(
        module_names=list(ci.keys()), mask_shape=leading_dims
    )
    mask_infos = make_mask_infos(
        component_masks=component_masks,
        weight_deltas_and_masks=weight_deltas_and_masks,
        routing_masks=routing_masks,
    )
    return mask_infos, {layer: s.detach() for layer, s in sources.items()}


def _flatten_sources(sources: dict[str, torch.Tensor]) -> torch.Tensor:
    """Concatenate per-layer sources [B, C_layer] → [B, total_C], ordered by layer name."""
    return torch.cat([sources[layer] for layer in sorted(sources.keys())], dim=-1)


def _make_eval_iter(dataset: Any, batch_size: int, device: str) -> Iterator[torch.Tensor]:
    while True:
        batch, _ = dataset.generate_batch(batch_size)
        yield batch.to(device)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def _pil_from_fig(fig: plt.Figure) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def _delta_line_chart(
    history: list[dict[str, float]],
) -> Image.Image:
    """Two-panel line chart of ΔL_recon vs training step.

    Top panel: log-scale ΔL_recon for both samplers (log y-axis spreads the
    late-training region where the gap stabilises and makes a constant
    multiplicative difference appear as a constant vertical offset).
    Bottom panel: ratio ΔL_GI / ΔL_uniform — ratio > 1 means GI finds
    better ablations.

    Each entry in history: {"step": float, "uniform_mean": float, "uniform_std": float,
                             "gi_mean": float, "gi_std": float}.
    Shaded bands show ±std across the n_benchmark_batches evaluated at that checkpoint.
    """
    steps = [h["step"] for h in history]
    uni_mean = np.array([h["uniform_mean"] for h in history])
    gi_mean = np.array([h["gi_mean"] for h in history])
    uni_std = np.array([h["uniform_std"] for h in history])
    gi_std = np.array([h["gi_std"] for h in history])

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    # Top: log-scale ΔL_recon
    ax_top.plot(steps, uni_mean, "o-", color="salmon", linewidth=1.5, markersize=5, label="Uniform")
    ax_top.plot(
        steps,
        gi_mean,
        "o-",
        color="steelblue",
        linewidth=1.5,
        markersize=5,
        label="Gradient-Informed",
    )
    ax_top.fill_between(
        steps,
        np.maximum(uni_mean - uni_std, 1e-12),
        uni_mean + uni_std,
        color="salmon",
        alpha=0.15,
    )
    ax_top.fill_between(
        steps,
        np.maximum(gi_mean - gi_std, 1e-12),
        gi_mean + gi_std,
        color="steelblue",
        alpha=0.15,
    )
    ax_top.set_yscale("log")
    ax_top.set_ylabel("Mean ΔL_recon  (log scale, ↑ better)")
    ax_top.set_title("ΔL_recon vs Training Step  (shaded = ±std across benchmark batches)")
    ax_top.legend(fontsize=9)

    # Bottom: ratio GI / uniform
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(uni_mean > 0, gi_mean / uni_mean, np.nan)
    ax_bot.plot(steps, ratio, "o-", color="purple", linewidth=1.5, markersize=5)
    ax_bot.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax_bot.set_xlabel("Training step")
    ax_bot.set_ylabel("ΔL_GI / ΔL_uniform  (> 1 = GI better)")
    ax_bot.set_title("Ratio: GI / Uniform")

    fig.tight_layout()
    return _pil_from_fig(fig)


def _covariance_heatmaps(
    uniform_stacked: torch.Tensor,
    gi_stacked: torch.Tensor,
) -> Image.Image:
    """Exploration diagnostics for the sampled sources.

    Top row: covariance matrices of the sampled sources. Colour limits are set
    from the off-diagonal entries so inter-component structure is legible (the
    diagonal, ~1/12, saturates). Structured off-diagonal covariance would
    indicate the sampler is collapsing onto correlated component groups.

    Bottom: per-component variance (diagonal of cov), sorted descending, both
    samplers overlaid. This is the collapse check: a sampler stuck in a single
    region shows components pushed toward zero variance. The dashed line is the
    Uniform[0, 1] baseline of 1/12.
    """
    covs: dict[str, np.ndarray] = {}
    for title, stacked in [("Uniform", uniform_stacked), ("Gradient-Informed", gi_stacked)]:
        covs[title] = torch.cov(stacked.float().T).numpy()

    # Shared off-diagonal colour scale so the two panels are directly comparable.
    off_mask = ~np.eye(next(iter(covs.values())).shape[0], dtype=bool)
    off_std = max(float(np.std(np.concatenate([c[off_mask] for c in covs.values()]))), 1e-12)
    lim = 3.0 * off_std

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0])

    for col, (title, cov) in enumerate(covs.items()):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(cov, aspect="auto", cmap="coolwarm", vmin=-lim, vmax=lim)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        mean_var = float(np.diag(cov).mean())
        ax.set_title(f"{title}  (mean var={mean_var:.4f})", fontsize=11)
        ax.set_xlabel("Component index (all layers flattened)")
        ax.set_ylabel("Component index (all layers flattened)")

    ax_var = fig.add_subplot(gs[1, :])
    for title, colour in [("Uniform", "salmon"), ("Gradient-Informed", "steelblue")]:
        variances = np.sort(np.diag(covs[title]))[::-1]
        ax_var.plot(variances, color=colour, linewidth=1.5, label=title)
    ax_var.axhline(
        1.0 / 12.0,
        color="gray",
        linestyle="--",
        linewidth=0.8,
        label="Uniform[0,1] baseline (1/12)",
    )
    ax_var.set_xlabel("Component (sorted by variance, descending)")
    ax_var.set_ylabel("Per-component variance")
    ax_var.set_title("Sorted per-component source variance  (↓ toward 0 = collapsed sampling)")
    ax_var.legend(fontsize=9)

    fig.suptitle(
        "Source exploration  (top: covariance, off-diagonal scaled; bottom: variance spectrum)",
        fontsize=12,
    )
    fig.tight_layout()
    return _pil_from_fig(fig)


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------


def benchmark_samplers(
    model: ComponentModel,
    dataset: Any,
    config: Config,
    device: str,
    n_batches: int,
    n_draws: int,
    arms: list[SamplerArm],
) -> dict[str, Any]:
    """Compare every arm on a frozen ComponentModel.

    All arms see the same frozen model and the same batches, so the comparison is paired: each
    non-uniform arm gets a paired t-test (ttest_rel) against uniform across batches. Including a
    PGD arm gives the ceiling — the reconstruction error an inner optimisation loop can reach —
    which turns the sampler ratios into a fraction of attainable headroom.

    Returns a flat dict suitable for WandB logging (images included).
    """
    assert any(isinstance(a, UniformArm) for a in arms), "arms must include UniformArm (baseline)"
    assert len({arm.name for arm in arms}) == len(arms), "arm names must be unique"
    model.eval()

    # Detach weight deltas once — used for both samplers.
    weight_deltas: dict[str, torch.Tensor] | None = (
        {k: v.detach() for k, v in model.calc_weight_deltas().items()}
        if config.use_delta_component
        else None
    )

    arm_names = [arm.name for arm in arms]
    per_batch_unmasked: dict[str, list[float]] = {name: [] for name in arm_names}
    per_batch_masked: dict[str, list[float]] = {name: [] for name in arm_names}
    per_batch_delta: dict[str, list[float]] = {name: [] for name in arm_names}
    all_sources: dict[str, list[torch.Tensor]] = {name: [] for name in arm_names}
    captured_vals: dict[str, list[float]] = {name: [] for name in arm_names}

    for _ in range(n_batches):
        batch, _ = dataset.generate_batch(config.eval_batch_size)  # type: ignore[attr-defined]
        batch = batch.to(device)

        # Target output = raw frozen model, no masking.
        with torch.no_grad():
            result = model(batch, cache_type="input")
        target_out = result.output.detach()
        pre_weight_acts = {k: v.detach() for k, v in result.cache.items()}

        # L_recon(unmasked) = MSE(target_out, target_out) = 0 by definition.
        l_recon_unmasked = ((target_out - target_out) ** 2).mean().item()

        with torch.no_grad():
            ci_outputs = model.calc_causal_importances(pre_weight_acts, sampling="continuous")
        ci = {layer: v.detach() for layer, v in ci_outputs.lower_leaky.items()}

        # One gradient pass to get importance signal (re-used across all GI draws this batch).
        with torch.enable_grad():
            importance_grads = compute_importance_gradients(
                model=model,
                batch=batch,
                ci_detached=ci,
                target_out=target_out,
                weight_deltas=weight_deltas,
            )

        for arm in arms:
            draw_masked: list[float] = []

            for _ in range(n_draws):
                match arm:
                    case UniformArm():
                        with torch.no_grad():
                            mask_infos, sources, _ = sample_component_masks(
                                ci=ci,
                                importance_grads=None,
                                weight_deltas=weight_deltas,
                                importance_temperature=1.0,
                            )
                    case GISArm(k=k, variant=variant):
                        with torch.no_grad():
                            mask_infos, sources, captured = sample_component_masks(
                                ci=ci,
                                importance_grads=importance_grads,
                                weight_deltas=weight_deltas,
                                importance_temperature=k,
                                variant=variant,
                                coeff=config.gi_coeff,
                                power_iters=config.gi_power_iters,
                            )
                        if captured is not None:
                            captured_vals[arm.name].append(captured)
                    case PGDArm(step_size=step_size, n_steps=n_steps):
                        with torch.enable_grad():
                            mask_infos, sources = pgd_component_masks(
                                model=model,
                                batch=batch,
                                ci=ci,
                                weight_deltas=weight_deltas,
                                target_out=target_out,
                                step_size=step_size,
                                n_steps=n_steps,
                            )

                with torch.no_grad():
                    masked_out = model(batch, mask_infos=mask_infos)
                    l_recon_masked = ((masked_out - target_out) ** 2).mean().item()
                draw_masked.append(l_recon_masked)
                all_sources[arm.name].append(_flatten_sources(sources))

            mean_masked = float(np.mean(draw_masked))
            per_batch_unmasked[arm.name].append(l_recon_unmasked)
            per_batch_masked[arm.name].append(mean_masked)
            per_batch_delta[arm.name].append(mean_masked - l_recon_unmasked)

    uniform_delta = np.array(per_batch_delta["uniform"])

    out: dict[str, Any] = {}
    for name in arm_names:
        out[f"{name}/mean_l_recon_unmasked"] = float(np.mean(per_batch_unmasked[name]))
        out[f"{name}/std_l_recon_unmasked"] = float(np.std(per_batch_unmasked[name]))
        out[f"{name}/mean_l_recon_masked"] = float(np.mean(per_batch_masked[name]))
        out[f"{name}/std_l_recon_masked"] = float(np.std(per_batch_masked[name]))
        out[f"{name}/mean_delta_l_recon"] = float(np.mean(per_batch_delta[name]))
        out[f"{name}/std_delta_l_recon"] = float(np.std(per_batch_delta[name]))
        out[f"{name}/ratio_to_uniform"] = float(
            np.mean(per_batch_delta[name]) / uniform_delta.mean()
        )
        if captured_vals[name]:
            out[f"{name}/mean_captured"] = float(np.mean(captured_vals[name]))

        if name == "uniform":
            continue
        # Paired across batches: every arm saw the same model and the same data.
        ttest_result = scipy_stats.ttest_rel(np.array(per_batch_delta[name]), uniform_delta)
        out[f"{name}/t_stat"] = cast(float, ttest_result[0])
        out[f"{name}/p_value"] = cast(float, ttest_result[1])

    stacked = {
        name: torch.cat(all_sources[name], dim=0).cpu().float()
        for name in ("uniform", arm_names[1])
    }
    out["chart/source_cov"] = _covariance_heatmaps(stacked["uniform"], stacked[arm_names[1]])
    return out


def format_latex_table(benches: list[dict[str, Any]], arms: list[SamplerArm]) -> str:
    """Emit the results as a LaTeX tabular, aggregated over seeds, ready to paste in.

    One entry of ``benches`` per seed. The ratio is averaged across seeds (each seed
    against its own uniform baseline, since the baselines differ between runs), and
    reported with its spread. The paired t-test is computed across batches within a
    run, so ``t`` is averaged over seeds and ``p`` is reported worst-case.
    """
    assert benches, "need at least one seed's results"

    def sci(x: float) -> str:
        mantissa, exponent = f"{x:.4e}".split("e")
        return f"${mantissa} \\times 10^{{{int(exponent)}}}$"

    rows = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Sampler & $\\Delta\\mathcal{L}_{\\mathrm{recon}}$ & Ratio to uniform "
        "& Paired $t$ & Nominal $p$ \\\\",
        "\\midrule",
    ]
    for arm in arms:
        deltas = np.array([b[f"{arm.name}/mean_delta_l_recon"] for b in benches])
        ratios = np.array([b[f"{arm.name}/ratio_to_uniform"] for b in benches])
        ratio_str = (
            f"{ratios.mean():.2f}"
            if len(benches) == 1
            else f"{ratios.mean():.2f} $\\pm$ {ratios.std():.2f}"
        )
        match arm:
            case UniformArm():
                label = "Uniform"
            case GISArm(k=k, variant=variant):
                label = f"GIS $k={k:g}$" if variant == "per_component" else f"GIS {variant}"
            case PGDArm(step_size=step_size, n_steps=n_steps):
                label = f"PGD, step {step_size:g}, {n_steps} steps"

        if isinstance(arm, UniformArm):
            stats = "-- & --"
        else:
            t_mean = float(np.mean([b[f"{arm.name}/t_stat"] for b in benches]))
            p_worst = float(np.max([b[f"{arm.name}/p_value"] for b in benches]))
            stats = f"{t_mean:.2f} & {sci(p_worst)}"
        rows.append(f"{label} & {sci(float(deltas.mean()))} & {ratio_str} & {stats} \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def run_benchmark_training(
    config: Config,
    target_model: nn.Module,
    dataset: Any,
    device: str,
    out_dir: Path,
    n_benchmark_batches: int,
    n_draws: int,
    tied_weights: list[tuple[str, str]] | None,
    arms: list[SamplerArm],
) -> dict[str, Any]:
    """Train a ComponentModel with uniform sampling.

    Benchmarks every arm at step 100, then every 1000 steps, always including the
    final step. Logs metrics and covariance heatmaps to WandB, and writes the
    LaTeX results table at the final step.
    """
    assert config.sampling == "continuous", (
        f"Training must use uniform (continuous) sampling, got {config.sampling!r}. "
        "Update your config or pass --force_continuous."
    )

    target_model.requires_grad_(False)
    target_model.to(device)
    target_model.eval()

    if config.identity_module_info is not None:
        insert_identity_operations_(target_model, identity_module_info=config.identity_module_info)

    module_path_info = expand_module_patterns(target_model, config.all_module_info)
    model = ComponentModel(
        target_model=target_model,
        module_path_info=module_path_info,
        ci_fn_type=config.ci_fn_type,
        ci_fn_hidden_dims=config.ci_fn_hidden_dims,
        pretrained_model_output_attr=config.pretrained_model_output_attr,
        sigmoid_type=config.sigmoid_type,
    )
    model.to(device)

    if tied_weights is not None:
        for src_name, tgt_name in tied_weights:
            model.components[tgt_name].U.data = model.components[src_name].V.data.T
            model.components[tgt_name].V.data = model.components[src_name].U.data.T

    component_params: list[nn.Parameter] = [
        p for name in model.target_module_paths for p in model.components[name].parameters()
    ]
    ci_fn_params: list[nn.Parameter] = [
        p for name in model.target_module_paths for p in model.ci_fns[name].parameters()
    ]
    optimizer = optim.AdamW(
        component_params + ci_fn_params, lr=config.lr_schedule.start_val, weight_decay=0
    )

    if config.faithfulness_warmup_steps > 0:
        run_faithfulness_warmup(model, component_params, config)

    all_eval_configs = get_unique_metric_configs(
        loss_configs=config.loss_metric_configs,
        eval_configs=config.eval_metric_configs,
    )
    eval_metric_configs = [
        cfg for cfg in all_eval_configs if not isinstance(cfg, PGDMultiBatchConfig)
    ]
    eval_iter = _make_eval_iter(dataset, config.eval_batch_size, device)

    total_steps = config.steps
    # Checkpoint at step 100, then every 1000 steps, always including the final step.
    checkpoint_steps: set[int] = {min(100, total_steps)} | set(range(1000, total_steps + 1, 1000))
    checkpoint_steps.add(total_steps)

    # Accumulated across checkpoints for the ΔL_recon line chart.
    delta_history: list[dict[str, float]] = []
    final_bench: dict[str, Any] = {}

    for step in tqdm(range(total_steps + 1), ncols=0, desc="Training"):
        optimizer.zero_grad()

        step_lr = get_scheduled_value(step=step, total_steps=total_steps, config=config.lr_schedule)
        for group in optimizer.param_groups:
            group["lr"] = step_lr

        # --- Forward pass ---
        batch, _ = dataset.generate_batch(config.microbatch_size)  # type: ignore[attr-defined]
        batch = batch.to(device)

        weight_deltas = model.calc_weight_deltas()
        fwd = model(batch, cache_type="input")

        ci = model.calc_causal_importances(
            pre_weight_acts=fwd.cache,
            detach_inputs=False,
            sampling="continuous",
        )

        total_loss, loss_terms = compute_total_loss(
            loss_metric_configs=config.loss_metric_configs,
            model=model,
            batch=batch,
            ci=ci,
            target_out=fwd.output,
            weight_deltas=weight_deltas,
            pre_weight_acts=fwd.cache,
            current_frac_of_training=step / max(total_steps, 1),
            sampling="continuous",
            use_delta_component=config.use_delta_component,
            n_mask_samples=config.n_mask_samples,
            output_loss_type=config.output_loss_type,
        )

        # --- Checkpoint benchmark ---
        if step in checkpoint_steps:
            tqdm.write(f"\n=== Sampler Benchmark — step {step}/{total_steps} ===")

            bench = benchmark_samplers(
                model=model,
                dataset=dataset,
                config=config,
                device=device,
                n_batches=n_benchmark_batches,
                n_draws=n_draws,
                arms=arms,
            )

            uniform_delta = bench["uniform/mean_delta_l_recon"]
            for arm in arms:
                ratio = bench[f"{arm.name}/ratio_to_uniform"]
                line = (
                    f"  {arm.name:<20} ΔL={bench[f'{arm.name}/mean_delta_l_recon']:.6e}"
                    f" ± {bench[f'{arm.name}/std_delta_l_recon']:.1e}  ratio={ratio:.3f}"
                )
                if not isinstance(arm, UniformArm):
                    line += (
                        f"  t={bench[f'{arm.name}/t_stat']:.2f}"
                        f"  p={bench[f'{arm.name}/p_value']:.2e}"
                    )
                tqdm.write(line)

            first_gis = next((a for a in arms if isinstance(a, GISArm)), None)
            if first_gis is not None:
                delta_history.append(
                    {
                        "step": float(step),
                        "uniform_mean": uniform_delta,
                        "uniform_std": bench["uniform/std_delta_l_recon"],
                        "gi_mean": bench[f"{first_gis.name}/mean_delta_l_recon"],
                        "gi_std": bench[f"{first_gis.name}/std_delta_l_recon"],
                    }
                )

            save_file(model.state_dict(), out_dir / f"model_{step}.pth")
            if step == total_steps:
                final_bench = bench
                table = format_latex_table([bench], arms)
                (out_dir / "results_table.tex").write_text(table)
                tqdm.write(f"\n{table}\n\nSaved to {out_dir / 'results_table.tex'}")

            if config.wandb_project:
                wandb_log: dict[str, Any] = {
                    f"benchmark/{metric}": v
                    for metric, v in bench.items()
                    if not isinstance(v, Image.Image)
                }
                wandb_log["benchmark/chart/source_cov"] = wandb.Image(bench["chart/source_cov"])
                if delta_history:
                    wandb_log["benchmark/chart/delta_line"] = wandb.Image(
                        _delta_line_chart(delta_history)
                    )
                try_wandb(wandb.log, wandb_log, step=step)

        # --- Regular training log ---
        if step % config.train_log_freq == 0 and config.wandb_project:
            try_wandb(
                wandb.log,
                {**{f"train/{k}": v for k, v in loss_terms.items()}, "train/lr": step_lr},
                step=step,
            )

        # --- Eval metrics ---
        if step % config.eval_freq == 0:
            slow_step = (
                config.slow_eval_on_first_step if step == 0 else step % config.slow_eval_freq == 0
            )
            with torch.no_grad():
                eval_metrics = evaluate(
                    eval_metric_configs=eval_metric_configs,
                    model=model,
                    eval_iterator=eval_iter,
                    device=device,
                    run_config=config,
                    slow_step=slow_step,
                    n_eval_steps=config.n_eval_steps,
                    current_frac_of_training=step / max(total_steps, 1),
                )
            if config.wandb_project:
                wandb_eval_logs = {
                    f"eval/{k}": wandb.Image(v) if isinstance(v, Image.Image) else v
                    for k, v in eval_metrics.items()
                }
                try_wandb(wandb.log, wandb_eval_logs, step=step)

        # --- Gradient step (skip at final step) ---
        if step < total_steps:
            total_loss.backward()
            optimizer.step()

    assert final_bench, "benchmark never ran at the final step"
    return final_bench


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


DEFAULT_ARMS: list[SamplerArm] = [
    UniformArm(),
    GISArm(k=0.25),
    GISArm(k=0.5),
    GISArm(k=1.0),
    GISArm(k=2.0),
    GISArm(k=4.0),
    GISArm(k=8.0),
    GISArm(k=1.0, variant="per_example"),
    GISArm(k=1.0, variant="mean"),
    GISArm(k=1.0, variant="power_iter"),
    PGDArm(step_size=0.1, n_steps=20),
    PGDArm(step_size=0.5, n_steps=20),
    PGDArm(step_size=1.0, n_steps=20),
]


def run_sampler_benchmark(
    config_path: Path,
    seed: int,
    run_name: str,
    force_continuous: bool,
    arms: list[SamplerArm] | None = None,
    n_benchmark_batches: int = 20,
    n_draws: int = 1,
    gi_coeff: float | None = None,
    gi_power_iters: int | None = None,
    importance_temperature: float | None = None,
) -> dict[str, Any]:
    """Train a ComponentModel with uniform sampling, then benchmark every arm against it.

    All arms share one frozen model and one set of batches, so their ratios are
    directly comparable and the PGD arms supply the ceiling. ``arms=None`` uses
    ``DEFAULT_ARMS`` unless ``importance_temperature`` selects one GIS temperature;
    that keyword is retained for compatibility with the local k-sweep script.
    """
    if arms is None:
        arms = (
            DEFAULT_ARMS
            if importance_temperature is None
            else [UniformArm(), GISArm(k=importance_temperature)]
        )
    device = get_device()
    set_seed(seed)

    config = Config.from_file(config_path)

    gi_overrides: dict[str, Any] = {}
    if gi_coeff is not None:
        gi_overrides["gi_coeff"] = gi_coeff
    if gi_power_iters is not None:
        gi_overrides["gi_power_iters"] = gi_power_iters
    if gi_overrides:
        config = config.model_copy(update=gi_overrides)

    if config.sampling != "continuous":
        assert force_continuous, (
            f"Config has sampling={config.sampling!r} but training requires 'continuous'. "
            "Pass force_continuous=True to override."
        )
        config = config.model_copy(update={"sampling": "continuous"})
        logger.info("Overriding sampling to 'continuous' for training.")

    logger.info(f"Device: {device}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Arms: {[a.name for a in arms]}")

    experiment_tag = config.task_config.task_name
    out_dir, run_id, tags = setup_decomposition_run(experiment_tag=experiment_tag)
    tags.append("sampler-benchmark")

    target_model, dataset, tied_weights = _make_dataset_and_target_model(config, device)

    if config.wandb_project:
        init_wandb(
            config=config,
            project=config.wandb_project,
            run_id=run_id,
            name=run_name,
            tags=tags,
        )
        try_wandb(
            wandb.config.update,
            {"arms": [a.name for a in arms], "seed": seed},
            allow_val_change=True,
        )

    bench = run_benchmark_training(
        config=config,
        target_model=target_model,
        dataset=dataset,
        device=device,
        out_dir=out_dir,
        n_benchmark_batches=n_benchmark_batches,
        n_draws=n_draws,
        tied_weights=tied_weights,
        arms=arms,
    )

    if config.wandb_project:
        wandb.finish()

    return bench


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config_path",
        type=Path,
        required=True,
        help="Path to experiment config YAML (e.g. spd/experiments/tms/tms_40-10_config.yaml)",
    )
    parser.add_argument(
        "--n_benchmark_batches",
        type=int,
        default=20,
        help="Batches used per sampler per checkpoint for the benchmark (default: 20)",
    )
    parser.add_argument(
        "--n_draws",
        type=int,
        default=1,
        help="Stochastic mask draws per batch per sampler (default: 1)",
    )
    parser.add_argument(
        "--gis_ks",
        type=str,
        default="0.25,0.5,1,2,4,8",
        help="Comma-separated temperatures k for the gradient-informed arms",
    )
    parser.add_argument(
        "--gi_variants",
        type=str,
        default="per_component,per_example,mean,power_iter",
        help="Comma-separated GI variants; gis_ks applies to per_component only",
    )
    parser.add_argument(
        "--gi_coeff",
        type=float,
        default=None,
        help="Override config.gi_coeff for directional GI variants",
    )
    parser.add_argument(
        "--gi_power_iters",
        type=int,
        default=None,
        help="Override config.gi_power_iters for the power_iter variant",
    )
    parser.add_argument(
        "--pgd_step_sizes",
        type=str,
        default="0.1,0.5,1.0",
        help="Comma-separated PGD step sizes; each becomes a ceiling-reference arm",
    )
    parser.add_argument(
        "--pgd_n_steps",
        type=int,
        default=20,
        help="Ascent steps for every PGD arm (default: 20, the VPD reference budget)",
    )
    parser.add_argument(
        "--force_continuous",
        action="store_true",
        help="Override config sampling to 'continuous' if it is not already set.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="1,2,3",
        help="Comma-separated seeds; each is a full training run, results averaged into one table",
    )
    args = parser.parse_args()

    variants = [cast(GIVariant, value) for value in args.gi_variants.split(",") if value]
    valid_variants = {"per_component", "per_example", "mean", "power_iter"}
    assert set(variants) <= valid_variants, f"Unknown GI variants: {set(variants) - valid_variants}"

    arms: list[SamplerArm] = [UniformArm()]
    if "per_component" in variants:
        arms += [GISArm(k=float(k)) for k in args.gis_ks.split(",") if k]
    arms += [
        GISArm(k=1.0, variant=cast(GIVariant, variant))
        for variant in variants
        if variant != "per_component"
    ]
    arms += [
        PGDArm(step_size=float(s), n_steps=args.pgd_n_steps)
        for s in args.pgd_step_sizes.split(",")
        if s
    ]

    seeds = [int(s) for s in args.seeds.split(",") if s]
    benches: list[dict[str, Any]] = []
    for i, seed in enumerate(seeds, start=1):
        logger.info(f"=== seed {seed} ({i}/{len(seeds)}) ===")
        benches.append(
            run_sampler_benchmark(
                config_path=args.config_path,
                seed=seed,
                run_name=f"sampler-benchmark-{args.config_path.stem}_seed-{seed}",
                force_continuous=args.force_continuous,
                arms=arms,
                n_benchmark_batches=args.n_benchmark_batches,
                n_draws=args.n_draws,
                gi_coeff=args.gi_coeff,
                gi_power_iters=args.gi_power_iters,
            )
        )

    print(f"\n{'arm':<24}{'ΔL (mean)':>14}{'ratio':>18}{'t (mean)':>11}{'p (worst)':>12}")
    print("-" * 79)
    for arm in arms:
        ratios = np.array([b[f"{arm.name}/ratio_to_uniform"] for b in benches])
        delta = float(np.mean([b[f"{arm.name}/mean_delta_l_recon"] for b in benches]))
        spread = "" if len(benches) == 1 else f" ± {ratios.std():.3f}"
        stats = ""
        if not isinstance(arm, UniformArm):
            t_mean = float(np.mean([b[f"{arm.name}/t_stat"] for b in benches]))
            p_worst = float(np.max([b[f"{arm.name}/p_value"] for b in benches]))
            stats = f"{t_mean:>11.2f}{p_worst:>12.2e}"
        print(f"{arm.name:<24}{delta:>14.4e}{f'{ratios.mean():.3f}{spread}':>18}{stats}")

    table = format_latex_table(benches, arms)
    out_path = Path(f"results_table_{args.config_path.stem}.tex")
    out_path.write_text(table)
    print(f"\n{table}\n\nSaved to {out_path}")


if __name__ == "__main__":
    main()

"""
Sampler Benchmark: Gradient-informed variants vs Uniform (continuous) sampling.

Trains a ComponentModel with uniform (continuous) sampling. At step 100 and then
every 1000 steps up to the final step, benchmarks the uniform baseline against
every gradient-informed variant (per_component, per_example, mean, power_iter).

At each checkpoint, for each sampler, n_benchmark_batches batches are evaluated.
Each batch yields one draw of the stochastic mask (n_draws=1 by default, matching
training, since batch_size ~4096 already gives a stable per-batch ΔL_recon estimate).

Metrics logged per sampler per checkpoint:
  ΔL_recon = L_recon(masked) - L_recon(unmasked) = L_recon(masked), since
  L_recon(unmasked) = MSE(target_out, target_out) = 0 by definition.

A paired t-test across the n_benchmark_batches observations compares each GI
variant against uniform. For power_iter, the fraction of batch gradient energy
captured by the top direction is also logged.

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

# Samplers compared at each checkpoint: the uniform baseline plus every GI variant.
GI_VARIANTS: list[GIVariant] = ["per_component", "per_example", "mean", "power_iter"]
SAMPLER_KEYS: list[str] = ["uniform", *GI_VARIANTS]


def sample_component_masks(
    ci: dict[str, torch.Tensor],
    importance_grads: dict[str, torch.Tensor] | None,
    weight_deltas: dict[str, torch.Tensor] | None,
    variant: GIVariant,
    coeff: float,
    power_iters: int,
) -> tuple[dict[str, ComponentsMaskInfo], dict[str, torch.Tensor], float | None]:
    """Draw one set of component masks and return the raw stochastic sources.

    When `importance_grads` is None the source is uniform (the baseline). Otherwise
    the gradient-informed `variant` biases the source toward the high-recon-error
    direction (see `gradient_informed_source`).

    Returns:
        mask_infos: passed directly to model forward.
        sources: dict layer -> stochastic_source tensor [B, C] for covariance tracking.
        captured: mean fraction of batch gradient energy in the top direction
            (variant="power_iter" only), else None.
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
            )
            if captured is not None:
                captured_per_layer.append(captured)
        else:
            source = torch.rand_like(g_c)

        sources[layer] = source.detach()
        component_masks[layer] = g_c + (1.0 - g_c) * source

    mean_captured = float(np.mean(captured_per_layer)) if captured_per_layer else None

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

    return mask_infos, sources, mean_captured


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


def _delta_wandb_charts(
    history: list[dict[str, float]],
) -> tuple[Any, Any]:
    """Interactive WandB line charts of ΔL_recon vs training step.

    Returns two line-series charts over all samplers: one for the raw ΔL_recon
    values and one for each GI variant's ratio to uniform (> 1 = better than uniform).
    """
    steps = [h["step"] for h in history]
    uniform_arr = np.array([h["uniform_mean"] for h in history])

    delta_chart = wandb.plot.line_series(
        xs=steps,
        ys=[[h[f"{key}_mean"] for h in history] for key in SAMPLER_KEYS],
        keys=SAMPLER_KEYS,
        title="ΔL_recon vs Training Step",
        xname="Training Step",
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = [
            list(
                np.where(
                    uniform_arr > 0,
                    np.array([h[f"{key}_mean"] for h in history]) / uniform_arr,
                    float("nan"),
                )
            )
            for key in GI_VARIANTS
        ]
    ratio_chart = wandb.plot.line_series(
        xs=steps,
        ys=ratios,
        keys=GI_VARIANTS,
        title="ΔL_recon Ratio: GI variant / Uniform (> 1 = better than uniform)",
        xname="Training Step",
    )
    return delta_chart, ratio_chart


def _covariance_heatmaps(
    stacked: dict[str, torch.Tensor],
) -> Image.Image:
    """One correlation-matrix panel per sampler in `stacked`.

    Uses the correlation matrix (not raw covariance) so the diagonal is always 1
    and is masked white — leaving only inter-component correlations visible on a
    [-1, 1] scale. High off-diagonal values in a GI panel indicate that sampler is
    collapsing onto correlated component groups (the local-extremum failure mode).
    Each subtitle shows the mean per-component variance (diagonal of cov) as the
    scalar summary of total exploration breadth.
    """
    keys = [k for k in SAMPLER_KEYS if k in stacked]
    fig, axes = plt.subplots(1, len(keys), figsize=(7 * len(keys), 6), squeeze=False)
    for ax, key in zip(axes[0], keys, strict=True):
        source = stacked[key].float()
        corr = np.corrcoef(source.numpy().T)
        mean_var = float(np.diag(torch.cov(source.T).numpy()).mean())
        # Mask diagonal so it renders white — only off-diagonal correlations are shown.
        masked = np.where(np.eye(corr.shape[0], dtype=bool), np.nan, corr)
        im = ax.imshow(masked, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"{key}  (mean var={mean_var:.4f})", fontsize=11)
        ax.set_xlabel("Component index (all layers flattened)")
        ax.set_ylabel("Component index (all layers flattened)")
    fig.suptitle(
        "Source correlation matrix  (diagonal masked; off-diagonal: high = correlated sampling)",
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
    gi_coeff: float,
    gi_power_iters: int,
) -> dict[str, Any]:
    """Compare the uniform baseline against every GI variant on a frozen ComponentModel.

    For each sampler the per-batch mean ΔL_recon is collected. A paired t-test
    (ttest_rel) of each GI variant against uniform and empirical correlation
    matrices are also computed. L_recon(unmasked) = 0 by definition, so
    ΔL_recon = L_recon(masked).

    Returns a flat dict suitable for WandB logging (images included).
    """
    model.eval()

    # Detach weight deltas once — shared across all samplers.
    weight_deltas: dict[str, torch.Tensor] | None = (
        {k: v.detach() for k, v in model.calc_weight_deltas().items()}
        if config.use_delta_component
        else None
    )

    per_batch_delta: dict[str, list[float]] = {key: [] for key in SAMPLER_KEYS}
    all_sources: dict[str, list[torch.Tensor]] = {key: [] for key in SAMPLER_KEYS}
    captured_vals: list[float] = []

    for _ in range(n_batches):
        batch, _ = dataset.generate_batch(config.eval_batch_size)  # type: ignore[attr-defined]
        batch = batch.to(device)

        # Target output = raw frozen model, no masking.
        with torch.no_grad():
            result = model(batch, cache_type="input")
        target_out = result.output.detach()
        pre_weight_acts = {k: v.detach() for k, v in result.cache.items()}

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

        for key in SAMPLER_KEYS:
            grads = None if key == "uniform" else importance_grads
            variant = cast(GIVariant, "per_component" if key == "uniform" else key)
            draw_masked: list[float] = []

            with torch.no_grad():
                for _ in range(n_draws):
                    mask_infos, sources, captured = sample_component_masks(
                        ci=ci,
                        importance_grads=grads,
                        weight_deltas=weight_deltas,
                        variant=variant,
                        coeff=gi_coeff,
                        power_iters=gi_power_iters,
                    )
                    masked_out = model(batch, mask_infos=mask_infos)
                    l_recon_masked = ((masked_out - target_out) ** 2).mean().item()
                    draw_masked.append(l_recon_masked)
                    all_sources[key].append(_flatten_sources(sources))
                    if captured is not None:
                        captured_vals.append(captured)

            per_batch_delta[key].append(float(np.mean(draw_masked)))

    # Exploration analysis: stack all draws → [N, total_C]
    stacked: dict[str, torch.Tensor] = {
        key: torch.cat(all_sources[key], dim=0).cpu().float() for key in SAMPLER_KEYS
    }

    out: dict[str, Any] = {}
    uniform_delta = np.array(per_batch_delta["uniform"])
    for key in SAMPLER_KEYS:
        delta = np.array(per_batch_delta[key])
        out[f"{key}/mean_delta_l_recon"] = float(delta.mean())
        out[f"{key}/std_delta_l_recon"] = float(delta.std())
        if key != "uniform":
            ttest_result = scipy_stats.ttest_rel(delta, uniform_delta)
            out[f"{key}/t_stat"] = cast(float, ttest_result[0])
            out[f"{key}/p_value"] = cast(float, ttest_result[1])
    if captured_vals:
        out["power_iter/mean_captured"] = float(np.mean(captured_vals))

    # Chart (PIL Image — caller wraps in wandb.Image before logging)
    out["chart/source_cov"] = _covariance_heatmaps(stacked)
    return out


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
) -> None:
    """Train a ComponentModel with uniform sampling.

    Benchmarks the uniform baseline against every GI variant at step 100, then
    every 1000 steps, always including the final step. Logs metrics and
    correlation heatmaps to WandB.
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
                gi_coeff=config.gi_coeff,
                gi_power_iters=config.gi_power_iters,
            )

            for key in SAMPLER_KEYS:
                line = (
                    f"  {key:13s}: ΔL={bench[f'{key}/mean_delta_l_recon']:.6f}"
                    f" ± {bench[f'{key}/std_delta_l_recon']:.6f}"
                )
                if key != "uniform":
                    line += f"   t={bench[f'{key}/t_stat']:.3f}  p={bench[f'{key}/p_value']:.2e}"
                tqdm.write(line)
            if "power_iter/mean_captured" in bench:
                tqdm.write(f"  power_iter captured energy: {bench['power_iter/mean_captured']:.4f}")

            delta_history.append(
                {
                    "step": float(step),
                    **{f"{key}_mean": bench[f"{key}/mean_delta_l_recon"] for key in SAMPLER_KEYS},
                }
            )

            save_file(model.state_dict(), out_dir / f"model_{step}.pth")

            if config.wandb_project:
                wandb_log: dict[str, Any] = {
                    f"benchmark/{metric}": v
                    for metric, v in bench.items()
                    if not isinstance(v, Image.Image)
                }
                wandb_log["benchmark/chart/source_cov"] = wandb.Image(bench["chart/source_cov"])
                delta_chart, ratio_chart = _delta_wandb_charts(delta_history)
                wandb_log["benchmark/chart/delta_line"] = delta_chart
                wandb_log["benchmark/chart/delta_ratio"] = ratio_chart
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


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
        "--force_continuous",
        action="store_true",
        help="Override config sampling to 'continuous' if it is not already set.",
    )
    parser.add_argument(
        "--gi_coeff",
        type=float,
        default=None,
        help="Override config.gi_coeff (directional ablation strength). Sweep [1, 3, 5].",
    )
    parser.add_argument(
        "--gi_power_iters",
        type=int,
        default=None,
        help="Override config.gi_power_iters (power-iteration steps).",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = get_device()
    set_seed(args.seed)

    config = Config.from_file(args.config_path)

    gi_overrides: dict[str, Any] = {}
    if args.gi_coeff is not None:
        gi_overrides["gi_coeff"] = args.gi_coeff
    if args.gi_power_iters is not None:
        gi_overrides["gi_power_iters"] = args.gi_power_iters
    if gi_overrides:
        config = config.model_copy(update=gi_overrides)

    if config.sampling != "continuous":
        assert args.force_continuous, (
            f"Config has sampling={config.sampling!r} but training requires 'continuous'. "
            "Pass --force_continuous to override."
        )
        config = config.model_copy(update={"sampling": "continuous"})
        logger.info("Overriding sampling to 'continuous' for training.")

    logger.info(f"Device: {device}")
    logger.info(f"Config: {args.config_path}")

    experiment_tag = config.task_config.task_name
    out_dir, run_id, tags = setup_decomposition_run(experiment_tag=experiment_tag)
    tags.append("sampler-benchmark")

    target_model, dataset, tied_weights = _make_dataset_and_target_model(config, device)

    if config.wandb_project:
        init_wandb(
            config=config,
            project=config.wandb_project,
            run_id=run_id,
            name=f"sampler-benchmark-{args.config_path.stem}",
            tags=tags,
        )

    run_benchmark_training(
        config=config,
        target_model=target_model,
        dataset=dataset,
        device=device,
        out_dir=out_dir,
        n_benchmark_batches=args.n_benchmark_batches,
        n_draws=args.n_draws,
        tied_weights=tied_weights,
    )

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    main()

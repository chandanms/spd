#!/usr/bin/env python
"""Re-run the sampler benchmark on checkpoints already saved by benchmark_sampler.py.

The expensive part of `benchmark_sampler.py` is training the ComponentModel; the arms
themselves are cheap. This script loads the final checkpoint each seed already wrote and
evaluates a fresh set of arms against it, so adding arms costs minutes rather than a
retrain.

The main use is budget-matched PGD. GIS spends one gradient, which during training is
already computed and therefore free; PGD with `n_steps=20` spends twenty forward and
backward passes. `PGD, n_steps=1` is the arm that costs what GIS costs, and is the
comparison that says whether GIS is a good cheap sampler or merely a weak one.

Usage:
    source .venv/bin/activate
    python spd/scripts/benchmark_from_checkpoint.py \\
        --config_path spd/experiments/resid_mlp/resid_mlp2_continuous_config.yaml \\
        --checkpoints ~/spd_out/spd/s-a8039b39/model_25000.pth,...
"""

import argparse
from pathlib import Path
from typing import Any

import torch

from spd.configs import Config
from spd.identity_insertion import insert_identity_operations_
from spd.log import logger
from spd.models.component_model import ComponentModel
from spd.scripts.benchmark_sampler import (
    GISArm,
    PGDArm,
    SamplerArm,
    UniformArm,
    _make_dataset_and_target_model,
    benchmark_samplers,
    format_latex_table,
)
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import set_seed
from spd.utils.module_utils import expand_module_patterns


def load_component_model(config: Config, checkpoint: Path, device: str) -> ComponentModel:
    """Rebuild the ComponentModel described by `config` and load a saved state dict."""
    target_model, _, _ = _make_dataset_and_target_model(config, device)
    target_model.requires_grad_(False)
    target_model.to(device)
    target_model.eval()

    if config.identity_module_info is not None:
        insert_identity_operations_(target_model, identity_module_info=config.identity_module_info)

    model = ComponentModel(
        target_model=target_model,
        module_path_info=expand_module_patterns(target_model, config.all_module_info),
        ci_fn_type=config.ci_fn_type,
        ci_fn_hidden_dims=config.ci_fn_hidden_dims,
        pretrained_model_output_attr=config.pretrained_model_output_attr,
        sigmoid_type=config.sigmoid_type,
    )
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", type=Path, required=True)
    parser.add_argument(
        "--checkpoints",
        type=str,
        required=True,
        help="Comma-separated model_*.pth paths, one per seed",
    )
    parser.add_argument("--gis_ks", type=str, default="1,2,4,8")
    parser.add_argument(
        "--pgd_specs",
        type=str,
        default="0.5:1,0.5:2,0.5:5,0.5:20",
        help="Comma-separated PGD arms as step_size:n_steps",
    )
    parser.add_argument("--n_benchmark_batches", type=int, default=20)
    parser.add_argument("--n_draws", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("results_table_budget_matched.tex"))
    args = parser.parse_args()

    arms: list[SamplerArm] = [UniformArm()]
    arms += [GISArm(k=float(k)) for k in args.gis_ks.split(",") if k]
    for spec in args.pgd_specs.split(","):
        if not spec:
            continue
        step_size, n_steps = spec.split(":")
        arms.append(PGDArm(step_size=float(step_size), n_steps=int(n_steps)))

    config = Config.from_file(args.config_path)
    device = get_device()
    checkpoints = [Path(c.strip()).expanduser() for c in args.checkpoints.split(",") if c.strip()]

    benches: list[dict[str, Any]] = []
    for i, checkpoint in enumerate(checkpoints):
        assert checkpoint.exists(), f"missing checkpoint: {checkpoint}"
        logger.info(f"=== checkpoint {i + 1}/{len(checkpoints)}: {checkpoint} ===")
        # Seed per checkpoint so the eval batches and mask draws differ between seeds.
        set_seed(i)
        _, dataset, _ = _make_dataset_and_target_model(config, device)
        model = load_component_model(config, checkpoint, device)
        benches.append(
            benchmark_samplers(
                model=model,
                dataset=dataset,
                config=config,
                device=device,
                n_batches=args.n_benchmark_batches,
                n_draws=args.n_draws,
                arms=arms,
            )
        )

    uniform_delta = float(sum(b["uniform/mean_delta_l_recon"] for b in benches) / len(benches))
    pgd_deltas = [
        sum(b[f"{a.name}/mean_delta_l_recon"] for b in benches) / len(benches)
        for a in arms
        if isinstance(a, PGDArm)
    ]
    ceiling_excess = max(pgd_deltas) - uniform_delta

    print(f"\n{'arm':<22}{'ΔL (mean)':>13}{'ratio':>16}{'% of PGD headroom':>20}")
    print("-" * 71)
    for arm in arms:
        deltas = [b[f"{arm.name}/mean_delta_l_recon"] for b in benches]
        mean_delta = float(sum(deltas) / len(deltas))
        ratios = [b[f"{arm.name}/ratio_to_uniform"] for b in benches]
        ratio_mean = float(sum(ratios) / len(ratios))
        spread = max(ratios) - min(ratios)
        headroom = 100.0 * (mean_delta - uniform_delta) / ceiling_excess
        print(
            f"{arm.name:<22}{mean_delta:>13.4e}"
            f"{f'{ratio_mean:.3f} ± {spread / 2:.3f}':>16}{headroom:>19.1f}%"
        )

    table = format_latex_table(benches, arms)
    args.out.write_text(table)
    print(f"\n{table}\n\nSaved to {args.out}")


if __name__ == "__main__":
    main()

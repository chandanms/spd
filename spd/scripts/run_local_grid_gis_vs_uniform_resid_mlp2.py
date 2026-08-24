#!/usr/bin/env python
"""GIS-vs-uniform sampler grid for resid_mlp2, judged by a fresh-PGD adversary.

Two training arms differing in exactly one field, `sampling`:

    continuous          uniform stochastic masks
    gradient_informed   GIS masks (spd/utils/component_utils.py)

crossed with two training-adversary budgets:

    pgd_n_steps = 0     PGDReconSubsetLoss dropped entirely
    pgd_n_steps = 2     the repo's current setting, unchanged

The 0-step arm matters because a strong training adversary can paper over any
difference between samplers; with no adversary the sampler is the only thing
shaping the decomposition, which is where a sampler effect should be largest.

The measurement is the eval `PGDReconLoss` (fresh PGD, random init, step_size 0.1,
n_steps 20), held identical across all arms. It logs to WandB as
`eval/loss/PGDReconLoss`. CI-L0 logs alongside as `eval/l0/0.1_<layer>`: a lower
adversarial loss bought by inflating CI is not robustness, and GIS has a specific
incentive to do that, since raising ci shrinks the (1 - ci) factor that is the
adversary's only leverage over the mask.

Usage:
    source .venv/bin/activate
    python spd/scripts/run_local_grid_gis_vs_uniform_resid_mlp2.py
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import fire

from spd.configs import Config, PGDReconSubsetLossConfig
from spd.settings import REPO_ROOT

BASE_CONFIG = (
    Path(__file__).parent.parent / "experiments" / "resid_mlp" / "resid_mlp2_gis_grid_config.yaml"
)
DECOMP_SCRIPT = REPO_ROOT / "spd" / "experiments" / "resid_mlp" / "resid_mlp_decomposition.py"

SAMPLERS = ["continuous", "gradient_informed"]
PGD_N_STEPS = [0, 2]
SEEDS = [0, 1, 2]


def build_arm_config(sampling: str, pgd_n_steps: int, seed: int, run_name: str) -> Config:
    """Derive one arm from the base config, changing only sampler, adversary budget and seed."""
    config = Config.from_file(BASE_CONFIG)

    loss_configs = list(config.loss_metric_configs)
    if pgd_n_steps == 0:
        loss_configs = [c for c in loss_configs if not isinstance(c, PGDReconSubsetLossConfig)]
    else:
        loss_configs = [
            c.model_copy(update={"n_steps": pgd_n_steps})
            if isinstance(c, PGDReconSubsetLossConfig)
            else c
            for c in loss_configs
        ]

    n_pgd_losses = sum(isinstance(c, PGDReconSubsetLossConfig) for c in loss_configs)
    assert n_pgd_losses == (0 if pgd_n_steps == 0 else 1), (
        f"expected {0 if pgd_n_steps == 0 else 1} PGD training losses, got {n_pgd_losses}"
    )

    return config.model_copy(
        update={
            "sampling": sampling,
            "seed": seed,
            "wandb_run_name": run_name,
            "loss_metric_configs": loss_configs,
        }
    )


def main(
    samplers: str | None = None,
    pgd_n_steps: str | None = None,
    seeds: str | None = None,
) -> None:
    """Run the grid, one subprocess per arm. A `None` filter means "the whole axis".

    Each arm runs in its own process so accumulated GPU state cannot carry between
    runs, and so a failure part-way through a multi-hour grid is recoverable: re-run
    with the filters narrowed to whatever is missing, e.g.

        python spd/scripts/run_local_grid_gis_vs_uniform_resid_mlp2.py \\
            --samplers gradient_informed --pgd_n_steps 2 --seeds 1,2
    """
    selected_samplers = SAMPLERS if samplers is None else [s.strip() for s in samplers.split(",")]
    selected_budgets = (
        PGD_N_STEPS if pgd_n_steps is None else [int(n) for n in str(pgd_n_steps).split(",")]
    )
    selected_seeds = SEEDS if seeds is None else [int(s) for s in str(seeds).split(",")]

    assert all(s in SAMPLERS for s in selected_samplers), f"unknown sampler in {selected_samplers}"

    arms = [
        (budget, sampling, seed)
        for budget in selected_budgets
        for sampling in selected_samplers
        for seed in selected_seeds
    ]
    print(f"Running {len(arms)} arms (~40 min each)")

    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, (budget, sampling, seed) in enumerate(arms, start=1):
            run_name = f"gis-grid_resid_mlp2_{sampling}_pgd-{budget}_seed-{seed}"
            print("=" * 70)
            print(f"[{i}/{len(arms)}] {run_name}")
            print("=" * 70)

            config = build_arm_config(
                sampling=sampling, pgd_n_steps=budget, seed=seed, run_name=run_name
            )
            # JSON is valid YAML, so Config.from_file reads this back unchanged.
            config_path = Path(tmp_dir) / f"{run_name}.yaml"
            config_path.write_text(config.model_dump_json(indent=2))

            result = subprocess.run(
                [sys.executable, str(DECOMP_SCRIPT), str(config_path)], check=False
            )
            assert result.returncode == 0, (
                f"arm {run_name} exited {result.returncode}; "
                f"re-run the remaining arms with --samplers/--pgd_n_steps/--seeds"
            )


if __name__ == "__main__":
    fire.Fire(main)

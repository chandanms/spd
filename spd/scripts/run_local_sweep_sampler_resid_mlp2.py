#!/usr/bin/env python
"""Local sweep comparing uniform vs gradient_informed (per_component) sampling."""

from pathlib import Path

from spd.scripts.benchmark_sampler import run_sampler_benchmark

RESID_MLP_DIR = Path(__file__).parent.parent / "experiments" / "resid_mlp"
TMS_DIR = Path(__file__).parent.parent / "experiments" / "tms"

# (condition_name, config_path, force_continuous)
conditions = [
    ("resid_mlp2", RESID_MLP_DIR / "resid_mlp2_continuous_config.yaml", False),
    ("tms_40-10", TMS_DIR / "tms_40-10_config.yaml", False),
    ("tms_40-10-id", TMS_DIR / "tms_40-10-id_config.yaml", True),
]
seeds = [1, 2, 3, 4, 5, 6, 7, 8]

for condition_name, config_path, force_continuous in conditions:
    for seed in seeds:
        run_name = f"sampler-benchmark-{condition_name}_seed-{seed}"
        print("========================================")
        print(f"Running: {run_name}")
        print("========================================")

        run_sampler_benchmark(
            config_path=config_path,
            seed=seed,
            run_name=run_name,
            force_continuous=force_continuous,
        )

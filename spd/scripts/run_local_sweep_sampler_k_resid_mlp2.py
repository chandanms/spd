#!/usr/bin/env python
"""Local sweep over the gradient-informed temperature k for resid_mlp2.

For each k the gradient-informed weights are w_c = |grad_c|^k / sum_c |grad_c|^k.
k=1 is the current baseline (mass spread ~1/C, sampler stays near uniform); larger
k concentrates weight on the top-attribution components. Sweeping k lets us see how
the GI-vs-uniform ΔL_recon gap and its variance change as the distribution sharpens.
"""

from pathlib import Path

from spd.scripts.benchmark_sampler import run_sampler_benchmark

RESID_MLP_DIR = Path(__file__).parent.parent / "experiments" / "resid_mlp"

config_path = RESID_MLP_DIR / "resid_mlp2_continuous_config.yaml"
force_continuous = False

ks = [1.0, 2.0, 4.0, 8.0, 16.0]
seeds = [1, 2, 3]

for k in ks:
    for seed in seeds:
        run_name = f"sampler-benchmark-k-resid_mlp2_k-{k}_seed-{seed}"
        print("========================================")
        print(f"Running: {run_name}")
        print("========================================")

        run_sampler_benchmark(
            config_path=config_path,
            seed=seed,
            run_name=run_name,
            force_continuous=force_continuous,
            importance_temperature=k,
        )

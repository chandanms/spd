#!/usr/bin/env python
"""Delete all WandB runs created by run_local_sweep_pgd.py."""

import wandb

ENTITY = "goodfire"
PROJECT = "spd-gradient-informed-sampling"

experiments = ["tms_5-2", "tms_5-2-id", "tms_40-10", "tms_40-10-id"]
seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

run_names = {f"{exp}_pgd_seed-{seed}" for exp in experiments for seed in seeds}

api = wandb.Api()
runs = api.runs(f"{ENTITY}/{PROJECT}")

deleted = 0
not_found = list(run_names)

for run in runs:
    if run.name in run_names:
        print(f"Deleting: {run.name} (id={run.id})")
        run.delete()
        not_found.remove(run.name)
        deleted += 1

print(f"\nDeleted {deleted} runs.")
if not_found:
    print(f"Not found: {not_found}")

#!/usr/bin/env python
"""Local sweep script for TMS PGD experiments."""

import json

from spd.configs import Config
from spd.experiments.tms.tms_decomposition import main

experiments = ["tms_5-2", "tms_5-2-id", "tms_40-10", "tms_40-10-id"]
seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for exp in experiments:
    config_path = f"/home/chandan/spd/spd/experiments/tms/{exp}_pgd_config.yaml"

    for seed in seeds:
        run_name = f"{exp}_pgd_revised_seed-{seed}"
        print(f"========================================")
        print(f"Running: {run_name}")
        print(f"========================================")

        config = Config.from_file(config_path)
        config_dict = config.model_dump(mode="json")
        config_dict["seed"] = seed
        config_dict["wandb_run_name"] = run_name

        config_json = "json:" + json.dumps(config_dict)
        main(config_json=config_json)

#!/usr/bin/env python
"""Local sweep comparing PGD vs gradient_informed vs stochastic_only on resid_mlp2."""

import json
from pathlib import Path

from spd.configs import Config
from spd.experiments.resid_mlp.resid_mlp_decomposition import main

CONFIG_DIR = Path(__file__).parent.parent / "experiments" / "resid_mlp"

conditions = [
    ("subset_gradient_informed_2.0", CONFIG_DIR / "resid_mlp2_gradient_informed_config.yaml"),
]
seeds = [1, 2, 3, 4, 5, 6, 7, 8]

for condition_name, config_path in conditions:
    for seed in seeds:
        run_name = f"resid_mlp2_{condition_name}_seed-{seed}"
        print(f"========================================")
        print(f"Running: {run_name}")
        print(f"========================================")

        config = Config.from_file(config_path)
        config_dict = config.model_dump(mode="json")
        config_dict["seed"] = seed
        config_dict["wandb_run_name"] = run_name

        config_json = "json:" + json.dumps(config_dict)
        main(config_json=config_json)

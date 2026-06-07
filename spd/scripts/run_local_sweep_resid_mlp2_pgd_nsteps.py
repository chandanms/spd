#!/usr/bin/env python
"""Local sweep over PGDReconSubsetLoss n_steps (coeff=2.0) on resid_mlp2."""

import json
from pathlib import Path

from spd.configs import Config
from spd.experiments.resid_mlp.resid_mlp_decomposition import main

CONFIG_DIR = Path(__file__).parent.parent / "experiments" / "resid_mlp"

seeds = [1, 2, 3, 4, 5, 6, 7, 8]
n_steps_values = [3, 4, 5]

for n_steps in n_steps_values:
    for seed in seeds:
        run_name = f"resid_mlp2_pgd_nsteps_{float(n_steps)}_seed-{seed}"
        print(f"========================================")
        print(f"Running: {run_name}")
        print(f"========================================")

        config = Config.from_file(CONFIG_DIR / "resid_mlp2_config.yaml")
        config_dict = config.model_dump(mode="json")
        config_dict["seed"] = seed
        config_dict["wandb_project"] = "spd-gradient-informed-sampling"
        config_dict["wandb_run_name"] = run_name

        for loss_cfg in config_dict["loss_metric_configs"]:
            if loss_cfg["classname"] == "PGDReconSubsetLoss":
                loss_cfg["coeff"] = 2.0
                loss_cfg["n_steps"] = n_steps
                break

        config_json = "json:" + json.dumps(config_dict)
        main(config_json=config_json)

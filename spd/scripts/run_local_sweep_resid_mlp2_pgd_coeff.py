#!/usr/bin/env python
"""Local sweep over PGDReconSubsetLoss coeff on resid_mlp2, and gradient_informed with StochasticReconSubsetLoss coeff=5.0."""

import json
from pathlib import Path

from spd.configs import Config
from spd.experiments.resid_mlp.resid_mlp_decomposition import main

CONFIG_DIR = Path(__file__).parent.parent / "experiments" / "resid_mlp"

seeds = [1, 2, 3, 4, 5, 6, 7, 8]

# resid_mlp2_config.yaml sweeping PGDReconSubsetLoss coeff
pgd_coeffs = [3.0, 4.0, 5.0]
for coeff in pgd_coeffs:
    for seed in seeds:
        run_name = f"resid_mlp2_pgd_{coeff}_seed-{seed}"
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
                loss_cfg["coeff"] = coeff
                break

        config_json = "json:" + json.dumps(config_dict)
        main(config_json=config_json)

# resid_mlp2_gradient_informed_config.yaml with StochasticReconSubsetLoss coeff=5.0
gradient_informed_coeff = 5.0
for seed in seeds:
    run_name = f"resid_mlp2_gradient_informed_{gradient_informed_coeff}_seed-{seed}"
    print(f"========================================")
    print(f"Running: {run_name}")
    print(f"========================================")

    config = Config.from_file(CONFIG_DIR / "resid_mlp2_gradient_informed_config.yaml")
    config_dict = config.model_dump(mode="json")
    config_dict["seed"] = seed
    config_dict["wandb_run_name"] = run_name

    for loss_cfg in config_dict["loss_metric_configs"]:
        if loss_cfg["classname"] == "StochasticReconSubsetLoss":
            loss_cfg["coeff"] = gradient_informed_coeff
            break

    config_json = "json:" + json.dumps(config_dict)
    main(config_json=config_json)

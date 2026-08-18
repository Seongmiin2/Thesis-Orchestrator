from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import load_config
from src.experiments.reinartz_f0_f1 import run_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    artifacts = Path(config["paths"]["artifacts"])
    split = pd.read_csv(artifacts / "reinartz_split_manifest.csv")
    scaler = pd.read_csv(artifacts / "reinartz_scaler_parameters.csv")
    features = np.load(Path(config["paths"]["cache"]) / "features.npy", mmap_mode="r")
    mean = scaler["mean"].to_numpy(np.float32)
    std = scaler["std"].to_numpy(np.float32)
    log_dir = Path(config["paths"]["logs"])
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for seed in args.seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        metrics, faults = run_model(
            config, split, features, mean, std, "F0-C", int(config["epochs"]), seed, hidden_dim=68
        )
        pd.DataFrame([metrics]).to_csv(artifacts / f"reinartz_f0_c_results_seed_{seed}.csv", index=False)
        pd.DataFrame(faults).to_csv(artifacts / f"reinartz_f0_c_fault_results_seed_{seed}.csv", index=False)
        logging.info("F0-C seed %d complete: %s", seed, json.dumps(metrics))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
PHYSICAL = ROOT.parent / "PhysicalAI_mini"
sys.path.insert(0, str(PHYSICAL))

from src.experiments import reinartz_f0_f1 as experiment  # noqa: E402


class TCNForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int, output_dim: int = 41):
        super().__init__()
        blocks = []
        width = input_dim
        for i in range(layers):
            dilation = 2**i
            blocks.extend([nn.Conv1d(width, hidden_dim, 3, padding=dilation, dilation=dilation), nn.GELU()])
            width = hidden_dim
        self.network = nn.Sequential(*blocks); self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, sequence):
        hidden = self.network(sequence.transpose(1, 2))[:, :, -1]
        return self.output(hidden)


class TransformerForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int, output_dim: int = 41):
        super().__init__()
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.position = nn.Parameter(torch.zeros(1, 20, hidden_dim))
        heads = 4 if hidden_dim % 4 == 0 else 1
        layer = nn.TransformerEncoderLayer(hidden_dim, heads, hidden_dim * 2, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(hidden_dim); self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, sequence):
        hidden = self.encoder(self.projection(sequence) + self.position[:, : sequence.shape[1]])
        return self.output(self.norm(hidden[:, -1]))


ARCH = "tcn"


def factory(input_dim: int, hidden_dim: int, layers: int):
    cls = TCNForecaster if ARCH == "tcn" else TransformerForecaster
    return cls(input_dim, hidden_dim, layers)


def count(input_dim: int, width: int, layers: int) -> int:
    return sum(p.numel() for p in factory(input_dim, width, layers).parameters())


def matched_width(target: int, layers: int) -> int:
    return min(range(8, 129), key=lambda width: abs(count(41, width, layers) - target))


def main() -> None:
    global ARCH
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/architecture_gate_g2.yaml")
    parser.add_argument("--architectures", nargs="+", choices=["tcn", "transformer"], default=["tcn", "transformer"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    import yaml
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8")); epochs = args.epochs or int(cfg["epochs"])
    split = pd.read_csv(ROOT / "outputs/final_gate_exp1/artifacts/reinartz_split_manifest.csv")
    scaler = pd.read_csv(ROOT / "outputs/final_gate_exp1/artifacts/reinartz_scaler_parameters.csv")
    mean, std = scaler["mean"].to_numpy(np.float32), scaler["std"].to_numpy(np.float32)
    features = np.load(Path(cfg["paths"]["cache"]) / "features.npy", mmap_mode="r")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    experiment.GRUForecaster = factory
    all_metrics, all_faults = [], []
    for architecture in args.architectures:
        ARCH = architecture
        width = int(cfg["hidden_dim"]); layers = int(cfg["layers"])
        target = count(52, width, layers); f0c_width = matched_width(target, layers)
        logging.info("architecture=%s F1 width=%d params=%d F0-C width=%d params=%d", architecture, width, target, f0c_width, count(41, f0c_width, layers))
        arch_cfg = dict(cfg); arch_cfg["paths"] = dict(cfg["paths"])
        arch_cfg["paths"]["artifacts"] = str(Path(cfg["paths"]["artifacts"]) / architecture)
        arch_cfg["paths"]["checkpoints"] = str(Path(cfg["paths"]["checkpoints"]) / architecture)
        Path(arch_cfg["paths"]["artifacts"]).mkdir(parents=True, exist_ok=True)
        for seed in args.seeds:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.use_deterministic_algorithms(True)
            for variant, variant_width in (("F0", width), ("F1", width), ("F0-C", f0c_width)):
                metrics, faults = experiment.run_model(arch_cfg, split, features, mean, std, variant, epochs, seed, hidden_dim=variant_width)
                metrics["architecture"] = architecture
                for row in faults: row["architecture"] = architecture
                all_metrics.append(metrics); all_faults.extend(faults)
                pd.DataFrame(all_metrics).to_csv(Path(cfg["paths"]["artifacts"]) / "G2_METRICS.partial.csv", index=False)
                pd.DataFrame(all_faults).to_csv(Path(cfg["paths"]["artifacts"]) / "G2_FAULT_RESULTS.partial.csv", index=False)
    pd.DataFrame(all_metrics).to_csv(Path(cfg["paths"]["artifacts"]) / "G2_METRICS.csv", index=False)
    pd.DataFrame(all_faults).to_csv(Path(cfg["paths"]["artifacts"]) / "G2_FAULT_RESULTS.csv", index=False)
    (Path(cfg["paths"]["artifacts"]) / "RUN_MANIFEST.json").write_text(json.dumps({"status":"COMPLETE", "architectures":args.architectures, "seeds":args.seeds, "epochs":epochs}, indent=2), encoding="utf-8")


if __name__ == "__main__": main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FAULTS = (4, 7, 19, 24, 25, 26)
WINDOW = 20
BANK_ENDPOINTS = (300, 450, 580)
NORMAL_ENDPOINTS = (520, 560, 590)
TEST_ENDPOINTS = (*NORMAL_ENDPOINTS, *range(600, 701, 10))


def representation(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Observable past-only level/slope summaries; no labels or future XMV."""
    mean = z.mean(axis=1)
    last = z[:, -1]
    slope = z[:, -1] - z[:, 0]
    return np.concatenate([mean[:, :41], last[:, :41], slope[:, :41]], axis=1), np.concatenate(
        [mean[:, 41:], last[:, 41:], slope[:, 41:]], axis=1
    )


def standardize(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std < 1e-6] = 1.0
    return tuple((x - mean) / std for x in (train, *others))


def make_rows(features: np.ndarray, run_indices: np.ndarray, endpoints: tuple[int, ...]):
    windows, targets, meta = [], [], []
    for run in run_indices:
        for endpoint in endpoints:
            windows.append(np.asarray(features[run, endpoint - WINDOW : endpoint], dtype=np.float32))
            targets.append(np.asarray(features[run, endpoint, :41], dtype=np.float32))
            meta.append((int(run), int(endpoint)))
    x, u = representation(np.asarray(windows))
    return x.astype(np.float32), u.astype(np.float32), np.asarray(targets), np.asarray(meta)


def distances(query: np.ndarray, bank: np.ndarray) -> np.ndarray:
    q2 = np.sum(query * query, axis=1, keepdims=True)
    b2 = np.sum(bank * bank, axis=1)[None, :]
    return np.maximum(q2 + b2 - 2.0 * query @ bank.T, 0.0)


def predict_methods(
    qx: np.ndarray,
    qu: np.ndarray,
    bank_x: np.ndarray,
    bank_u: np.ndarray,
    bank_y: np.ndarray,
    threshold: float,
    top_k: int = 5,
    shortlist: int = 50,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    predictions = {name: [] for name in ("sensor", "joint", "oracle_proxy")}
    discordant = applicable = total_short = 0
    batch = 64
    for start in range(0, len(qx), batch):
        end = min(start + batch, len(qx))
        dx = distances(qx[start:end], bank_x)
        du = distances(qu[start:end], bank_u)
        joint = dx / bank_x.shape[1] + du / bank_u.shape[1]
        for row in range(end - start):
            sensor_order = np.argpartition(dx[row], shortlist)[:shortlist]
            sensor_order = sensor_order[np.argsort(dx[row, sensor_order])]
            joint_order = np.argpartition(joint[row], top_k)[:top_k]
            joint_order = joint_order[np.argsort(joint[row, joint_order])]
            valid = sensor_order[du[row, sensor_order] <= threshold]
            oracle_order = valid[:top_k] if len(valid) >= top_k else sensor_order[:top_k]
            predictions["sensor"].append(bank_y[sensor_order[:top_k]].mean(axis=0))
            predictions["joint"].append(bank_y[joint_order].mean(axis=0))
            predictions["oracle_proxy"].append(bank_y[oracle_order].mean(axis=0))
            flags = du[row, sensor_order] <= threshold
            applicable += int(flags.sum())
            total_short += len(flags)
            discordant += int((~flags[:top_k]).sum())
    stats = {
        "shortlist_applicable_fraction": applicable / total_short,
        "top5_sensor_inapplicable_fraction": discordant / (len(qx) * top_k),
    }
    return {k: np.asarray(v) for k, v in predictions.items()}, stats


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    result = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        result[order[i:j]] = (i + j - 1) / 2 + 1
        i = j
    return result


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    pos, neg = labels == 1, labels == 0
    return float((ranks(scores)[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores)
    y = labels[order]
    precision = np.cumsum(y) / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


def evaluate(scores: np.ndarray, meta: np.ndarray, thresholds: dict[str, float], name: str) -> dict:
    labels = (meta[:, 1] >= 600).astype(np.int8)
    threshold = thresholds[name]
    delays, detected = [], []
    for run in np.unique(meta[:, 0]):
        idx = np.flatnonzero(meta[:, 0] == run)
        post = idx[meta[idx, 1] >= 600]
        exceed = scores[post] >= threshold
        hits = np.convolve(exceed.astype(np.int8), np.ones(3, dtype=np.int8), mode="valid")
        found = np.flatnonzero(hits >= 3)
        detected.append(bool(len(found)))
        delays.append(float(meta[post[found[0]], 1] - 600) if len(found) else np.nan)
    return {
        "method": name,
        "auroc": auroc(labels, scores),
        "auprc": auprc(labels, scores),
        "prefault_fpr": float(np.mean(scores[labels == 0] >= threshold)),
        "detected_run_ratio": float(np.mean(detected)),
        "median_detection_delay_detected": float(np.nanmedian(delays)) if np.any(np.isfinite(delays)) else None,
        "threshold": threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-root", type=Path, default=Path("../PhysicalAI_mini"))
    parser.add_argument("--output", type=Path, default=Path("outputs/methodology/g0"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = args.physical_root.resolve()
    features = np.load(root / "data/processed/reinartz_f0_f1/features.npy", mmap_mode="r")
    split = pd.read_csv(root / "artifacts/tables/reinartz_split_manifest.csv")
    train_runs = split[split["split"] == "train"].run_index.to_numpy(int)
    val_runs = split[split["split"] == "validation"].run_index.to_numpy(int)
    test = split[(split["split"] == "test") & split.fault_id.isin(FAULTS)]
    test_runs = test.run_index.to_numpy(int)

    bx, bu, by, _ = make_rows(features, train_runs, BANK_ENDPOINTS)
    vx, vu, vy, vm = make_rows(features, val_runs, NORMAL_ENDPOINTS)
    tx, tu, ty, tm = make_rows(features, test_runs, TEST_ENDPOINTS)
    bx, vx, tx = standardize(bx, vx, tx)
    bu, vu, tu = standardize(bu, vu, tu)

    rng = np.random.default_rng(42)
    pairs_a = rng.integers(0, len(bu), size=20000)
    pairs_b = rng.integers(0, len(bu), size=20000)
    control_distance_threshold = float(np.quantile(np.sum((bu[pairs_a] - bu[pairs_b]) ** 2, axis=1), 0.25))

    val_pred, val_stats = predict_methods(vx, vu, bx, bu, by, control_distance_threshold)
    test_pred, test_stats = predict_methods(tx, tu, bx, bu, by, control_distance_threshold)
    val_scores = {k: np.mean(np.abs(v - vy), axis=1) for k, v in val_pred.items()}
    test_scores = {k: np.mean(np.abs(v - ty), axis=1) for k, v in test_pred.items()}
    thresholds = {k: float(np.quantile(v, 0.99)) for k, v in val_scores.items()}
    metrics = [evaluate(test_scores[k], tm, thresholds, k) for k in test_scores]

    sensor = next(x for x in metrics if x["method"] == "sensor")
    oracle = next(x for x in metrics if x["method"] == "oracle_proxy")
    headroom = {
        "delta_auroc": oracle["auroc"] - sensor["auroc"],
        "delta_auprc": oracle["auprc"] - sensor["auprc"],
        "delta_detected_run_ratio": oracle["detected_run_ratio"] - sensor["detected_run_ratio"],
        "delay_reduction_detected": None if sensor["median_detection_delay_detected"] is None or oracle["median_detection_delay_detected"] is None else sensor["median_detection_delay_detected"] - oracle["median_detection_delay_detected"],
    }
    result = {
        "experiment_id": "G0-ORACLE-PROXY-001",
        "status": "PROXY_ONLY_NOT_GOLD_APPLICABILITY",
        "observable_variables": ["past XMEAS level/last/slope", "past XMV level/last/slope"],
        "excluded_unavailable_variables": ["operating mode", "XMV12", "verified fault mechanism", "known future control"],
        "applicability_proxy": "past-XMV summary distance below a train-derived fixed first-quartile threshold",
        "non_circularity": "No downstream error, future observation, fault label, or test result enters the proxy label.",
        "samples": {"bank": len(bx), "validation_queries": len(vx), "test_queries": len(tx), "test_runs": len(test_runs)},
        "control_distance_threshold": control_distance_threshold,
        "validation_pair_stats": val_stats,
        "test_pair_stats": test_stats,
        "metrics": metrics,
        "oracle_proxy_headroom_vs_sensor": headroom,
        "scientific_limit": "This experiment tests observable control-consistency filtering, which RAG4CTS already substantially covers. It cannot establish mechanism-level applicability or novelty.",
    }
    (args.output / "g0_oracle_proxy_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(metrics).to_csv(args.output / "g0_oracle_proxy_metrics.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

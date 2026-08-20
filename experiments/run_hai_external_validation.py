from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
LABEL_COLUMNS = ["attack", "attack_P1", "attack_P2", "attack_P3"]


class GRUForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int, output_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(sequence)
        return self.output(hidden[-1])


class WindowDataset(Dataset):
    def __init__(
        self,
        sequences: list[np.ndarray],
        input_indices: np.ndarray,
        output_indices: np.ndarray,
        window: int,
        stride: int,
    ):
        self.sequences = sequences
        self.input_indices = input_indices
        self.output_indices = output_indices
        self.window = window
        self.lookup: list[tuple[int, int]] = []
        for sequence_index, sequence in enumerate(sequences):
            self.lookup.extend(
                (sequence_index, target)
                for target in range(window, len(sequence), stride)
            )

    def __len__(self) -> int:
        return len(self.lookup)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_index, target = self.lookup[index]
        sequence = self.sequences[sequence_index]
        x = sequence[target - self.window : target, self.input_indices]
        y = sequence[target, self.output_indices]
        return torch.from_numpy(x), torch.from_numpy(y)


@dataclass
class TestEpisode:
    name: str
    features: np.ndarray
    labels: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def parameter_count(input_dim: int, hidden_dim: int, layers: int, output_dim: int) -> int:
    return sum(
        parameter.numel()
        for parameter in GRUForecaster(input_dim, hidden_dim, layers, output_dim).parameters()
    )


def matched_hidden_dim(
    target_parameters: int, input_dim: int, layers: int, output_dim: int
) -> int:
    return min(
        range(1, 257),
        key=lambda hidden: abs(
            parameter_count(input_dim, hidden, layers, output_dim) - target_parameters
        ),
    )


def load_data(
    source: Path,
    prepared: Path,
    f1_columns: list[str],
    train_fraction: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[TestEpisode], np.ndarray, np.ndarray]:
    if len(f1_columns) != len(set(f1_columns)):
        raise ValueError("F1 feature list contains duplicate columns")
    segments = pd.read_csv(prepared / "HAI_TRAIN_SEGMENTS.csv")
    train_sequences: list[np.ndarray] = []
    validation_sequences: list[np.ndarray] = []
    scaler_rows: list[np.ndarray] = []
    for file_name, file_segments in segments.groupby("file", sort=True):
        frame = pd.read_csv(source / file_name, usecols=f1_columns)[f1_columns]
        if frame.columns.tolist() != f1_columns:
            raise RuntimeError(f"Feature-order mismatch after reading {file_name}")
        for segment in file_segments.itertuples(index=False):
            values = frame.iloc[segment.start_row : segment.end_row + 1].to_numpy(
                dtype=np.float32, copy=True
            )
            split = int(len(values) * train_fraction)
            if split <= 30 or len(values) - split <= 30:
                raise ValueError(f"Segment too short after split: {file_name}")
            train_sequences.append(values[:split])
            validation_sequences.append(values[split:])
            scaler_rows.append(values[:split])

    stacked = np.concatenate(scaler_rows, axis=0).astype(np.float64)
    mean = stacked.mean(axis=0).astype(np.float32)
    std = stacked.std(axis=0).astype(np.float32)
    if np.any(std <= 1e-8):
        constant = np.asarray(f1_columns)[std <= 1e-8].tolist()
        raise ValueError(f"Active columns became constant after chronological split: {constant}")
    del stacked

    def scale(sequence: np.ndarray) -> np.ndarray:
        return ((sequence - mean) / std).astype(np.float32)

    train_sequences = [scale(sequence) for sequence in train_sequences]
    validation_sequences = [scale(sequence) for sequence in validation_sequences]
    test_episodes: list[TestEpisode] = []
    for path in sorted(source.glob("test*.csv.gz")):
        requested_columns = [*f1_columns, *LABEL_COLUMNS]
        frame = pd.read_csv(path, usecols=requested_columns)[requested_columns]
        if frame.columns.tolist() != requested_columns:
            raise RuntimeError(f"Feature-order mismatch after reading {path.name}")
        features = scale(frame[f1_columns].to_numpy(dtype=np.float32, copy=True))
        labels = frame[LABEL_COLUMNS].to_numpy(dtype=np.int8, copy=True)
        test_episodes.append(TestEpisode(path.name, features, labels))
    return train_sequences, validation_sequences, test_episodes, mean, std


def make_loader(
    sequences: list[np.ndarray],
    input_indices: np.ndarray,
    output_indices: np.ndarray,
    window: int,
    stride: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = WindowDataset(
        sequences, input_indices, output_indices, window=window, stride=stride
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def mean_loss(
    model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool
) -> float:
    total = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = torch.mean((model(x) - y) ** 2)
            total += float(loss) * len(x)
            count += len(x)
    return total / count


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    cfg: dict,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[dict], int, float]:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    bad_epochs = 0
    history: list[dict] = []
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        train_total = 0.0
        train_count = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = torch.mean((model(x) - y) ** 2)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["gradient_clip"]))
            scaler.step(optimizer)
            scaler.update()
            train_total += float(loss.detach()) * len(x)
            train_count += len(x)
        validation_loss = mean_loss(model, validation_loader, device, use_amp)
        train_loss = train_total / train_count
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss}
        )
        logging.info(
            "epoch=%d train_loss=%.6f validation_loss=%.6f",
            epoch,
            train_loss,
            validation_loss,
        )
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(cfg["patience"]):
                break
    if best_state is None:
        raise RuntimeError("No best model state was recorded")
    return best_state, history, len(history), best_loss


def score_sequences(
    model: nn.Module,
    sequences: list[np.ndarray],
    input_indices: np.ndarray,
    output_indices: np.ndarray,
    window: int,
    batch_size: int,
    device: torch.device,
) -> list[np.ndarray]:
    model.eval()
    results: list[np.ndarray] = []
    use_amp = device.type == "cuda"
    with torch.no_grad():
        for sequence in sequences:
            scores: list[np.ndarray] = []
            for start in range(window, len(sequence), batch_size):
                targets = np.arange(start, min(start + batch_size, len(sequence)))
                x = np.stack(
                    [sequence[target - window : target, input_indices] for target in targets]
                )
                y = sequence[targets[:, None], output_indices]
                xb = torch.from_numpy(x).to(device, non_blocking=True)
                yb = torch.from_numpy(y).to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=use_amp
                ):
                    prediction = model(xb)
                    batch_scores = torch.mean(torch.abs(prediction - yb), dim=1)
                scores.append(batch_scores.float().cpu().numpy())
            results.append(np.concatenate(scores))
    return results


def apply_alarm_rule(scores: np.ndarray, threshold: float, consecutive: int) -> np.ndarray:
    raw = scores > threshold
    if consecutive <= 1:
        return raw
    starts = np.flatnonzero(
        np.convolve(raw.astype(np.int8), np.ones(consecutive, dtype=np.int8), mode="valid")
        == consecutive
    )
    alarms = np.zeros(len(scores), dtype=bool)
    alarms[starts + consecutive - 1] = True
    return alarms


def event_metrics(labels: np.ndarray, predictions: np.ndarray) -> tuple[int, float, float]:
    padded = np.r_[0, labels.astype(np.int8), 0]
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1) - 1
    delays: list[int] = []
    detected = 0
    for start, end in zip(starts, ends):
        hits = np.flatnonzero(predictions[start : end + 1])
        if len(hits):
            detected += 1
            delays.append(int(hits[0]))
        else:
            delays.append(int(end - start + 1))
    return len(starts), detected / len(starts) if len(starts) else np.nan, float(np.mean(delays)) if delays else np.nan


def etapr_metrics(
    labels: list[np.ndarray],
    predictions: list[np.ndarray],
    repository: Path,
    theta_p: float,
    theta_r: float,
    delta: float,
) -> dict:
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    from eTaPR_pkg import etapr

    joined_labels: list[int] = []
    joined_predictions: list[int] = []
    for label, prediction in zip(labels, predictions):
        joined_labels.extend(label.astype(int).tolist())
        joined_predictions.extend(prediction.astype(int).tolist())
        joined_labels.append(0)
        joined_predictions.append(0)
    result = etapr.evaluate_w_streams(
        joined_labels,
        joined_predictions,
        theta_p=theta_p,
        theta_r=theta_r,
        delta=delta,
    )
    return {
        "etap": float(result["eTaP"]),
        "etar": float(result["eTaR"]),
        "etaf1": float(result["f1"]),
        "false_alarm_seconds": int(result["False Alarm"]),
        "n_false_alarms": int(result["N False Alarm"]),
    }


def evaluate_model(
    model: nn.Module,
    validation_sequences: list[np.ndarray],
    test_episodes: list[TestEpisode],
    input_indices: np.ndarray,
    output_indices: np.ndarray,
    cfg: dict,
    device: torch.device,
    etapr_repository: Path,
    attack_targets: pd.DataFrame,
) -> tuple[float, list[dict], list[dict], list[dict]]:
    validation_scores = score_sequences(
        model,
        validation_sequences,
        input_indices,
        output_indices,
        int(cfg["window"]),
        int(cfg["batch_size"]),
        device,
    )
    threshold = float(
        np.percentile(np.concatenate(validation_scores), float(cfg["threshold_percentile"]))
    )
    test_scores = score_sequences(
        model,
        [episode.features for episode in test_episodes],
        input_indices,
        output_indices,
        int(cfg["window"]),
        int(cfg["batch_size"]),
        device,
    )

    full_scores: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for episode, scores in zip(test_episodes, test_scores):
        aligned = np.full(len(episode.features), np.nan, dtype=np.float32)
        aligned[int(cfg["window"]) :] = scores
        full_scores.append(aligned)
        episode_predictions = np.zeros(len(episode.features), dtype=bool)
        episode_predictions[int(cfg["window"]) :] = apply_alarm_rule(
            scores, threshold, int(cfg["alarm_consecutive"])
        )
        predictions.append(episode_predictions)

    metric_rows: list[dict] = []
    episode_rows: list[dict] = []
    for label_index, label_name in enumerate(LABEL_COLUMNS):
        labels = [episode.labels[:, label_index].astype(bool) for episode in test_episodes]
        valid_labels = np.concatenate(
            [label[int(cfg["window"]) :] for label in labels]
        )
        valid_scores = np.concatenate(
            [score[int(cfg["window"]) :] for score in full_scores]
        )
        valid_predictions = np.concatenate(
            [prediction[int(cfg["window"]) :] for prediction in predictions]
        )
        events = 0
        detected_weighted = 0.0
        delay_sum = 0.0
        for episode, label, prediction in zip(test_episodes, labels, predictions):
            count, detected_ratio, censored_delay = event_metrics(label, prediction)
            events += count
            if count:
                detected_weighted += detected_ratio * count
                delay_sum += censored_delay * count
            episode_rows.append(
                {
                    "file": episode.name,
                    "label": label_name,
                    "rows": len(label),
                    "attack_rows": int(label.sum()),
                    "events": count,
                    "fpr": float(np.mean(prediction[~label])) if np.any(~label) else np.nan,
                    "event_detected_ratio": detected_ratio,
                    "censored_delay_mean": censored_delay,
                }
            )
        etapr_result = etapr_metrics(
            labels,
            predictions,
            etapr_repository,
            float(cfg["etapr_theta_p"]),
            float(cfg["etapr_theta_r"]),
            float(cfg["etapr_delta"]),
        )
        metric_rows.append(
            {
                "label": label_name,
                "rows": len(valid_labels),
                "attack_rows": int(valid_labels.sum()),
                "events": events,
                "auroc": roc_auc_score(valid_labels, valid_scores),
                "auprc": average_precision_score(valid_labels, valid_scores),
                "fpr": float(np.mean(valid_predictions[~valid_labels])),
                "event_detected_ratio": detected_weighted / events if events else np.nan,
                "censored_delay_mean": delay_sum / events if events else np.nan,
                **etapr_result,
            }
        )

    target_rows: list[dict] = []
    for episode, scores, prediction in zip(test_episodes, full_scores, predictions):
        labels = episode.labels[:, 0]
        padded = np.r_[0, labels, 0]
        starts = np.flatnonzero(np.diff(padded) == 1)
        ends = np.flatnonzero(np.diff(padded) == -1) - 1
        metadata = attack_targets.loc[attack_targets.file == episode.name].set_index(
            "event_index"
        )
        if len(metadata) != len(starts):
            raise ValueError(f"Attack-target count mismatch in {episode.name}")
        for event_index, (start, end) in enumerate(zip(starts, ends), start=1):
            meta = metadata.loc[event_index]
            hits = np.flatnonzero(prediction[start : end + 1])
            detected = bool(len(hits))
            target_rows.append(
                {
                    "file": episode.name,
                    "event_index": event_index,
                    "global_event": int(meta.global_event),
                    "attack_id": meta.id,
                    "target_points": meta.target_points,
                    "target_class": meta.target_class,
                    "start_row": int(start),
                    "end_row": int(end),
                    "duration_seconds": int(end - start + 1),
                    "detected": detected,
                    "alarm_delay_censored": int(hits[0]) if detected else int(end - start + 1),
                    "mean_event_score": float(np.nanmean(scores[start : end + 1])),
                    "max_event_score": float(np.nanmax(scores[start : end + 1])),
                }
            )
    return threshold, metric_rows, episode_rows, target_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hai_external_validation.yaml")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--variants", nargs="+")
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    seeds = args.seeds or cfg["seeds"]
    variants = args.variants or cfg["variants"]
    source = (ROOT / cfg["source"]).resolve()
    prepared = (ROOT / cfg["prepared"]).resolve()
    output = (ROOT / cfg["output"]).resolve()
    etapr_repository = (ROOT / cfg["etapr_repository"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(output / "HAI_EXTERNAL_VALIDATION.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    preparation = json.loads(
        (prepared / "HAI_2103_PREPARATION_MANIFEST.json").read_text(encoding="utf-8")
    )
    role_manifest = json.loads(
        (ROOT / cfg["roles_manifest"]).read_text(encoding="utf-8")
    )
    attack_target_manifest = json.loads(
        (ROOT / cfg["attack_target_manifest"]).read_text(encoding="utf-8")
    )
    attack_targets = pd.read_csv((ROOT / cfg["attack_targets"]).resolve())
    if preparation["assessment"] != "PASS_TO_ROLE_MAPPING":
        raise ValueError("HAI preparation gate has not passed")
    if preparation["remaining_train_test_hash_candidates"] != 0:
        raise ValueError("Train-test overlap remains after preparation")
    if role_manifest["assessment"] != "PASS_TO_MODELING":
        raise ValueError("HAI point-role gate has not passed")
    if attack_target_manifest["assessment"] != "PASS_WITH_SCOPE_LIMITATION":
        raise ValueError("HAI attack-target gate has not passed")
    if len(attack_targets) != 50:
        raise ValueError("Expected 50 validated HAI attack targets")

    sensor_columns = role_manifest["f0_active_columns"]
    f1_columns = role_manifest["f1_active_columns"]
    if f1_columns[: len(sensor_columns)] != sensor_columns:
        raise ValueError("F1 columns must begin with the ordered F0 sensor columns")
    train_sequences, validation_sequences, test_episodes, mean, std = load_data(
        source, prepared, f1_columns, float(cfg["train_fraction"])
    )
    feature_order_sha256 = hashlib.sha256(
        "\n".join(f1_columns).encode("utf-8")
    ).hexdigest()
    pd.DataFrame(
        {
            "position": np.arange(len(f1_columns), dtype=np.int64),
            "feature": f1_columns,
            "role": ["sensor"] * len(sensor_columns)
            + ["control"] * (len(f1_columns) - len(sensor_columns)),
            "mean": mean,
            "std": std,
        }
    ).to_csv(output / "HAI_EXTERNAL_SCALER.csv", index=False)
    sensor_indices = np.arange(len(sensor_columns), dtype=np.int64)
    f1_indices = np.arange(len(f1_columns), dtype=np.int64)
    target_indices = sensor_indices
    device = torch.device(
        cfg["device"] if cfg["device"] == "cpu" or torch.cuda.is_available() else "cpu"
    )
    f1_parameter_target = parameter_count(
        len(f1_indices), int(cfg["hidden_dim"]), int(cfg["layers"]), len(target_indices)
    )
    f0c_hidden = matched_hidden_dim(
        f1_parameter_target, len(sensor_indices), int(cfg["layers"]), len(target_indices)
    )
    definitions = {
        "F0": (sensor_indices, int(cfg["hidden_dim"])),
        "F1": (f1_indices, int(cfg["hidden_dim"])),
        "F0-C": (sensor_indices, f0c_hidden),
    }

    metric_path = output / "HAI_EXTERNAL_METRICS.partial.csv"
    episode_path = output / "HAI_EXTERNAL_EPISODES.partial.csv"
    training_path = output / "HAI_EXTERNAL_TRAINING.partial.csv"
    target_path = output / "HAI_EXTERNAL_TARGET_EVENTS.partial.csv"
    metric_frame = pd.read_csv(metric_path) if metric_path.exists() else pd.DataFrame()
    episode_frame = pd.read_csv(episode_path) if episode_path.exists() else pd.DataFrame()
    training_frame = pd.read_csv(training_path) if training_path.exists() else pd.DataFrame()
    target_frame = pd.read_csv(target_path) if target_path.exists() else pd.DataFrame()
    completed: set[tuple[int, str]] = set()
    if len(metric_frame) and len(episode_frame):
        for (seed, variant), group in metric_frame.groupby(["seed", "variant"]):
            episode_group = episode_frame.loc[
                (episode_frame.seed == seed) & (episode_frame.variant == variant)
            ]
            target_group = target_frame.loc[
                (target_frame.seed == seed) & (target_frame.variant == variant)
            ] if len(target_frame) else pd.DataFrame()
            checkpoint_path = checkpoint_dir / f"hai_{str(variant).lower()}_seed_{int(seed)}.pt"
            checkpoint_valid = False
            if checkpoint_path.exists() and variant in definitions:
                checkpoint = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
                expected_inputs = [
                    f1_columns[index] for index in definitions[str(variant)][0]
                ]
                checkpoint_valid = (
                    checkpoint.get("input_columns") == expected_inputs
                    and checkpoint.get("target_columns") == sensor_columns
                    and np.allclose(checkpoint.get("mean"), mean)
                    and np.allclose(checkpoint.get("std"), std)
                )
            if (
                len(group) == len(LABEL_COLUMNS)
                and set(group.label) == set(LABEL_COLUMNS)
                and len(episode_group) == len(test_episodes) * len(LABEL_COLUMNS)
                and len(target_group) == len(attack_targets)
                and checkpoint_valid
            ):
                completed.add((int(seed), str(variant)))
    if completed:
        metric_frame = metric_frame.loc[
            [
                (int(row.seed), str(row.variant)) in completed
                for row in metric_frame.itertuples(index=False)
            ]
        ]
        episode_frame = episode_frame.loc[
            [
                (int(row.seed), str(row.variant)) in completed
                for row in episode_frame.itertuples(index=False)
            ]
        ]
        if len(training_frame):
            training_frame = training_frame.loc[
                [
                    (int(row.seed), str(row.variant)) in completed
                    for row in training_frame.itertuples(index=False)
                ]
            ]
        if len(target_frame):
            target_frame = target_frame.loc[
                [
                    (int(row.seed), str(row.variant)) in completed
                    for row in target_frame.itertuples(index=False)
                ]
            ]
    else:
        metric_frame = pd.DataFrame()
        episode_frame = pd.DataFrame()
        training_frame = pd.DataFrame()
        target_frame = pd.DataFrame()
    metric_rows: list[dict] = metric_frame.to_dict("records")
    episode_rows: list[dict] = episode_frame.to_dict("records")
    training_rows: list[dict] = training_frame.to_dict("records")
    target_rows: list[dict] = target_frame.to_dict("records")
    for seed in seeds:
        for variant in variants:
            if variant not in definitions:
                raise ValueError(f"Unknown variant: {variant}")
            if (seed, variant) in completed:
                logging.info("skip completed seed=%d variant=%s", seed, variant)
                continue
            set_seed(seed)
            input_indices, hidden_dim = definitions[variant]
            model = GRUForecaster(
                len(input_indices), hidden_dim, int(cfg["layers"]), len(target_indices)
            ).to(device)
            params = sum(parameter.numel() for parameter in model.parameters())
            logging.info(
                "seed=%d variant=%s inputs=%d hidden=%d params=%d device=%s",
                seed,
                variant,
                len(input_indices),
                hidden_dim,
                params,
                device,
            )
            train_loader = make_loader(
                train_sequences,
                input_indices,
                target_indices,
                int(cfg["window"]),
                int(cfg["train_stride"]),
                int(cfg["batch_size"]),
                shuffle=True,
            )
            validation_loader = make_loader(
                validation_sequences,
                input_indices,
                target_indices,
                int(cfg["window"]),
                1,
                int(cfg["batch_size"]),
                shuffle=False,
            )
            best_state, history, epochs_run, best_validation_loss = train_model(
                model, train_loader, validation_loader, cfg, device
            )
            model.load_state_dict(best_state)
            checkpoint_path = checkpoint_dir / f"hai_{variant.lower()}_seed_{seed}.pt"
            torch.save(
                {
                    "state_dict": best_state,
                    "seed": seed,
                    "variant": variant,
                    "input_columns": [f1_columns[index] for index in input_indices],
                    "target_columns": sensor_columns,
                    "mean": mean,
                    "std": std,
                    "hidden_dim": hidden_dim,
                    "layers": int(cfg["layers"]),
                    "parameters": params,
                    "history": history,
                },
                checkpoint_path,
            )
            threshold, model_metrics, model_episodes, model_targets = evaluate_model(
                model,
                validation_sequences,
                test_episodes,
                input_indices,
                target_indices,
                cfg,
                device,
                etapr_repository,
                attack_targets,
            )
            for row in model_metrics:
                metric_rows.append(
                    {
                        "seed": seed,
                        "variant": variant,
                        "input_dim": len(input_indices),
                        "hidden_dim": hidden_dim,
                        "parameters": params,
                        "epochs_run": epochs_run,
                        "best_validation_loss": best_validation_loss,
                        "threshold": threshold,
                        **row,
                    }
                )
            for row in model_episodes:
                episode_rows.append({"seed": seed, "variant": variant, **row})
            for row in history:
                training_rows.append({"seed": seed, "variant": variant, **row})
            for row in model_targets:
                target_rows.append({"seed": seed, "variant": variant, **row})
            pd.DataFrame(metric_rows).to_csv(
                metric_path, index=False
            )
            pd.DataFrame(episode_rows).to_csv(
                episode_path, index=False
            )
            pd.DataFrame(training_rows).to_csv(
                training_path, index=False
            )
            pd.DataFrame(target_rows).to_csv(target_path, index=False)

    pd.DataFrame(metric_rows).to_csv(output / "HAI_EXTERNAL_METRICS.csv", index=False)
    pd.DataFrame(episode_rows).to_csv(output / "HAI_EXTERNAL_EPISODES.csv", index=False)
    pd.DataFrame(training_rows).to_csv(output / "HAI_EXTERNAL_TRAINING.csv", index=False)
    pd.DataFrame(target_rows).to_csv(output / "HAI_EXTERNAL_TARGET_EVENTS.csv", index=False)
    manifest = {
        "status": "COMPLETE",
        "device": str(device),
        "seeds": seeds,
        "variants": variants,
        "source_revision": preparation["source_revision"],
        "train_rows_after_overlap_exclusion": preparation["train_retained_rows"],
        "train_sequences": [len(sequence) for sequence in train_sequences],
        "validation_sequences": [len(sequence) for sequence in validation_sequences],
        "test_sequences": {episode.name: len(episode.features) for episode in test_episodes},
        "sensor_columns": len(sensor_columns),
        "sensor_column_names": sensor_columns,
        "control_columns_added": len(f1_columns) - len(sensor_columns),
        "f1_column_names": f1_columns,
        "feature_order_sha256": feature_order_sha256,
        "f1_parameter_target": f1_parameter_target,
        "f0c_hidden_dim": f0c_hidden,
        "metric_rows": len(metric_rows),
        "episode_rows": len(episode_rows),
        "target_event_rows": len(target_rows),
        "attack_target_class_counts": attack_target_manifest["target_class_counts"],
        "attack_target_scope_limitation": attack_target_manifest["scope_limitation"],
        "etapr_repository_revision": subprocess_revision(etapr_repository),
    }
    (output / "RUN_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def subprocess_revision(repository: Path) -> str:
    import subprocess

    return subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()


if __name__ == "__main__":
    main()

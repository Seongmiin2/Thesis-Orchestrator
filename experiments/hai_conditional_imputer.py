from __future__ import annotations

import numpy as np


def predictor_matrix(sequence: np.ndarray, sensor_count: int) -> np.ndarray:
    control_count = sequence.shape[1] - sensor_count
    if sensor_count < 1 or control_count < 1:
        raise ValueError("Sequence must contain both sensor and control columns")
    return np.c_[
        sequence[1:, :sensor_count],
        sequence[:-1, :sensor_count],
        sequence[:-1, sensor_count:],
        np.ones(len(sequence) - 1, dtype=np.float32),
    ]


def fit_leave_one_channel_out(
    sequences: list[np.ndarray], sensor_count: int, ridge: float
) -> np.ndarray:
    control_count = sequences[0].shape[1] - sensor_count
    predictor_dim = sensor_count * 2 + control_count + 1
    xtx = np.zeros((predictor_dim, predictor_dim), dtype=np.float64)
    xty = np.zeros((predictor_dim, control_count), dtype=np.float64)
    for sequence in sequences:
        x = predictor_matrix(np.asarray(sequence, dtype=np.float64), sensor_count)
        y = np.asarray(sequence[1:, sensor_count:], dtype=np.float64)
        xtx += x.T @ x
        xty += x.T @ y

    weights = np.zeros((predictor_dim, control_count), dtype=np.float32)
    penalty = np.eye(predictor_dim, dtype=np.float64) * float(ridge)
    penalty[-1, -1] = 0.0
    previous_control_start = sensor_count * 2
    all_columns = np.arange(predictor_dim)
    for channel in range(control_count):
        excluded = previous_control_start + channel
        selected = all_columns[all_columns != excluded]
        solution = np.linalg.solve(
            xtx[np.ix_(selected, selected)] + penalty[np.ix_(selected, selected)],
            xty[selected, channel],
        )
        weights[selected, channel] = solution.astype(np.float32)
    return weights


def predict_leave_one_channel_out(
    sequence: np.ndarray, sensor_count: int, weights: np.ndarray
) -> np.ndarray:
    return predictor_matrix(sequence, sensor_count) @ weights


def build_residual_bank(
    sequences: list[np.ndarray], sensor_count: int, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    residuals: list[np.ndarray] = []
    segment_rows: list[tuple[int, int]] = []
    offset = 0
    for sequence in sequences:
        predicted = predict_leave_one_channel_out(sequence, sensor_count, weights)
        residual = sequence[1:, sensor_count:] - predicted
        residuals.append(residual.astype(np.float32))
        segment_rows.append((offset, offset + len(residual)))
        offset += len(residual)
    bank = np.concatenate(residuals)
    bank -= bank.mean(axis=0, keepdims=True)
    return bank.astype(np.float32), np.asarray(segment_rows, dtype=np.int64)


def sample_residual_blocks(
    residual_bank: np.ndarray,
    segments: np.ndarray,
    channel: int,
    length: int,
    rng: np.random.Generator,
    block_length: int,
) -> np.ndarray:
    if channel < 0 or channel >= residual_bank.shape[1]:
        raise ValueError("Invalid residual channel")
    if block_length < 1:
        raise ValueError("Residual block length must be positive")
    available = segments[:, 1] - segments[:, 0]
    if np.any(available < 1):
        raise ValueError("Residual segments must be non-empty")
    result = np.empty(length, dtype=np.float32)
    offset = 0
    while offset < length:
        take = min(block_length, length - offset)
        eligible = available >= take
        starts_available = np.where(eligible, available - take + 1, 0)
        probabilities = starts_available / starts_available.sum()
        segment_index = int(rng.choice(len(segments), p=probabilities))
        segment_start, segment_end = segments[segment_index]
        start = int(rng.integers(segment_start, segment_end - take + 1))
        result[offset : offset + take] = residual_bank[
            start : start + take, channel
        ]
        offset += take
    return result


def replace_control_channel(
    sequence: np.ndarray,
    predictions: np.ndarray,
    sensor_count: int,
    channel: int,
    mode: str,
    residual_bank: np.ndarray | None = None,
    segments: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    block_length: int = 20,
) -> np.ndarray:
    replaced = np.array(sequence, copy=True)
    column = sensor_count + channel
    if mode == "zero":
        replaced[:, column] = 0.0
    elif mode == "loo_sample":
        if residual_bank is None or segments is None or rng is None:
            raise ValueError("LOO sampling requires residual data and an RNG")
        sampled = sample_residual_blocks(
            residual_bank, segments, channel, len(sequence) - 1, rng, block_length
        )
        replaced[1:, column] = predictions[:, channel] + sampled
    else:
        raise ValueError(f"Unknown replacement mode: {mode}")
    return replaced

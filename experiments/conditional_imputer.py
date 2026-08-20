from __future__ import annotations

import numpy as np


N_XMEAS = 41
N_XMV = 11
PREDICTOR_DIM = N_XMEAS * 2 + N_XMV + 1


def predictor_matrix(standardized: np.ndarray) -> np.ndarray:
    return np.c_[
        standardized[1:, :N_XMEAS],
        standardized[:-1, :N_XMEAS],
        standardized[:-1, N_XMEAS:],
        np.ones(len(standardized) - 1, dtype=np.float32),
    ]


def fit_leave_one_channel_out(
    features: np.ndarray,
    train_runs: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    ridge: float = 1e-2,
) -> np.ndarray:
    xtx = np.zeros((PREDICTOR_DIM, PREDICTOR_DIM), dtype=np.float64)
    xty = np.zeros((PREDICTOR_DIM, N_XMV), dtype=np.float64)
    for run in train_runs:
        array = np.asarray(features[int(run), :599], dtype=np.float64)
        standardized = (array - mean) / std
        x = predictor_matrix(standardized)
        y = standardized[1:, N_XMEAS:]
        xtx += x.T @ x
        xty += x.T @ y

    weights = np.zeros((PREDICTOR_DIM, N_XMV), dtype=np.float32)
    penalty = np.eye(PREDICTOR_DIM, dtype=np.float64) * ridge
    penalty[-1, -1] = 0.0
    previous_xmv_start = N_XMEAS * 2
    all_columns = np.arange(PREDICTOR_DIM)
    for channel in range(N_XMV):
        excluded = previous_xmv_start + channel
        selected = all_columns[all_columns != excluded]
        solution = np.linalg.solve(
            xtx[np.ix_(selected, selected)] + penalty[np.ix_(selected, selected)],
            xty[selected, channel],
        )
        weights[selected, channel] = solution.astype(np.float32)
    return weights


def predict_leave_one_channel_out(
    standardized: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    return predictor_matrix(standardized) @ weights


def build_residual_bank(
    features: np.ndarray,
    train_runs: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    residuals = np.empty((len(train_runs), 598, N_XMV), dtype=np.float32)
    for index, run in enumerate(train_runs):
        array = np.asarray(features[int(run), :599], dtype=np.float32)
        standardized = (array - mean) / std
        predicted = predict_leave_one_channel_out(standardized, weights)
        residuals[index] = standardized[1:, N_XMEAS:] - predicted
    residuals -= residuals.mean(axis=(0, 1), keepdims=True)
    return residuals


def sample_residual_blocks(
    residual_bank: np.ndarray,
    channel: int,
    length: int,
    rng: np.random.Generator,
    block_length: int,
) -> np.ndarray:
    if channel < 0 or channel >= N_XMV:
        raise ValueError(f"Channel must be in [0, {N_XMV - 1}]")
    if block_length < 1 or block_length > residual_bank.shape[1]:
        raise ValueError("Invalid residual block length")
    result = np.empty(length, dtype=np.float32)
    offset = 0
    while offset < length:
        take = min(block_length, length - offset)
        run = int(rng.integers(0, len(residual_bank)))
        start = int(rng.integers(0, residual_bank.shape[1] - take + 1))
        result[offset : offset + take] = residual_bank[run, start : start + take, channel]
        offset += take
    return result

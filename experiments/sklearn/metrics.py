from __future__ import annotations

import numpy as np


def roc_auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.arange(1, len(scores) + 1, dtype=np.float64)
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    ends = np.r_[starts[1:], len(scores)]
    for lo, hi in zip(starts, ends):
        ranks[lo:hi] = (lo + 1 + hi) / 2
    ranked = np.empty_like(ranks)
    ranked[order] = ranks
    positive = labels == 1
    n_pos = int(positive.sum())
    n_neg = len(labels) - n_pos
    return float((ranked[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision_score(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    order = np.argsort(-np.asarray(scores), kind="stable")
    sorted_labels = labels[order]
    n_pos = int(sorted_labels.sum())
    precision = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
    return float(precision[sorted_labels == 1].sum() / n_pos)

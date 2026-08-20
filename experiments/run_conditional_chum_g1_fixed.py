"""Corrected runner shim aligning imputed XMV at raw time r with predicted index r-1."""
from __future__ import annotations

import numpy as np
import torch

import run_conditional_chum_g1 as experiment


def score_runs(model, features, run_ids, mean, std, weights, channel, start, end, window, device, batch_size):
    rows = []
    with torch.no_grad():
        for run in run_ids:
            array = np.asarray(features[int(run)], dtype=np.float32)
            z = (array - mean) / std
            predictors = np.c_[z[1:, :41], z[:-1, 41:], np.ones(len(z) - 1, dtype=np.float32)]
            predicted = predictors @ weights
            windows = np.lib.stride_tricks.sliding_window_view(array, (window, 52))[: 2000 - window, 0]
            lo, hi = start - 1 - window, end - window
            x = (np.array(windows[lo:hi], copy=True) - mean) / std
            if channel is not None:
                replacement = np.lib.stride_tricks.sliding_window_view(predicted[:, channel], window)[lo - 1:hi - 1]
                x[:, :, 41 + channel] = replacement
            y = z[start - 1:end, :41]
            scores = []
            for offset in range(0, len(x), batch_size):
                xb = torch.from_numpy(x[offset:offset + batch_size]).to(device)
                yb = torch.from_numpy(y[offset:offset + batch_size]).to(device)
                scores.append(torch.mean(torch.abs(model(xb) - yb), dim=1).cpu().numpy())
            rows.append(np.concatenate(scores))
    return np.stack(rows)


if __name__ == "__main__":
    experiment.score_runs = score_runs
    experiment.main()

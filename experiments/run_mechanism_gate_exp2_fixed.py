"""Runner shim containing the corrected inclusive target slice (see issue found by smoke test)."""
from __future__ import annotations

import numpy as np
import torch

import run_mechanism_gate_exp2 as experiment


def score_runs(model, features, run_ids, mean, std, variant, start, end, window, device, batch_size):
    rows = []
    with torch.no_grad():
        for run in run_ids:
            array = np.asarray(features[int(run)], dtype=np.float32)
            windows = np.lib.stride_tricks.sliding_window_view(array, (window, 52))[: 2000 - window, 0]
            lo, hi = start - 1 - window, end - window
            x = (np.array(windows[lo:hi], copy=True) - mean) / std
            x = experiment.perturb(x, variant)
            y = (array[start - 1:end, :41] - mean[:41]) / std[:41]
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

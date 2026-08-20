# Existing CHUM Evidence Audit

## Overall Assessment: READY_TO_EXTEND

This audit independently recomputed the headline G1/G2 decisions from saved CSV artifacts.

## Split Integrity

Rows/unique runs: `2800/2800`; split counts: `{'train': 1792, 'test': 560, 'validation': 448}`; per-fault counts: `{'test': 20, 'train': 64, 'validation': 16}`; maximum splits per run: `1`.

## Architecture Gate

Recomputed GAIN sets: `{'tcn': [4, 7, 19, 23, 24, 25, 26], 'transformer': [4, 7, 19, 23, 24, 25, 26]}`.

## Conditional Attribution

Stable material cells under mean AUROC loss >= 0.02, >=4/5 positive seeds, and max absolute FPR shift <= 0.005: `7`.

|   fault_id |   channel |   mean_delta_auroc |   mean_delta_auprc |   positive_seeds |   max_abs_pre_fpr_shift |
|-----------:|----------:|-------------------:|-------------------:|-----------------:|------------------------:|
|          4 |        10 |             0.4191 |             0.3269 |                5 |                  0.0012 |
|         26 |         4 |             0.084  |             0.0533 |                5 |                  0.0008 |
|         27 |        10 |             0.0334 |             0.0215 |                5 |                  0.0005 |
|         19 |         8 |             0.0289 |             0.0176 |                5 |                  0.0007 |
|         23 |         3 |             0.0265 |             0.0221 |                5 |                  0.0003 |
|         11 |        10 |             0.0239 |             0.0133 |                5 |                  0.0005 |
|         25 |         2 |             0.0217 |             0.0142 |                5 |                  0.0008 |

## FPR Comparison

Seed-level fault-channel cells exceeding absolute pre-FPR shift 0.005: zero occlusion `9`, conditional replacement `0`.

## Issues

No headline-result discrepancy was found.

## Required Caveats

- Five training seeds are repeated model fits on one fixed dataset, not five independent datasets.
- Conditional replacement estimates predictive utility, not causal controller effects or physical root cause.
- The existing architecture result is event-level; individual channel consensus requires G3.

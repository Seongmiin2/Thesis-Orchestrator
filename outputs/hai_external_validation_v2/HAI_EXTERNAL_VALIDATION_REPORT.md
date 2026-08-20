# HAI 21.03 External Validation Report

## Decision

**EXTERNAL_SUPPORT** under the descriptive rule locked before the complete v2 result analysis. This is an external dataset check, not an independent causal proof.


## Global Attack Label

| variant   |   parameters |   auroc_mean |   auprc_mean |   etap_mean |   etar_mean |   etaf1_mean |   fpr_mean |   detected_mean |   delay_mean |
|:----------|-------------:|-------------:|-------------:|------------:|------------:|-------------:|-----------:|----------------:|-------------:|
| F0        |        26909 |       0.8331 |       0.4743 |      0.669  |      0.5269 |       0.5895 |     0.0027 |          0.82   |        47.48 |
| F0-C      |        31021 |       0.8301 |       0.4663 |      0.6621 |      0.5172 |       0.5807 |     0.0026 |          0.8133 |        49.18 |
| F1        |        30941 |       0.8518 |       0.5049 |      0.688  |      0.558  |       0.6162 |     0.0027 |          0.84   |        43.74 |

## Paired Seed Differences

Positive AUROC/AUPRC/eTaF1 values favor F1. Positive FPR values mean F1 has fewer false alarms; positive delay values mean F1 is earlier.

| label     | contrast   |   mean_delta_auroc |   positive_seeds_auroc |   mean_delta_auprc |   positive_seeds_auprc |   mean_delta_etaf1 |   positive_seeds_etaf1 |   mean_delta_fpr |   mean_delta_censored_delay_mean |
|:----------|:-----------|-------------------:|-----------------------:|-------------------:|-----------------------:|-------------------:|-----------------------:|-----------------:|---------------------------------:|
| attack    | F1-F0      |             0.0187 |                      3 |             0.0306 |                      3 |             0.0267 |                      3 |          -0      |                           3.74   |
| attack    | F1-F0-C    |             0.0217 |                      3 |             0.0386 |                      3 |             0.0355 |                      3 |          -0.0001 |                           5.44   |
| attack_P1 | F1-F0      |             0.0209 |                      3 |             0.0281 |                      3 |            -0.0009 |                      1 |          -0.0003 |                           0.6554 |
| attack_P1 | F1-F0-C    |             0.025  |                      3 |             0.0332 |                      3 |             0.0043 |                      2 |          -0.0003 |                           2.1695 |
| attack_P2 | F1-F0      |            -0.0011 |                      2 |            -0.0107 |                      0 |             0.0207 |                      3 |          -0.0008 |                           2.9477 |
| attack_P2 | F1-F0-C    |            -0.0019 |                      1 |             0.0019 |                      1 |             0.0276 |                      3 |          -0.0009 |                           2.8889 |
| attack_P3 | F1-F0      |             0.0253 |                      3 |             0.0068 |                      3 |             0.0181 |                      2 |          -0.001  |                           0.625  |
| attack_P3 | F1-F0-C    |             0.024  |                      3 |             0.0101 |                      3 |             0.0099 |                      3 |          -0.0011 |                           0.5833 |

## Process Labels

| variant   | label     |   auroc_mean |   auprc_mean |   etaf1_mean |   fpr_mean |   detected_mean |
|:----------|:----------|-------------:|-------------:|-------------:|-----------:|----------------:|
| F0        | attack_P1 |       0.8499 |       0.3583 |       0.4694 |     0.0042 |          0.5593 |
| F0        | attack_P2 |       0.8398 |       0.3046 |       0.1117 |     0.0101 |          0.8039 |
| F0        | attack_P3 |       0.8483 |       0.0445 |       0.1204 |     0.0117 |          0.75   |
| F0-C      | attack_P1 |       0.8457 |       0.3533 |       0.4642 |     0.0042 |          0.5537 |
| F0-C      | attack_P2 |       0.8406 |       0.2921 |       0.1047 |     0.01   |          0.8039 |
| F0-C      | attack_P3 |       0.8496 |       0.0413 |       0.1286 |     0.0116 |          0.75   |
| F1        | attack_P1 |       0.8707 |       0.3864 |       0.4685 |     0.0045 |          0.5706 |
| F1        | attack_P2 |       0.8387 |       0.2939 |       0.1323 |     0.0109 |          0.8366 |
| F1        | attack_P3 |       0.8736 |       0.0513 |       0.1385 |     0.0126 |          0.75   |
Process-specific FPR treats attacks assigned only to another process as negatives, so it is not directly comparable to global-label FPR.

## Attack-Target Subgroups

All 50 HAI 21.03 events directly target at least one control-history point: 31 target only control points and 19 jointly target control and sensor points.

| variant   | target_class         |   seeds |   events |   detected_ratio_mean |   detected_ratio_sd |   censored_delay_mean |
|:----------|:---------------------|--------:|---------:|----------------------:|--------------------:|----------------------:|
| F0        | control_only         |       3 |       31 |                0.7742 |              0      |               45.8172 |
| F0        | mixed_control_sensor |       3 |       19 |                0.8947 |              0      |               50.193  |
| F0-C      | control_only         |       3 |       31 |                0.7742 |              0      |               45.8065 |
| F0-C      | mixed_control_sensor |       3 |       19 |                0.8772 |              0.0304 |               54.6842 |
| F1        | control_only         |       3 |       31 |                0.7742 |              0      |               45.4301 |
| F1        | mixed_control_sensor |       3 |       19 |                0.9474 |              0      |               40.9825 |

Positive detection and delay deltas favor F1.

| target_class         | contrast   |   seeds |   mean_delta_detected_ratio |   positive_seeds_detected_ratio |   mean_delta_censored_delay |   positive_seeds_delay |
|:---------------------|:-----------|--------:|----------------------------:|--------------------------------:|----------------------------:|-----------------------:|
| control_only         | F1-F0      |       3 |                      0      |                               0 |                      0.3871 |                      3 |
| control_only         | F1-F0-C    |       3 |                      0      |                               0 |                      0.3763 |                      3 |
| mixed_control_sensor | F1-F0      |       3 |                      0.0526 |                               3 |                      9.2105 |                      3 |
| mixed_control_sensor | F1-F0-C    |       3 |                      0.0702 |                               3 |                     13.7018 |                      3 |

## Design Controls

F0 and F1 predict the same 29 next-step sensor/model targets. F1 adds 28 nonconstant control-history inputs. F0-C uses F0 inputs with a widened hidden state; its parameter gap from F1 is `0.26%`.
Training used `878,401` retained official-normal rows before chronological splitting. Exact train-test duplicates were excluded before scaling and window construction.
The official eTaPR implementation was used with fixed validation-calibrated thresholds. CSV files remain separate episodes during windowing and are separated by a normal gap for range evaluation.

## Statistical Boundary

Only three seeds are available, so the bootstrap intervals are descriptive sensitivity summaries and no seed-level significance claim is made. Process labels are partial annotations; the global attack label is primary. Medium-confidence HAI 21.03 alias mappings remain a documented external-validity limitation.
Because every event directly manipulates a control-history point, HAI cannot validate the stronger claim that control history helps when no control channel is attacked. The TEP fault experiments remain primary for that claim.

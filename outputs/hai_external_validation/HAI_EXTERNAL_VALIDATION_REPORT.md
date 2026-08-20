# HAI 21.03 External Validation Report

## Decision

**EXTERNAL_NONCONFIRMATION** under the preregistered descriptive rule. This is an external dataset check, not an independent causal proof.

## Global Attack Label

| variant   |   parameters |   auroc_mean |   auprc_mean |   etap_mean |   etar_mean |   etaf1_mean |   fpr_mean |   detected_mean |   delay_mean |
|:----------|-------------:|-------------:|-------------:|------------:|------------:|-------------:|-----------:|----------------:|-------------:|
| F0        |        26909 |       0.5077 |       0.2231 |           0 |           0 |            0 |          1 |               1 |            0 |
| F0-C      |        31021 |       0.5077 |       0.2231 |           0 |           0 |            0 |          1 |               1 |            0 |
| F1        |        30941 |       0.5077 |       0.2231 |           0 |           0 |            0 |          1 |               1 |            0 |

## Paired Seed Differences

Positive AUROC/AUPRC/eTaF1 values favor F1. Positive FPR values mean F1 has fewer false alarms; positive delay values mean F1 is earlier.

| label     | contrast   |   mean_delta_auroc |   positive_seeds_auroc |   mean_delta_auprc |   positive_seeds_auprc |   mean_delta_etaf1 |   positive_seeds_etaf1 |   mean_delta_fpr |   mean_delta_censored_delay_mean |
|:----------|:-----------|-------------------:|-----------------------:|-------------------:|-----------------------:|-------------------:|-----------------------:|-----------------:|---------------------------------:|
| attack    | F1-F0      |            -0      |                      1 |                  0 |                      3 |                  0 |                      0 |                0 |                                0 |
| attack    | F1-F0-C    |            -0      |                      1 |                  0 |                      2 |                  0 |                      0 |                0 |                                0 |
| attack_P1 | F1-F0      |            -0      |                      1 |                  0 |                      3 |                  0 |                      0 |                0 |                                0 |
| attack_P1 | F1-F0-C    |            -0      |                      1 |                  0 |                      1 |                  0 |                      0 |                0 |                                0 |
| attack_P2 | F1-F0      |             0      |                      1 |                  0 |                      2 |                  0 |                      0 |                0 |                                0 |
| attack_P2 | F1-F0-C    |            -0      |                      2 |                  0 |                      1 |                  0 |                      0 |                0 |                                0 |
| attack_P3 | F1-F0      |             0      |                      1 |                  0 |                      3 |                  0 |                      0 |                0 |                                0 |
| attack_P3 | F1-F0-C    |             0.0001 |                      2 |                 -0 |                      2 |                  0 |                      0 |                0 |                                0 |

## Process Labels

| variant   | label     |   auroc_mean |   auprc_mean |   etaf1_mean |   fpr_mean |   detected_mean |
|:----------|:----------|-------------:|-------------:|-------------:|-----------:|----------------:|
| F0        | attack_P1 |       0.5132 |       0.2691 |            0 |          1 |               1 |
| F0        | attack_P2 |       0.5453 |       0.0174 |            0 |          1 |               1 |
| F0        | attack_P3 |       0.3593 |       0.0239 |            0 |          1 |               1 |
| F0-C      | attack_P1 |       0.5131 |       0.2691 |            0 |          1 |               1 |
| F0-C      | attack_P2 |       0.5453 |       0.0174 |            0 |          1 |               1 |
| F0-C      | attack_P3 |       0.3592 |       0.0239 |            0 |          1 |               1 |
| F1        | attack_P1 |       0.5131 |       0.2691 |            0 |          1 |               1 |
| F1        | attack_P2 |       0.5453 |       0.0174 |            0 |          1 |               1 |
| F1        | attack_P3 |       0.3593 |       0.0239 |            0 |          1 |               1 |

## Attack-Target Subgroups

All 50 HAI 21.03 events directly target at least one control-history point: 31 target only control points and 19 jointly target control and sensor points.

| variant   | target_class         |   seeds |   events |   detected_ratio_mean |   detected_ratio_sd |   censored_delay_mean |
|:----------|:---------------------|--------:|---------:|----------------------:|--------------------:|----------------------:|
| F0        | control_only         |       3 |       31 |                     1 |                   0 |                     0 |
| F0        | mixed_control_sensor |       3 |       19 |                     1 |                   0 |                     0 |
| F0-C      | control_only         |       3 |       31 |                     1 |                   0 |                     0 |
| F0-C      | mixed_control_sensor |       3 |       19 |                     1 |                   0 |                     0 |
| F1        | control_only         |       3 |       31 |                     1 |                   0 |                     0 |
| F1        | mixed_control_sensor |       3 |       19 |                     1 |                   0 |                     0 |

Positive detection and delay deltas favor F1.

| target_class         | contrast   |   seeds |   mean_delta_detected_ratio |   positive_seeds_detected_ratio |   mean_delta_censored_delay |   positive_seeds_delay |
|:---------------------|:-----------|--------:|----------------------------:|--------------------------------:|----------------------------:|-----------------------:|
| control_only         | F1-F0      |       3 |                           0 |                               0 |                           0 |                      0 |
| control_only         | F1-F0-C    |       3 |                           0 |                               0 |                           0 |                      0 |
| mixed_control_sensor | F1-F0      |       3 |                           0 |                               0 |                           0 |                      0 |
| mixed_control_sensor | F1-F0-C    |       3 |                           0 |                               0 |                           0 |                      0 |

## Design Controls

F0 and F1 predict the same 29 next-step sensor/model targets. F1 adds 28 nonconstant control-history inputs. F0-C uses F0 inputs with a widened hidden state; its parameter gap from F1 is `0.26%`.
Training used `878,401` retained official-normal rows before chronological splitting. Exact train-test duplicates were excluded before scaling and window construction.
The official eTaPR implementation was used with fixed validation-calibrated thresholds. CSV files remain separate episodes during windowing and are separated by a normal gap for range evaluation.

## Statistical Boundary

Only three seeds are available, so the bootstrap intervals are descriptive sensitivity summaries and no seed-level significance claim is made. Process labels are partial annotations; the global attack label is primary. Medium-confidence HAI 21.03 alias mappings remain a documented external-validity limitation.
Because every event directly manipulates a control-history point, HAI cannot validate the stronger claim that control history helps when no control channel is attacked. The TEP fault experiments remain primary for that claim.

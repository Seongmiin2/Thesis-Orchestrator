# HAI 21.03 Conditional CHUM Report

## Decision

**EXTERNAL_CHANNEL_SUPPORT**: `5` directly targeted event-channel cells passed the locked conditional criteria; `3` were required.
Conditional replacement produced `0` FPR exceptions across reliable seed-channel cells versus `0` for the zero-replacement diagnostic.

## Passing Targeted Cells

|   global_event | attack_id   | target_class         | feature   |   mean_delta_normalized_score |   min_delta_normalized_score |   positive_seeds |   mean_detection_drop |   mean_alarm_delay_increase |   max_abs_fpr_shift |
|---------------:|:------------|:---------------------|:----------|------------------------------:|-----------------------------:|-----------------:|----------------------:|----------------------------:|--------------------:|
|             30 | A305        | control_only         | P2_CO_rpm |                        0.2146 |                       0.1663 |                3 |                0      |                       0     |                   0 |
|              8 | A203        | control_only         | P1_LCV01D |                        0.1892 |                       0.1519 |                3 |                0      |                       0     |                   0 |
|             44 | A506        | mixed_control_sensor | P2_CO_rpm |                        0.1078 |                       0.0827 |                3 |                0      |                       0     |                   0 |
|             20 | A215        | mixed_control_sensor | P2_CO_rpm |                        0.0907 |                       0.0452 |                3 |                0.6667 |                     100.667 |                   0 |
|             21 | A216        | control_only         | P1_FCV03D |                        0.0588 |                       0.0503 |                3 |                0      |                       0     |                   0 |

## Strongest Quality-Gated Targeted Cells

|   global_event | attack_id   | target_class         | feature   |   mean_delta_normalized_score |   min_delta_normalized_score |   positive_seeds |   mean_detection_drop |   mean_alarm_delay_increase |   max_abs_fpr_shift |
|---------------:|:------------|:---------------------|:----------|------------------------------:|-----------------------------:|-----------------:|----------------------:|----------------------------:|--------------------:|
|             30 | A305        | control_only         | P2_CO_rpm |                        0.2146 |                       0.1663 |                3 |                0      |                      0      |                   0 |
|              8 | A203        | control_only         | P1_LCV01D |                        0.1892 |                       0.1519 |                3 |                0      |                      0      |                   0 |
|             44 | A506        | mixed_control_sensor | P2_CO_rpm |                        0.1078 |                       0.0827 |                3 |                0      |                      0      |                   0 |
|             20 | A215        | mixed_control_sensor | P2_CO_rpm |                        0.0907 |                       0.0452 |                3 |                0.6667 |                    100.667  |                   0 |
|             21 | A216        | control_only         | P1_FCV03D |                        0.0588 |                       0.0503 |                3 |                0      |                      0      |                   0 |
|             37 | A404        | mixed_control_sensor | P2_CO_rpm |                        0.0499 |                       0.0352 |                3 |                0      |                      0      |                   0 |
|             46 | A508        | control_only         | P1_LCV01D |                        0.0439 |                       0.0147 |                3 |                0      |                      1.6667 |                   0 |
|              4 | A104        | control_only         | P2_CO_rpm |                        0.0421 |                      -0.1711 |                2 |                0      |                      0      |                   0 |
|             41 | A503        | mixed_control_sensor | P2_CO_rpm |                        0.0408 |                       0.0314 |                3 |                0      |                      0      |                   0 |
|             40 | A502        | mixed_control_sensor | P2_CO_rpm |                        0.0227 |                       0.0133 |                3 |                0      |                      0      |                   0 |
|             46 | A508        | control_only         | P1_FCV03D |                        0.0192 |                       0.0035 |                3 |                0      |                      0      |                   0 |
|             48 | A510        | mixed_control_sensor | P1_LCV01D |                        0.0083 |                      -0.0029 |                2 |                0      |                      1      |                   0 |
|             26 | A301        | control_only         | P1_LCV01D |                        0.0081 |                       0.0016 |                3 |                0      |                      0.6667 |                   0 |
|             42 | A504        | control_only         | P1_FCV03D |                        0.0028 |                      -0.0021 |                2 |                0      |                      0      |                   0 |
|             50 | A512        | mixed_control_sensor | P1_FCV03D |                        0.0015 |                       0.0003 |                3 |                0      |                      0      |                   0 |
|             40 | A502        | mixed_control_sensor | P1_FCV03D |                        0.0011 |                      -0.0015 |                2 |                0      |                      0      |                   0 |
|             11 | A206        | mixed_control_sensor | P1_FCV03D |                        0.0006 |                      -0.0003 |                2 |                0      |                      0      |                   0 |
|             23 | A218        | control_only         | P1_FCV03D |                        0.0002 |                      -0.003  |                1 |                0      |                      0      |                   0 |
|             10 | A205        | control_only         | P1_B2016  |                        0.0002 |                      -0      |                2 |                0      |                      0      |                   0 |
|              7 | A202        | mixed_control_sensor | P1_B2016  |                        0.0001 |                      -0.0009 |                2 |                0      |                      0      |                   0 |

## FPR Robustness

| mode       |   seed_channel_cells |   fpr_exceptions |   max_abs_fpr_shift |
|:-----------|---------------------:|-----------------:|--------------------:|
| loo_sample |                   36 |                0 |             2.5e-05 |
| zero       |                   36 |                0 |             8.1e-05 |

## Imputer Gate

`12` of 28 active control channels passed the held-out predictive, variance, mean, lag, and range checks: `['P1_B2004', 'P1_B2016', 'P1_B4002', 'P1_B4022', 'P1_FCV03D', 'P1_FCV03Z', 'P1_LCV01D', 'P1_LCV01Z', 'P2_CO_rpm', 'P4_LD', 'P4_ST_GOV', 'P4_ST_LD']`.
Only `4` quality-gated channels are directly named by at least one audited HAI attack and therefore enter targeted cell decisions: `['P1_B2016', 'P1_FCV03D', 'P1_LCV01D', 'P2_CO_rpm']`.

## Interpretation Boundary

Positive score deltas mean that replacing the directly attacked control history reduced the threshold-normalized attack score. The comparison uses fixed completed F1 checkpoints and recalibrates each threshold on equally perturbed validation-normal data.
These are conditional predictive-information results, not physical interventions, controller root-cause labels, or causal plant-mechanism estimates. Because every HAI event directly attacks at least one control-history point, TEP remains the primary evidence for faults without direct control manipulation.
Only three seeds are available, so seed summaries are descriptive and the all-seed direction rule is used as a stability filter rather than a significance test.

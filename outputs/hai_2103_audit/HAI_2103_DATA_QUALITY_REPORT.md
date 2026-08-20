# HAI 21.03 Data Quality Audit

## Overall Assessment: BLOCKED

Source commit: `2a814cebc9a66b06c9e5cd545e2d72e65d383737`. Intended grain: one SCADA observation per second within each of eight continuous CSV episodes.

## Dataset and Grain

Files: `8`; rows: `1,323,608`; schema: `time + 79 points + 4 labels`; test attack rows: `8,947`.

| file          | split   |   rows |   columns | first_time          | last_time           |   invalid_timestamps |   duplicate_timestamps |   non_one_second_steps |   numeric_nulls |   infinities |   invalid_label_values |   process_label_outside_global |   global_attack_without_process_label |   attack_positive_rows |   attack_P1_positive_rows |   attack_P2_positive_rows |   attack_P3_positive_rows |
|:--------------|:--------|-------:|----------:|:--------------------|:--------------------|---------------------:|-----------------------:|-----------------------:|----------------:|-------------:|-----------------------:|-------------------------------:|--------------------------------------:|-----------------------:|--------------------------:|--------------------------:|--------------------------:|
| test1.csv.gz  | test    |  43201 |        84 | 2020-07-07 15:00:00 | 2020-07-08 03:00:00 |                    0 |                      0 |                      0 |               0 |            0 |                      0 |                              0 |                                     0 |                    629 |                       480 |                       149 |                         0 |
| test2.csv.gz  | test    | 118801 |        84 | 2020-07-09 15:00:00 | 2020-07-11 00:00:00 |                    0 |                      0 |                      0 |               0 |            0 |                      0 |                              0 |                                   154 |                   3449 |                      2414 |                       643 |                       238 |
| test3.csv.gz  | test    | 108001 |        84 | 2020-07-13 00:00:00 | 2020-07-14 06:00:00 |                    0 |                      0 |                      0 |               0 |            0 |                      0 |                              0 |                                     3 |                   1535 |                      1337 |                       372 |                         0 |
| test4.csv.gz  | test    |  39601 |        84 | 2020-07-28 12:00:00 | 2020-07-28 23:00:00 |                    0 |                      0 |                      0 |               0 |            0 |                      0 |                              0 |                                     0 |                   1157 |                      1035 |                       175 |                       360 |
| test5.csv.gz  | test    |  92401 |        84 | 2020-07-30 10:50:00 | 2020-07-31 12:30:00 |                    0 |                      0 |                      0 |               0 |            0 |                      0 |                              0 |                                    22 |                   2177 |                      1771 |                       525 |                       360 |
| train1.csv.gz | train   | 216001 |        84 | 2020-07-11 00:00:00 | 2020-07-13 12:00:00 |                    0 |                      0 |                      0 |               0 |            0 |                      0 |                              0 |                                     0 |                      0 |                         0 |                         0 |                         0 |
| train2.csv.gz | train   | 226801 |        84 | 2020-07-31 22:00:00 | 2020-08-03 13:00:00 |                    0 |                      0 |                      0 |               0 |            0 |                      0 |                              0 |                                     0 |                      0 |                         0 |                         0 |                         0 |
| train3.csv.gz | train   | 478801 |        84 | 2020-08-04 22:00:00 | 2020-08-10 11:00:00 |                    0 |                      0 |                      0 |               0 |            0 |                      0 |                              0 |                                     0 |                      0 |                         0 |                         0 |                         0 |

## Core Findings

Train attack-labeled rows: `0`.
Numeric nulls/infinities: `0/0`.
Within-file duplicate timestamps: `0`; non-one-second transitions: `0`.
Rows participating in cross-file duplicate timestamps: `86404`. Cross-file overlap is not treated as within-episode duplication but must be respected when constructing train/validation episodes.
Exact train-test telemetry overlap: `43202` training rows; pair details: `[{'train_file': 'train1.csv.gz', 'test_file': 'test2.csv.gz', 'matching_train_rows': 1, 'matching_test_rows': 1}, {'train_file': 'train1.csv.gz', 'test_file': 'test3.csv.gz', 'matching_train_rows': 43201, 'matching_test_rows': 43201}]`.
Global attack rows without a process-specific positive label: `179`. Process labels are treated as partial annotations; the global label remains primary.
Points constant across all train episodes: `['P1_PCV02D', 'P1_PP01AD', 'P1_PP01AR', 'P1_PP01BD', 'P1_PP01BR', 'P1_PP02D', 'P1_PP02R', 'P1_STSP', 'P2_ASD', 'P2_AutoGO', 'P2_Emerg', 'P2_MSD', 'P2_ManualGO', 'P2_OnOff', 'P2_RTR', 'P2_TripEx', 'P2_VTR01', 'P2_VTR02', 'P2_VTR03', 'P2_VTR04', 'P3_LH', 'P3_LL']`; across all test episodes: `['P1_PP01AD', 'P1_PP01AR', 'P1_PP01BD', 'P1_PP01BR', 'P1_PP02D', 'P1_PP02R', 'P1_STSP', 'P2_ASD', 'P2_AutoGO', 'P2_MSD', 'P2_ManualGO', 'P2_RTR', 'P2_TripEx', 'P2_VTR01', 'P2_VTR02', 'P2_VTR03', 'P2_VTR04', 'P3_LH', 'P3_LL']`.

## Attack Intervals

Global attack events: `50`; P1/P2/P3 events: `{'attack_P1': 59, 'attack_P2': 51, 'attack_P3': 8}`.

## Blockers

- Exact train-test telemetry overlap affects 43202 training rows

## Required Next Gate

The 79 points cannot be split into sensor and control inputs from names alone. A manual, source-page-cited role table is required before F0/F1/F0-C modeling. Label columns must never enter model inputs, CSV boundaries must remain episode boundaries, and validation must be a chronological normal-only subset of official train episodes.
All exact train-test telemetry duplicates must be excluded from the training side before scaling or window construction. The exclusion uses only time and telemetry fingerprints, never attack labels.

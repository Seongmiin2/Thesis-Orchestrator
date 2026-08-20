# HAI Train-Test Shift Diagnostic

This diagnostic was triggered after the first full seed produced test-normal FPR 1.0 under a validation-only threshold. It is post-observation and cannot be described as preregistered.

| policy                    | calibration               |   threshold |   rows |   attack_rows |   auroc |   auprc |    fpr |   event_detected_ratio |   censored_delay_mean |
|:--------------------------|:--------------------------|------------:|-------:|--------------:|--------:|--------:|-------:|-----------------------:|----------------------:|
| raw_mean                  | train_validation          |      0.309  | 401855 |          8947 |  0.5077 |  0.2231 | 1      |                   1    |                  0    |
| raw_mean                  | fixed_600s_episode_warmup |    nan      | 399005 |          8947 |  0.5077 |  0.2233 | 0.1596 |                   0.56 |                 88    |
| channel_calibrated_mean   | train_validation          |      1.2949 | 401855 |          8947 |  0.5056 |  0.2279 | 1      |                   1    |                  0    |
| channel_calibrated_mean   | fixed_600s_episode_warmup |    nan      | 399005 |          8947 |  0.5056 |  0.2281 | 0.1593 |                   0.58 |                 87.42 |
| channel_calibrated_median | train_validation          |      1.0542 | 401855 |          8947 |  0.52   |  0.025  | 1      |                   1    |                  0    |
| channel_calibrated_median | fixed_600s_episode_warmup |    nan      | 399005 |          8947 |  0.5208 |  0.0253 | 0.0879 |                   0.22 |                148.12 |

## Largest Channel Shifts

| point      |   validation_median_abs_error |   validation_q995_abs_error |   test_normal_median_abs_error |   test_normal_q995_abs_error |   median_error_ratio |
|:-----------|------------------------------:|----------------------------:|-------------------------------:|-----------------------------:|---------------------:|
| P4_ST_TT01 |                        0.0602 |                      0.412  |                      47173.5   |                    47180.5   |            783032    |
| P4_ST_PT01 |                        0.0639 |                      0.8221 |                      20349.4   |                    20466.8   |            318323    |
| P1_FT01    |                        0.0502 |                      0.3717 |                      10994.1   |                    12526.1   |            218878    |
| P1_FT01Z   |                        0.0695 |                      0.552  |                       6617.37  |                     7092.44  |             95206    |
| P4_ST_PO   |                        0.0697 |                      1.3827 |                       5258.27  |                     7166.89  |             75456.1  |
| P3_LIT01   |                        0.0786 |                      0.5062 |                       4582.56  |                     6466.2   |             58336.4  |
| P1_FT03    |                        0.0458 |                      0.323  |                        431.346 |                      467.749 |              9427.85 |
| P4_ST_FD   |                        0.0617 |                      0.4349 |                        129.422 |                      129.765 |              2096.26 |
| P2_HILout  |                        0.0593 |                      0.3094 |                        101.12  |                      104.062 |              1706.47 |
| P2_SIT01   |                        0.0669 |                      0.269  |                        109.595 |                      111.683 |              1637.15 |

A fixed one-hour episode warm-up is label-free but uses target-domain observations. It is acceptable only as an explicitly adaptive deployment protocol, not as pure zero-shot external validation.

# Architecture Conditional CHUM G3 Report

## Decision

**PASS**: primary mode `loo_sample` has two-architecture consensus on `3` of the four locked primary faults: `[4, 19, 25]`. The pass rule requires at least three distinct primary faults.

## Primary Faults

| architecture   |   fault_id |   channel |   mean_delta_auroc |   mean_delta_auprc |   positive_auroc_seeds |   max_abs_pre_fpr_shift | imputer_reliable   | run_ci_excludes_zero   | stable_material   |   run_delta_auroc_ci_low |   run_delta_auroc_ci_high |
|:---------------|-----------:|----------:|-------------------:|-------------------:|-----------------------:|------------------------:|:-------------------|:-----------------------|:------------------|-------------------------:|--------------------------:|
| tcn            |          4 |        10 |             0.1653 |             0.1211 |                      5 |                  0.0003 | True               | True                   | True              |                   0.1241 |                    0.2054 |
| tcn            |         19 |         7 |             0.1355 |             0.0925 |                      5 |                  0.0008 | True               | True                   | True              |                   0.127  |                    0.1439 |
| tcn            |         25 |         2 |             0.2899 |             0.2358 |                      5 |                  0.0005 | True               | True                   | True              |                   0.285  |                    0.2946 |
| tcn            |         26 |         4 |             0.3652 |             0.3193 |                      5 |                  0.0008 | False              | True                   | False             |                   0.3602 |                    0.3713 |
| transformer    |          4 |        10 |             0.1055 |             0.0882 |                      5 |                  0.0007 | True               | True                   | True              |                   0.0882 |                    0.1199 |
| transformer    |         19 |         7 |             0.1237 |             0.0853 |                      5 |                  0.0008 | True               | True                   | True              |                   0.1135 |                    0.1334 |
| transformer    |         25 |         2 |             0.2637 |             0.1961 |                      5 |                  0.0013 | True               | True                   | True              |                   0.2547 |                    0.2732 |
| transformer    |         26 |         4 |             0.3967 |             0.3345 |                      5 |                  0.0008 | False              | True                   | False             |                   0.391  |                    0.403  |

## Consensus Cells

| mode       |   fault_id |   channel |   stable_architecture_count | stable_architectures   |   raw_stable_architecture_count | raw_stable_architectures   | excluded_after_quality_or_ci   | two_architecture_consensus   | three_architecture_consensus   |
|:-----------|-----------:|----------:|----------------------------:|:-----------------------|--------------------------------:|:---------------------------|:-------------------------------|:-----------------------------|:-------------------------------|
| loo_sample |          4 |        10 |                           2 | tcn,transformer        |                               2 | tcn,transformer            |                                | True                         | False                          |
| loo_sample |         19 |         7 |                           2 | tcn,transformer        |                               2 | tcn,transformer            |                                | True                         | False                          |
| loo_sample |         19 |         8 |                           2 | tcn,transformer        |                               2 | tcn,transformer            |                                | True                         | False                          |
| loo_sample |         25 |         2 |                           2 | tcn,transformer        |                               2 | tcn,transformer            |                                | True                         | False                          |

## Conditional Imputer Reliability Gate

Reliable `loo_sample` channels under held-out distribution and lag checks: `[1, 2, 3, 6, 7, 8, 10, 11]`.
Excluded `loo_sample` channels: `[4, 5, 9]`.
Raw material cells on excluded channels are retained in the CSV for diagnosis but cannot establish conditional-attribution consensus.

## FPR Robustness

| architecture   | mode        |   cells |   fpr_exceptions |
|:---------------|:------------|--------:|-----------------:|
| gru            | conditional |    1540 |                0 |
| gru            | zero        |    1540 |                9 |
| tcn            | conditional |    1540 |                0 |
| tcn            | loo_sample  |    1540 |                0 |
| tcn            | zero        |    1540 |                0 |
| transformer    | conditional |    1540 |                0 |
| transformer    | loo_sample  |    1540 |                0 |
| transformer    | zero        |    1540 |               14 |

## Architecture Rank Agreement

Median `loo_sample` channel Spearman correlation: `0.6621`.

## Statistical Boundary

The run-level intervals use paired hierarchical bootstrap resampling of model seeds and test runs. They quantify stability on this fixed TEP test distribution; they do not create independent datasets or support causal controller claims. Five seeds still cannot attain a two-sided exact seed sign-flip p-value below 0.05.

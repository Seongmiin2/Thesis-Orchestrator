# Integrated Gradients Attribution Baseline

## Scope

Integrated Gradients attributes the instantaneous forecasting-error score from a normal-mean baseline. CHUM measures the change in fault-detection performance after replacing one control channel. Agreement is corroboration; disagreement is expected because the estimands differ.

## Rank Agreement

| architecture   |   fault_id | fault_class             |   spearman |   ig_top_channel |   chum_top_channel | top1_agreement   |   top3_overlap |   material_chum_cells |   mean_abs_convergence_delta |
|:---------------|-----------:|:------------------------|-----------:|-----------------:|-------------------:|:-----------------|---------------:|----------------------:|-----------------------------:|
| tcn            |          1 | negative_or_exploratory |     0.7078 |                3 |                  4 | False            |              1 |                     0 |                       0.0094 |
| tcn            |          4 | primary                 |     0.3425 |                6 |                 10 | False            |              2 |                     1 |                       0.001  |
| tcn            |         11 | negative_or_exploratory |     0.0868 |                6 |                 10 | False            |              2 |                     0 |                       0.0014 |
| tcn            |         19 | primary                 |     0.5799 |                7 |                  7 | True             |              2 |                     2 |                       0.0024 |
| tcn            |         23 | negative_or_exploratory |     0.4977 |                3 |                  3 | True             |              2 |                     1 |                       0.0009 |
| tcn            |         25 | primary                 |     0.5525 |                2 |                  2 | True             |              1 |                     1 |                       0.0025 |
| tcn            |         26 | primary                 |     0.6895 |                4 |                  4 | True             |              2 |                     0 |                       0.003  |
| tcn            |         27 | negative_or_exploratory |    -0.1781 |                6 |                 10 | False            |              0 |                     0 |                       0.0009 |
| transformer    |          1 | negative_or_exploratory |     0.5068 |                3 |                  4 | False            |              1 |                     0 |                       0.0029 |
| transformer    |          4 | primary                 |     0.1324 |               10 |                 10 | True             |              1 |                     1 |                       0.0015 |
| transformer    |         11 | negative_or_exploratory |     0.032  |                6 |                 10 | False            |              1 |                     0 |                       0.0014 |
| transformer    |         19 | primary                 |     0.4247 |                7 |                  7 | True             |              2 |                     2 |                       0.0038 |
| transformer    |         23 | negative_or_exploratory |    -0.0959 |                2 |                  3 | False            |              1 |                     1 |                       0.0013 |
| transformer    |         25 | primary                 |     0.8082 |                2 |                  2 | True             |              2 |                     1 |                       0.0039 |
| transformer    |         26 | primary                 |     0.5342 |                4 |                  4 | True             |              2 |                     0 |                       0.0035 |
| transformer    |         27 | negative_or_exploratory |     0.1142 |                6 |                  4 | False            |              2 |                     0 |                       0.0011 |

## Fault-Class Summary

| fault_class             |   cells |   median_spearman |   mean_spearman |   top1_agreements |   mean_top3_overlap |
|:------------------------|--------:|------------------:|----------------:|------------------:|--------------------:|
| negative_or_exploratory |       8 |            0.1005 |          0.2089 |                 1 |                1.25 |
| primary                 |       8 |            0.5434 |          0.508  |                 7 |                1.75 |

## Interpretation Boundary

Across the `8` architecture-by-primary-fault cells, exact top-channel agreement occurs in `7` and the median top-three overlap is `2.0` of 3.
Integrated Gradients is baseline-dependent and describes local score sensitivity, not conditional channel necessity. CHUM remains the primary utility measure; this analysis checks whether a standard gradient attribution method tells a compatible or materially different story.

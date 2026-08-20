# HAI 21.03 Conditional CHUM Preregistration

Status: `LOCKED_BEFORE_CONDITIONAL_RESULTS`

## Purpose

The HAI 21.03 v2 experiment established that the sensor-plus-control GRU (F1) improves the global attack metrics over both the sensor-only GRU (F0) and the capacity-matched sensor-only GRU (F0-C). This follow-up asks which active control-history channels carry that incremental predictive information. It does not claim causal control effects or root-cause identification.

## Frozen inputs

- Use only the completed F1 checkpoints in `outputs/hai_external_validation_v2` for seeds 42, 43, and 44.
- Preserve the v2 feature order: 29 active sensor targets followed by 28 active control inputs.
- Keep the window length (30), validation-only 99.5th-percentile calibration, three-consecutive-alarm rule, official eTaPR implementation, chronological train/validation split, and five separate test episodes unchanged.
- Never use the invalidated v1 checkpoints.

## Conditional replacement

For every active control channel, fit a normal-training-only leave-one-channel-out ridge predictor of the current control value from current sensors, previous sensors, and all previous controls except the target channel's own previous value. Add a length-20 block-bootstrap draw from centered normal-training residuals. Average anomaly scores over three fixed draws. Zero replacement is retained only as an out-of-distribution diagnostic.

The target channel's own previous value is excluded to prevent a persistence shortcut from making the imputer appear accurate while leaving the observed channel history effectively unchanged.

## Imputer quality gate

A channel is reliable for primary conditional conclusions only when all of the following hold on held-out chronological normal validation data:

- leave-one-channel-out R-squared is at least 0.25;
- sampled standard-deviation ratio is between 0.80 and 1.20;
- absolute sampled mean shift is at most 0.10 validation standard deviations;
- absolute lag-1 autocorrelation error is at most 0.20;
- no more than 1% of sampled values fall outside the normal-training range.

At least five channels must pass. Otherwise the HAI conditional experiment is classified as `BLOCKED_IMPUTER_QUALITY`; the v2 F0/F1/F0-C comparison remains usable, but HAI channel attribution is not claimed.

## Primary analysis

For each seed, perturb one control channel at a time and recalibrate its threshold on equally perturbed validation-normal data. Report global AUROC, AUPRC, eTaF1, false-positive rate, event detection, and censored delay. For every attack event, report the threshold-normalized mean anomaly score, detection, and censored delay.

A primary targeted event-channel cell must:

1. directly name the active control channel in the audited attack-target table;
2. use a quality-gated conditional imputer channel;
3. reduce the threshold-normalized event score by at least 0.05 on average when replaced;
4. have a positive score reduction in all three seeds; and
5. keep the maximum absolute normal false-positive-rate shift within 0.005.

HAI conditional CHUM supports external channel-level replication if at least three targeted event-channel cells pass and conditional replacement has no more false-positive exceptions than zero replacement among reliable channels. All intervals and seed summaries are descriptive because only three seeds are available.

## Claim boundary

Every HAI 21.03 attack directly targets at least one control-history point. Therefore a positive result shows that attacked control histories can provide reproducible incremental predictive information in this HIL dataset. It cannot establish that control history helps when controls are not attacked, nor can it identify causal plant mechanisms. The TEP experiment remains primary for those broader event-specific utility claims.

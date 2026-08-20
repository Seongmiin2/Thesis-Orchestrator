# HAI 21.03 Leakage-Controlled Preparation

## Overall Assessment: PASS_TO_ROLE_MAPPING

Official source commit: `2a814cebc9a66b06c9e5cd545e2d72e65d383737`.

The official gzip files were not modified. Candidate overlap was found with a telemetry-only row fingerprint and every excluded row was then verified by exact equality over `time + 79 SCADA points`. Attack labels were excluded from matching.

Training rows: `921,603` source, `43,202` excluded, `878,401` retained.
Exact train-test match pairs: `43,202`; remaining fingerprint candidates after masking: `0`.
Contiguous retained training segments: `3`. Model windows must be constructed within these segment boundaries.

| file          |   source_rows |   excluded_rows |   retained_rows | keep_mask                               |
|:--------------|--------------:|----------------:|----------------:|:----------------------------------------|
| train1.csv.gz |        216001 |           43202 |          172799 | train_keep_masks\train1.csv.gz.keep.npy |
| train2.csv.gz |        226801 |               0 |          226801 | train_keep_masks\train2.csv.gz.keep.npy |
| train3.csv.gz |        478801 |               0 |          478801 | train_keep_masks\train3.csv.gz.keep.npy |

The masks must be applied before train/validation splitting, scaler fitting, or window construction. Test files remain untouched.

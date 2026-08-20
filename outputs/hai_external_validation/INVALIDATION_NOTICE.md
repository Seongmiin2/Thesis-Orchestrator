# Invalidated HAI Run

This directory is retained only as a preprocessing-bug audit trail. Training arrays used source CSV column order while test arrays used the role-manifest order, which misaligned feature identities and scaler parameters across splits.

No metric or report in this directory is valid evidence. The corrected deterministic run is `outputs/hai_external_validation_v2`.

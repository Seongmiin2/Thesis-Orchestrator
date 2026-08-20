# HAI 21.03 External Validation Audit

## Decision

**SUITABLE WITH ROLE-MAPPING AND ATTACK-TARGET CONTROLS.** HAI 21.03 is approved as the first external dataset for CHUM replication. It must be treated as a hardware-in-the-loop cyber-physical attack dataset, not as a second process-fault dataset.

## Acquired source

- Official repository: `external/hai`, branch `master`.
- Selected release: `hai-21.03`.
- Files: 3 gzip-compressed train CSVs and 5 test CSVs; 178.6 MiB compressed.
- Schema: 84 columns = time + 79 recorded process/control points + global/process attack labels (`attack`, `attack_P1`, `attack_P2`, `attack_P3`).
- License/provenance: official HAI repository and technical manual; repository states CC BY-SA 4.0.

## Why it fits CHUM

The official manual explicitly models feedback-control attacks using setpoints (SP), process variables (PV), controller/control outputs (CV/CO), and internal controller parameters. Recorded points include:

- setpoints and demands, e.g. `P1_B2016`, `P1_B3004`, `P2_ASD`;
- valve/pump commands and controller outputs, e.g. `P1_FCV01D`, `P1_LCV01D`, `P1_PCV01D`, `P2_CO_rpm`, `P3_LCP01D`, `P3_LCV01D`;
- observed valve/running states, e.g. `*Z`, `*R` fields;
- physical measurements, e.g. pressure, flow, level, temperature, speed, and vibration points.

This is richer than the current Reinartz TEP representation because command, observed state, setpoint, and process measurement roles can be separated from official documentation.

## Required role mapping

Before training, create a versioned table with one row per point:

`point, process, role={PV,SP,CO,ACTUATOR_STATE,MODE,MODEL_SIGNAL,OTHER}, unit, description, source_page, include_as_sensor, include_as_control`

No point may be classified solely by suffix heuristics when the manual provides a description. Ambiguous P4 HIL/model signals remain `OTHER` until verified.

## Leakage and construct controls

- Exclude all four attack-label columns from input.
- Preserve each CSV as a separate continuous episode; never create windows across file boundaries.
- Train only on official train files after verifying that all attack labels are zero.
- Split normal training time chronologically into train and validation; thresholds use validation only.
- Report global and process-specific labels separately.
- Separate attacks targeting SP/PV/CO. Directly attacked control channels can make detection trivially easy and must be a prespecified subgroup, not pooled with indirect effects.
- Compare sensor-only, sensor+control, and capacity-matched sensor-only under identical episode splits.
- Use the official enhanced time-series aware precision/recall (eTaPR) in addition to event detection rate, delay, and pre-attack FPR.

## External replication question

Does event-level control-history utility vary systematically between attacks that directly target control-loop variables and attacks whose primary targets are process measurements or other components, while remaining distinguishable from model-capacity and false-alarm effects?

## Minimal execution

1. Build and manually audit the 79-point role table.
2. Verify train-label purity, timestamp continuity, missingness, constant channels, and attack intervals.
3. Train GRU sensor-only/sensor+control/capacity-control at three seeds.
4. If heterogeneous event-level utility exists, run conditional CHUM on event groups and one non-GRU architecture.
5. Keep HAI as external replication; do not retune the TEP hypotheses using HAI test results.

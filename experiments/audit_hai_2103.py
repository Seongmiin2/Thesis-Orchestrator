from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LABEL_COLUMNS = ["attack", "attack_P1", "attack_P2", "attack_P3"]


def git_revision(repository: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()


def contiguous_intervals(file_name: str, frame: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for label in LABEL_COLUMNS:
        values = frame[label].to_numpy(dtype=np.int8)
        padded = np.r_[0, values, 0]
        starts = np.flatnonzero(np.diff(padded) == 1)
        ends = np.flatnonzero(np.diff(padded) == -1) - 1
        for event_index, (start, end) in enumerate(zip(starts, ends), start=1):
            rows.append(
                {
                    "file": file_name,
                    "label": label,
                    "event_index": event_index,
                    "start_row": int(start),
                    "end_row": int(end),
                    "start_time": frame.time.iloc[start],
                    "end_time": frame.time.iloc[end],
                    "duration_seconds": int(end - start + 1),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="../HAI/hai-21.03")
    parser.add_argument("--output", default="outputs/hai_2103_audit")
    parser.add_argument("--chunksize", type=int, default=100_000)
    args = parser.parse_args()
    source = (ROOT / args.source).resolve()
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    files = sorted(source.glob("*.csv.gz"))
    if [path.name for path in files] != [
        "test1.csv.gz",
        "test2.csv.gz",
        "test3.csv.gz",
        "test4.csv.gz",
        "test5.csv.gz",
        "train1.csv.gz",
        "train2.csv.gz",
        "train3.csv.gz",
    ]:
        raise ValueError("Expected three train and five test files for HAI 21.03")

    canonical_columns: list[str] | None = None
    file_rows: list[dict] = []
    interval_rows: list[dict] = []
    column_stats: dict[tuple[str, str], dict] = {}
    all_timestamps: list[pd.Series] = []
    global_point_min: dict[tuple[str, str], float] = {}
    global_point_max: dict[tuple[str, str], float] = {}
    file_fingerprints: dict[str, np.ndarray] = {}

    for path in files:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        if canonical_columns is None:
            canonical_columns = header
        if header != canonical_columns:
            raise ValueError(f"Schema drift in {path.name}")
        point_columns = header[1:-4]
        if len(point_columns) != 79:
            raise ValueError(f"Expected 79 SCADA points, found {len(point_columns)}")

        rows = 0
        nulls = 0
        infinities = 0
        invalid_label_values = 0
        process_outside_global = 0
        global_without_process = 0
        duplicate_timestamps = 0
        non_one_second_steps = 0
        invalid_timestamps = 0
        first_time: pd.Timestamp | None = None
        last_time: pd.Timestamp | None = None
        previous_time: pd.Timestamp | None = None
        label_sums = {label: 0 for label in LABEL_COLUMNS}
        per_file_min = {column: np.inf for column in point_columns}
        per_file_max = {column: -np.inf for column in point_columns}
        label_frames: list[pd.DataFrame] = []
        fingerprint_chunks: list[np.ndarray] = []

        for chunk in pd.read_csv(path, chunksize=args.chunksize):
            rows += len(chunk)
            timestamps = pd.to_datetime(chunk.time, errors="coerce")
            invalid_timestamps += int(timestamps.isna().sum())
            valid_timestamps = timestamps.dropna()
            if len(valid_timestamps):
                if first_time is None:
                    first_time = valid_timestamps.iloc[0]
                if previous_time is not None:
                    boundary_delta = (valid_timestamps.iloc[0] - previous_time).total_seconds()
                    duplicate_timestamps += int(boundary_delta == 0)
                    non_one_second_steps += int(boundary_delta != 1)
                deltas = valid_timestamps.diff().dt.total_seconds().dropna()
                duplicate_timestamps += int((deltas == 0).sum())
                non_one_second_steps += int((deltas != 1).sum())
                previous_time = valid_timestamps.iloc[-1]
                last_time = valid_timestamps.iloc[-1]
                all_timestamps.append(valid_timestamps)

            numeric = chunk[point_columns + LABEL_COLUMNS]
            nulls += int(numeric.isna().to_numpy().sum())
            values = numeric.to_numpy(dtype=np.float64, copy=False)
            infinities += int(np.isinf(values).sum())
            for column in point_columns:
                column_values = chunk[column].to_numpy(dtype=np.float64)
                finite = column_values[np.isfinite(column_values)]
                if len(finite):
                    per_file_min[column] = min(per_file_min[column], float(finite.min()))
                    per_file_max[column] = max(per_file_max[column], float(finite.max()))
            for label in LABEL_COLUMNS:
                label_values = chunk[label]
                invalid_label_values += int((~label_values.isin([0, 1])).sum())
                label_sums[label] += int(label_values.sum())
            process_or = chunk[["attack_P1", "attack_P2", "attack_P3"]].max(axis=1)
            process_outside_global += int(((process_or == 1) & (chunk.attack == 0)).sum())
            global_without_process += int(((process_or == 0) & (chunk.attack == 1)).sum())
            point_hash = pd.util.hash_pandas_object(
                chunk[point_columns].astype(np.float64), index=False
            ).to_numpy(dtype=np.uint64)
            time_hash = pd.util.hash_pandas_object(
                chunk.time.astype(str), index=False
            ).to_numpy(dtype=np.uint64)
            fingerprint_chunks.append(point_hash ^ np.left_shift(time_hash, np.uint64(1)))
            label_frames.append(chunk[["time", *LABEL_COLUMNS]])

        label_frame = pd.concat(label_frames, ignore_index=True)
        file_fingerprints[path.name] = np.concatenate(fingerprint_chunks)
        interval_rows.extend(contiguous_intervals(path.name, label_frame))
        split = "train" if path.name.startswith("train") else "test"
        for column in point_columns:
            column_stats[(path.name, column)] = {
                "file": path.name,
                "split": split,
                "point": column,
                "minimum": per_file_min[column],
                "maximum": per_file_max[column],
                "constant_in_file": per_file_min[column] == per_file_max[column],
            }
            key = (split, column)
            global_point_min[key] = min(global_point_min.get(key, np.inf), per_file_min[column])
            global_point_max[key] = max(global_point_max.get(key, -np.inf), per_file_max[column])

        file_rows.append(
            {
                "file": path.name,
                "split": split,
                "rows": rows,
                "columns": len(header),
                "first_time": first_time,
                "last_time": last_time,
                "invalid_timestamps": invalid_timestamps,
                "duplicate_timestamps": duplicate_timestamps,
                "non_one_second_steps": non_one_second_steps,
                "numeric_nulls": nulls,
                "infinities": infinities,
                "invalid_label_values": invalid_label_values,
                "process_label_outside_global": process_outside_global,
                "global_attack_without_process_label": global_without_process,
                **{f"{label}_positive_rows": label_sums[label] for label in LABEL_COLUMNS},
            }
        )

    file_profile = pd.DataFrame(file_rows)
    intervals = pd.DataFrame(interval_rows)
    point_profile = pd.DataFrame(column_stats.values())
    combined_profile = pd.DataFrame(
        [
            {
                "split": split,
                "point": point,
                "minimum": minimum,
                "maximum": global_point_max[(split, point)],
                "constant": minimum == global_point_max[(split, point)],
            }
            for (split, point), minimum in global_point_min.items()
        ]
    ).sort_values(["split", "point"])

    timestamps = pd.concat(all_timestamps, ignore_index=True)
    cross_file_duplicate_timestamps = int(timestamps.duplicated(keep=False).sum())
    overlap_rows: list[dict] = []
    train_names = sorted(name for name in file_fingerprints if name.startswith("train"))
    test_names = sorted(name for name in file_fingerprints if name.startswith("test"))
    for train_name in train_names:
        for test_name in test_names:
            train_matches = int(
                np.isin(file_fingerprints[train_name], file_fingerprints[test_name]).sum()
            )
            test_matches = int(
                np.isin(file_fingerprints[test_name], file_fingerprints[train_name]).sum()
            )
            if train_matches or test_matches:
                overlap_rows.append(
                    {
                        "train_file": train_name,
                        "test_file": test_name,
                        "matching_train_rows": train_matches,
                        "matching_test_rows": test_matches,
                    }
                )
    overlap = pd.DataFrame(overlap_rows)
    exact_train_test_overlap_rows = int(overlap.matching_train_rows.sum()) if len(overlap) else 0
    train_attack_rows = int(
        file_profile.loc[file_profile.split == "train", "attack_positive_rows"].sum()
    )
    total_rows = int(file_profile.rows.sum())
    test_attack_rows = int(
        file_profile.loc[file_profile.split == "test", "attack_positive_rows"].sum()
    )
    train_constants = combined_profile.loc[
        (combined_profile.split == "train") & combined_profile.constant, "point"
    ].tolist()
    test_constants = combined_profile.loc[
        (combined_profile.split == "test") & combined_profile.constant, "point"
    ].tolist()

    file_profile.to_csv(output / "HAI_FILE_PROFILE.csv", index=False)
    intervals.to_csv(output / "HAI_ATTACK_INTERVALS.csv", index=False)
    point_profile.to_csv(output / "HAI_POINT_FILE_PROFILE.csv", index=False)
    combined_profile.to_csv(output / "HAI_POINT_SPLIT_PROFILE.csv", index=False)
    overlap.to_csv(output / "HAI_EXACT_TRAIN_TEST_OVERLAP.csv", index=False)

    blockers: list[str] = []
    if train_attack_rows != 0:
        blockers.append(f"Official train files contain {train_attack_rows} attack-labeled rows")
    if int(file_profile.numeric_nulls.sum()) > 0:
        blockers.append("Numeric null values are present")
    if int(file_profile.infinities.sum()) > 0:
        blockers.append("Infinite numeric values are present")
    if int(file_profile.invalid_label_values.sum()) > 0:
        blockers.append("Attack labels contain values outside {0, 1}")
    if int(file_profile.process_label_outside_global.sum()) > 0:
        blockers.append("A process attack label is positive while the global attack label is zero")
    if int(file_profile.duplicate_timestamps.sum()) > 0:
        blockers.append("Duplicate timestamps occur within an episode")
    if exact_train_test_overlap_rows > 0:
        blockers.append(
            f"Exact train-test telemetry overlap affects {exact_train_test_overlap_rows} training rows"
        )
    assessment = "PASS_TO_ROLE_MAPPING" if not blockers else "BLOCKED"
    source_revision = git_revision(source.parent)
    report = [
        "# HAI 21.03 Data Quality Audit",
        "",
        f"## Overall Assessment: {assessment}",
        "",
        f"Source commit: `{source_revision}`. Intended grain: one SCADA observation per second within each of eight continuous CSV episodes.",
        "",
        "## Dataset and Grain",
        "",
        f"Files: `8`; rows: `{total_rows:,}`; schema: `time + 79 points + 4 labels`; test attack rows: `{test_attack_rows:,}`.",
        "",
        file_profile.to_markdown(index=False),
        "",
        "## Core Findings",
        "",
        f"Train attack-labeled rows: `{train_attack_rows}`.",
        f"Numeric nulls/infinities: `{int(file_profile.numeric_nulls.sum())}/{int(file_profile.infinities.sum())}`.",
        f"Within-file duplicate timestamps: `{int(file_profile.duplicate_timestamps.sum())}`; non-one-second transitions: `{int(file_profile.non_one_second_steps.sum())}`.",
        f"Rows participating in cross-file duplicate timestamps: `{cross_file_duplicate_timestamps}`. Cross-file overlap is not treated as within-episode duplication but must be respected when constructing train/validation episodes.",
        f"Exact train-test telemetry overlap: `{exact_train_test_overlap_rows}` training rows; pair details: `{overlap.to_dict('records')}`.",
        f"Global attack rows without a process-specific positive label: `{int(file_profile.global_attack_without_process_label.sum())}`. Process labels are treated as partial annotations; the global label remains primary.",
        f"Points constant across all train episodes: `{train_constants}`; across all test episodes: `{test_constants}`.",
        "",
        "## Attack Intervals",
        "",
        f"Global attack events: `{len(intervals[intervals.label == 'attack'])}`; P1/P2/P3 events: `{intervals[intervals.label != 'attack'].groupby('label').size().to_dict()}`.",
        "",
        "## Blockers",
        "",
        "\n".join(f"- {item}" for item in blockers) if blockers else "No raw-data blocker was found.",
        "",
        "## Required Next Gate",
        "",
        "The 79 points cannot be split into sensor and control inputs from names alone. A manual, source-page-cited role table is required before F0/F1/F0-C modeling. Label columns must never enter model inputs, CSV boundaries must remain episode boundaries, and validation must be a chronological normal-only subset of official train episodes.",
        "All exact train-test telemetry duplicates must be excluded from the training side before scaling or window construction. The exclusion uses only time and telemetry fingerprints, never attack labels.",
    ]
    (output / "HAI_2103_DATA_QUALITY_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    manifest = {
        "status": "COMPLETE",
        "assessment": assessment,
        "source_revision": source_revision,
        "files": len(files),
        "rows": total_rows,
        "points": 79,
        "train_attack_rows": train_attack_rows,
        "test_attack_rows": test_attack_rows,
        "global_attack_events": int(len(intervals[intervals.label == "attack"])),
        "cross_file_duplicate_timestamp_rows": cross_file_duplicate_timestamps,
        "exact_train_test_overlap_rows": exact_train_test_overlap_rows,
        "global_attack_without_process_label_rows": int(
            file_profile.global_attack_without_process_label.sum()
        ),
        "train_constant_points": train_constants,
        "test_constant_points": test_constants,
        "blockers": blockers,
    }
    (output / "HAI_2103_AUDIT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

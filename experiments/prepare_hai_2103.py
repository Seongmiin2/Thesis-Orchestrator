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


def row_fingerprints(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    normalized = frame[columns].copy()
    normalized[columns[0]] = normalized[columns[0]].astype(str)
    normalized[columns[1:]] = normalized[columns[1:]].astype(np.float64)
    return pd.util.hash_pandas_object(
        normalized, index=False, categorize=False
    ).to_numpy(dtype=np.uint64)


def read_fingerprints(
    path: Path, telemetry_columns: list[str], chunksize: int
) -> tuple[np.ndarray, int]:
    chunks: list[np.ndarray] = []
    rows = 0
    for chunk in pd.read_csv(path, usecols=telemetry_columns, chunksize=chunksize):
        chunks.append(row_fingerprints(chunk, telemetry_columns))
        rows += len(chunk)
    return np.concatenate(chunks), rows


def contiguous_keep_segments(file_name: str, keep: np.ndarray) -> list[dict]:
    padded = np.r_[False, keep, False]
    starts = np.flatnonzero(np.diff(padded.astype(np.int8)) == 1)
    ends = np.flatnonzero(np.diff(padded.astype(np.int8)) == -1) - 1
    return [
        {
            "file": file_name,
            "segment_index": index,
            "start_row": int(start),
            "end_row": int(end),
            "rows": int(end - start + 1),
        }
        for index, (start, end) in enumerate(zip(starts, ends), start=1)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="../HAI/hai-21.03")
    parser.add_argument("--output", default="outputs/hai_2103_prepared")
    parser.add_argument("--chunksize", type=int, default=100_000)
    args = parser.parse_args()

    source = (ROOT / args.source).resolve()
    output = (ROOT / args.output).resolve()
    mask_dir = output / "train_keep_masks"
    output.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    train_files = sorted(source.glob("train*.csv.gz"))
    test_files = sorted(source.glob("test*.csv.gz"))
    if len(train_files) != 3 or len(test_files) != 5:
        raise ValueError("Expected three train and five test files for HAI 21.03")

    header = pd.read_csv(test_files[0], nrows=0).columns.tolist()
    telemetry_columns = header[:-len(LABEL_COLUMNS)]
    point_columns = telemetry_columns[1:]
    if header[-len(LABEL_COLUMNS) :] != LABEL_COLUMNS or len(point_columns) != 79:
        raise ValueError("Unexpected HAI 21.03 schema")
    for path in [*train_files, *test_files]:
        if pd.read_csv(path, nrows=0).columns.tolist() != header:
            raise ValueError(f"Schema drift in {path.name}")

    test_hashes: dict[str, np.ndarray] = {}
    all_test_hashes: list[np.ndarray] = []
    test_rows: dict[str, int] = {}
    for path in test_files:
        hashes, rows = read_fingerprints(path, telemetry_columns, args.chunksize)
        test_hashes[path.name] = hashes
        all_test_hashes.append(hashes)
        test_rows[path.name] = rows
    candidate_hashes = np.unique(np.concatenate(all_test_hashes))

    train_hashes: dict[str, np.ndarray] = {}
    candidate_train_rows: list[pd.DataFrame] = []
    train_rows: dict[str, int] = {}
    for path in train_files:
        hashes, rows = read_fingerprints(path, telemetry_columns, args.chunksize)
        train_hashes[path.name] = hashes
        train_rows[path.name] = rows
        candidate_positions = np.flatnonzero(np.isin(hashes, candidate_hashes))
        if len(candidate_positions):
            candidate = pd.read_csv(path, usecols=telemetry_columns).iloc[
                candidate_positions
            ].copy()
            candidate.insert(0, "train_row", candidate_positions)
            candidate.insert(0, "train_file", path.name)
            candidate_train_rows.append(candidate)

    train_candidate_hashes = np.unique(
        np.concatenate(
            [
                train_hashes[path.name][
                    np.isin(train_hashes[path.name], candidate_hashes)
                ]
                for path in train_files
            ]
        )
    )
    candidate_test_rows: list[pd.DataFrame] = []
    for path in test_files:
        candidate_positions = np.flatnonzero(
            np.isin(test_hashes[path.name], train_candidate_hashes)
        )
        if len(candidate_positions):
            candidate = pd.read_csv(path, usecols=telemetry_columns).iloc[
                candidate_positions
            ].copy()
            candidate.insert(0, "test_row", candidate_positions)
            candidate.insert(0, "test_file", path.name)
            candidate_test_rows.append(candidate)

    train_candidates = pd.concat(candidate_train_rows, ignore_index=True)
    test_candidates = pd.concat(candidate_test_rows, ignore_index=True)
    exact_matches = train_candidates.merge(
        test_candidates,
        on=telemetry_columns,
        how="inner",
        validate="many_to_many",
    )[["train_file", "train_row", "test_file", "test_row", "time"]]
    exact_matches = exact_matches.sort_values(
        ["train_file", "train_row", "test_file", "test_row"]
    ).reset_index(drop=True)
    excluded = exact_matches[["train_file", "train_row"]].drop_duplicates()

    segment_rows: list[dict] = []
    mask_rows: list[dict] = []
    retained_hashes: list[np.ndarray] = []
    for path in train_files:
        keep = np.ones(train_rows[path.name], dtype=bool)
        file_excluded = excluded.loc[
            excluded.train_file == path.name, "train_row"
        ].to_numpy(dtype=np.int64)
        keep[file_excluded] = False
        np.save(mask_dir / f"{path.name}.keep.npy", keep, allow_pickle=False)
        retained_hashes.append(train_hashes[path.name][keep])
        segment_rows.extend(contiguous_keep_segments(path.name, keep))
        mask_rows.append(
            {
                "file": path.name,
                "source_rows": int(len(keep)),
                "excluded_rows": int((~keep).sum()),
                "retained_rows": int(keep.sum()),
                "keep_mask": str(
                    (mask_dir / f"{path.name}.keep.npy").relative_to(output)
                ),
            }
        )

    # All first-pass hash candidates were verified by an exact 80-column merge.
    # A remaining hash overlap would therefore indicate a preparation bug.
    remaining_hash_overlap = int(
        np.isin(np.concatenate(retained_hashes), np.concatenate(all_test_hashes)).sum()
    )
    if remaining_hash_overlap != 0:
        raise RuntimeError(
            f"Prepared train data still has {remaining_hash_overlap} test hash matches"
        )

    segments = pd.DataFrame(segment_rows)
    masks = pd.DataFrame(mask_rows)
    exact_matches.to_csv(output / "HAI_EXACT_EXCLUSIONS.csv", index=False)
    segments.to_csv(output / "HAI_TRAIN_SEGMENTS.csv", index=False)
    masks.to_csv(output / "HAI_TRAIN_MASKS.csv", index=False)

    source_rows = int(sum(train_rows.values()))
    excluded_rows = int(len(excluded))
    retained_rows = source_rows - excluded_rows
    manifest = {
        "status": "COMPLETE",
        "assessment": "PASS_TO_ROLE_MAPPING",
        "source_revision": git_revision(source.parent),
        "source": str(source),
        "telemetry_columns": len(telemetry_columns),
        "point_columns": len(point_columns),
        "label_columns_excluded_from_matching": LABEL_COLUMNS,
        "train_source_rows": source_rows,
        "train_excluded_rows": excluded_rows,
        "train_retained_rows": retained_rows,
        "test_rows": int(sum(test_rows.values())),
        "exact_match_pairs": int(len(exact_matches)),
        "remaining_train_test_hash_candidates": remaining_hash_overlap,
        "train_files": masks.to_dict("records"),
        "segments": int(len(segments)),
    }
    (output / "HAI_2103_PREPARATION_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    report = [
        "# HAI 21.03 Leakage-Controlled Preparation",
        "",
        "## Overall Assessment: PASS_TO_ROLE_MAPPING",
        "",
        f"Official source commit: `{manifest['source_revision']}`.",
        "",
        "The official gzip files were not modified. Candidate overlap was found with a telemetry-only row fingerprint and every excluded row was then verified by exact equality over `time + 79 SCADA points`. Attack labels were excluded from matching.",
        "",
        f"Training rows: `{source_rows:,}` source, `{excluded_rows:,}` excluded, `{retained_rows:,}` retained.",
        f"Exact train-test match pairs: `{len(exact_matches):,}`; remaining fingerprint candidates after masking: `{remaining_hash_overlap}`.",
        f"Contiguous retained training segments: `{len(segments)}`. Model windows must be constructed within these segment boundaries.",
        "",
        masks.to_markdown(index=False),
        "",
        "The masks must be applied before train/validation splitting, scaler fitting, or window construction. Test files remain untouched.",
    ]
    (output / "HAI_2103_PREPARATION_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

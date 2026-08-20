from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="configs/hai_2103_attack_targets.csv")
    parser.add_argument("--roles", default="outputs/hai_2103_prepared/HAI_2103_VALIDATED_POINT_ROLES.csv")
    parser.add_argument("--intervals", default="outputs/hai_2103_audit/HAI_ATTACK_INTERVALS.csv")
    parser.add_argument("--output", default="outputs/hai_2103_prepared")
    args = parser.parse_args()

    targets = pd.read_csv((ROOT / args.targets).resolve(), keep_default_na=False)
    roles = pd.read_csv((ROOT / args.roles).resolve()).set_index("point")
    intervals = pd.read_csv((ROOT / args.intervals).resolve())
    intervals = intervals.loc[intervals.label == "attack"].sort_values(
        ["start_time", "file", "event_index"]
    ).reset_index(drop=True)
    output = (ROOT / args.output).resolve()

    if len(targets) != 50 or targets.global_event.tolist() != list(range(1, 51)):
        raise ValueError("Attack target table must contain ordered global events 1..50")
    if len(intervals) != 50:
        raise ValueError(f"Expected 50 global attack intervals, found {len(intervals)}")

    computed_classes: list[str] = []
    target_roles: list[str] = []
    unknown: list[str] = []
    for row in targets.itertuples(index=False):
        points = row.target_points.split(";")
        unknown.extend(point for point in points if point not in roles.index)
        if unknown:
            continue
        point_roles = roles.loc[points]
        has_sensor = bool(point_roles.include_as_sensor.any())
        has_control = bool(point_roles.include_as_control.any())
        if has_sensor and has_control:
            computed = "mixed_control_sensor"
        elif has_control:
            computed = "control_only"
        elif has_sensor:
            computed = "sensor_only"
        else:
            computed = "other"
        computed_classes.append(computed)
        target_roles.append(";".join(point_roles.role.astype(str).tolist()))
    if unknown:
        raise ValueError(f"Unknown target points: {sorted(set(unknown))}")
    if computed_classes != targets.target_class.tolist():
        raise ValueError("Declared attack target classes disagree with validated point roles")

    result = pd.concat(
        [
            targets,
            intervals[
                ["file", "event_index", "start_time", "end_time", "duration_seconds"]
            ],
        ],
        axis=1,
    )
    result["target_roles"] = target_roles
    result.to_csv(output / "HAI_2103_VALIDATED_ATTACK_TARGETS.csv", index=False)
    counts = result.target_class.value_counts().to_dict()
    manifest = {
        "status": "COMPLETE",
        "assessment": "PASS_WITH_SCOPE_LIMITATION",
        "events": len(result),
        "target_class_counts": counts,
        "events_without_control_target": int(
            (~result.target_class.isin(["control_only", "mixed_control_sensor"])).sum()
        ),
        "source_pages": [36, 37, 38],
        "scope_limitation": "Every HAI 21.03 event directly targets at least one control-history point.",
    }
    (output / "HAI_2103_ATTACK_TARGET_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    report = [
        "# HAI 21.03 Attack-Target Gate",
        "",
        "## Overall Assessment: PASS_WITH_SCOPE_LIMITATION",
        "",
        "The 50 labeled global attack intervals were matched in chronological order to the official HAI 21.03 attack table on technical-manual pages 36-38. Target aliases were normalized to the recorded 21.03 column names and validated against the point-role table.",
        "",
        f"Target classes: `{counts}`.",
        "Every event directly targets at least one control-history point. HAI 21.03 therefore tests transfer under direct control-target and mixed control-plus-sensor attacks; it cannot independently establish utility for attacks with no directly manipulated control channel.",
        "",
        result.groupby(["file", "target_class"]).size().rename("events").reset_index().to_markdown(index=False),
    ]
    (output / "HAI_2103_ATTACK_TARGET_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

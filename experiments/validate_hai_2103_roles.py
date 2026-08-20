from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LABEL_COLUMNS = ["attack", "attack_P1", "attack_P2", "attack_P3"]
ALLOWED_ROLES = {"PV", "SP", "CO", "ACTUATOR_STATE", "MODE", "MODEL_SIGNAL", "OTHER"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="../HAI/hai-21.03")
    parser.add_argument("--roles", default="configs/hai_2103_point_roles.csv")
    parser.add_argument("--profile", default="outputs/hai_2103_audit/HAI_POINT_SPLIT_PROFILE.csv")
    parser.add_argument("--output", default="outputs/hai_2103_prepared")
    args = parser.parse_args()

    source = (ROOT / args.source).resolve()
    roles_path = (ROOT / args.roles).resolve()
    profile_path = (ROOT / args.profile).resolve()
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    header = pd.read_csv(source / "test1.csv.gz", nrows=0).columns.tolist()
    points = header[1:-len(LABEL_COLUMNS)]
    roles = pd.read_csv(roles_path)
    for column in ["include_as_sensor", "include_as_control"]:
        roles[column] = roles[column].map({True: True, False: False, "true": True, "false": False})

    errors: list[str] = []
    if roles.point.duplicated().any():
        errors.append(f"Duplicate role rows: {roles.loc[roles.point.duplicated(), 'point'].tolist()}")
    missing = sorted(set(points) - set(roles.point))
    extra = sorted(set(roles.point) - set(points))
    if missing:
        errors.append(f"Missing points: {missing}")
    if extra:
        errors.append(f"Extra points: {extra}")
    invalid_roles = sorted(set(roles.role) - ALLOWED_ROLES)
    if invalid_roles:
        errors.append(f"Invalid roles: {invalid_roles}")
    if roles[["include_as_sensor", "include_as_control"]].isna().any().any():
        errors.append("Invalid sensor/control inclusion values")
    overlap = roles.include_as_sensor & roles.include_as_control
    if overlap.any():
        errors.append(f"Points assigned to both groups: {roles.loc[overlap, 'point'].tolist()}")
    omitted = ~(roles.include_as_sensor | roles.include_as_control)
    if omitted.any():
        errors.append(f"Points omitted from both groups: {roles.loc[omitted, 'point'].tolist()}")
    if errors:
        raise ValueError("; ".join(errors))

    profile = pd.read_csv(profile_path)
    train_profile = profile.loc[profile.split == "train", ["point", "constant"]]
    roles = roles.merge(train_profile, on="point", how="left", validate="one_to_one")
    if roles.constant.isna().any():
        raise ValueError("Train profile is incomplete")
    roles["active_in_train"] = ~roles.constant

    sensors = roles.loc[roles.include_as_sensor, "point"].tolist()
    controls = roles.loc[roles.include_as_control, "point"].tolist()
    active_sensors = roles.loc[roles.include_as_sensor & roles.active_in_train, "point"].tolist()
    active_controls = roles.loc[roles.include_as_control & roles.active_in_train, "point"].tolist()
    manifest = {
        "status": "COMPLETE",
        "assessment": "PASS_TO_MODELING",
        "roles_source": str(roles_path),
        "technical_manual_pages": [12, 13, 14],
        "points": len(roles),
        "sensor_points": len(sensors),
        "control_points": len(controls),
        "active_sensor_points": len(active_sensors),
        "active_control_points": len(active_controls),
        "constant_sensor_points": sorted(set(sensors) - set(active_sensors)),
        "constant_control_points": sorted(set(controls) - set(active_controls)),
        "mapping_confidence": roles.mapping_confidence.value_counts().to_dict(),
        "f0_active_columns": active_sensors,
        "f1_active_columns": [*active_sensors, *active_controls],
    }
    (output / "HAI_2103_ROLE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    roles.to_csv(output / "HAI_2103_VALIDATED_POINT_ROLES.csv", index=False)
    report = [
        "# HAI 21.03 Point-Role Gate",
        "",
        "## Overall Assessment: PASS_TO_MODELING",
        "",
        "Roles were assigned manually from the official HAI technical manual, pages 12-14. Version-specific aliases whose semantics are not printed verbatim in the latest table are marked `medium` confidence.",
        "",
        f"All `{len(roles)}` SCADA points are assigned exactly once: `{len(sensors)}` sensor/model points and `{len(controls)}` control-history points.",
        f"After removing train-constant points, F0 has `{len(active_sensors)}` inputs and F1 has `{len(active_sensors) + len(active_controls)}` inputs (`{len(active_controls)}` added control-history channels).",
        f"Mapping confidence: `{manifest['mapping_confidence']}`.",
        "",
        "F0 includes physical measurements and HIL model signals. F1 adds setpoints, controller outputs, commands, actuator states, and operating modes. Constant training channels are excluded from both models before scaling.",
        "",
        roles.groupby(["process", "role"]).size().rename("points").reset_index().to_markdown(index=False),
    ]
    (output / "HAI_2103_ROLE_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

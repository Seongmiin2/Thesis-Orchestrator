from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/professor_report_v2"
TITLE = "제어 이력의 조건부 정보 효용: 석사논문 연구 결과"


def records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records"))


def source(source_id: str, label: str, path: str, sql: str, description: str) -> dict:
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": sql,
            "description": description,
            "tables_used": [path],
        },
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone(timedelta(hours=9))).replace(
        microsecond=0
    ).isoformat()

    g3_summary = pd.read_csv(
        ROOT / "outputs/architecture_chum_g3/G3_CELL_SUMMARY.csv"
    )
    g3_consensus = pd.read_csv(
        ROOT / "outputs/architecture_chum_g3/G3_ARCHITECTURE_CONSENSUS.csv"
    )
    accepted = g3_consensus.loc[
        (g3_consensus["mode"] == "loo_sample")
        & g3_consensus.fault_id.isin([4, 19, 25, 26])
        & g3_consensus.two_architecture_consensus.astype(bool),
        ["mode", "fault_id", "channel"],
    ]
    g3 = g3_summary.merge(
        accepted,
        on=["mode", "fault_id", "channel"],
        how="inner",
        validate="many_to_one",
    )
    g3 = g3.loc[g3.architecture.isin(["tcn", "transformer"])].copy()
    g3["fault_channel"] = g3.apply(
        lambda row: f"Fault {int(row.fault_id)} / XMV{int(row.channel)}", axis=1
    )
    g3["architecture_label"] = g3.architecture.map(
        {"tcn": "TCN", "transformer": "Transformer"}
    )
    g3 = g3[
        [
            "fault_channel",
            "architecture_label",
            "mean_delta_auroc",
            "mean_delta_auprc",
            "positive_auroc_seeds",
            "max_abs_pre_fpr_shift",
            "run_delta_auroc_ci_low",
            "run_delta_auroc_ci_high",
        ]
    ].sort_values(["fault_channel", "architecture_label"])
    consensus_faults = accepted.fault_id.nunique()

    ig = pd.read_csv(
        ROOT / "outputs/integrated_gradients_baseline/IG_CHUM_CLASS_SUMMARY.csv"
    )
    ig["fault_group"] = ig.fault_class.map(
        {
            "primary": "Locked primary",
            "negative_or_exploratory": "Negative / exploratory",
        }
    )
    ig["top1_rate"] = ig.top1_agreements / ig.cells
    ig = ig[
        [
            "fault_group",
            "cells",
            "median_spearman",
            "top1_agreements",
            "top1_rate",
            "mean_top3_overlap",
        ]
    ]
    ig_primary = ig.loc[ig.fault_group == "Locked primary"].iloc[0]

    hai = pd.read_csv(
        ROOT / "outputs/hai_external_validation_v2/HAI_EXTERNAL_AGGREGATE.csv"
    )
    hai = hai.loc[hai.label == "attack"].copy()
    hai["variant_order"] = hai.variant.map({"F0": 0, "F0-C": 1, "F1": 2})
    hai = hai.sort_values("variant_order")
    hai_chart = hai.melt(
        id_vars=["variant"],
        value_vars=["auroc_mean", "auprc_mean", "etaf1_mean"],
        var_name="metric",
        value_name="score",
    )
    hai_chart["metric"] = hai_chart.metric.map(
        {"auroc_mean": "AUROC", "auprc_mean": "AUPRC", "etaf1_mean": "eTaF1"}
    )
    hai_table = hai[
        [
            "variant",
            "parameters",
            "auroc_mean",
            "auprc_mean",
            "etaf1_mean",
            "fpr_mean",
            "detected_mean",
            "delay_mean",
        ]
    ]
    hai_index = hai.set_index("variant")
    hai_delta_auroc = float(
        hai_index.loc["F1", "auroc_mean"] - hai_index.loc["F0", "auroc_mean"]
    )

    conditional_manifest = json.loads(
        (
            ROOT
            / "outputs/hai_conditional_chum/HAI_CONDITIONAL_ANALYSIS_MANIFEST.json"
        ).read_text(encoding="utf-8")
    )
    conditional = pd.read_csv(
        ROOT / "outputs/hai_conditional_chum/HAI_CONDITIONAL_TARGETED_SUMMARY.csv"
    )
    stable = conditional.loc[
        (conditional["mode"] == "loo_sample")
        & conditional.stable_targeted_cell.astype(bool)
    ].copy()
    stable["event_channel"] = stable.apply(
        lambda row: f"{row.attack_id} / {row.feature}", axis=1
    )
    stable = stable[
        [
            "event_channel",
            "global_event",
            "target_class",
            "mean_delta_normalized_score",
            "min_delta_normalized_score",
            "positive_seeds",
            "mean_detection_drop",
            "mean_alarm_delay_increase",
            "max_abs_fpr_shift",
        ]
    ].sort_values(
        "mean_delta_normalized_score", ascending=False
    )
    stable_chart = stable.head(12).copy()
    quality = pd.read_csv(
        ROOT / "outputs/hai_conditional_chum/HAI_IMPUTER_QUALITY.csv"
    )
    quality = quality.loc[quality.reliable.astype(bool), [
        "feature",
        "r2",
        "sampled_sd_ratio",
        "sampled_mean_shift_sd",
        "lag1_error",
        "outside_train_range_fraction",
    ]].sort_values("r2", ascending=False)

    validation = pd.read_csv(
        ROOT / "outputs/final_evidence_validation/FINAL_EVIDENCE_CHECKS.csv"
    )
    validation = validation[["check", "status", "evidence"]]

    sources = [
        source(
            "src-g3",
            "Architecture Conditional CHUM G3",
            "outputs/architecture_chum_g3/G3_CELL_SUMMARY.csv",
            "SELECT * FROM read_csv_auto('outputs/architecture_chum_g3/G3_CELL_SUMMARY.csv') WHERE mode = 'loo_sample' AND stable_material = true",
            "아키텍처별 조건부 채널 효용 요약",
        ),
        source(
            "src-ig",
            "Integrated Gradients agreement",
            "outputs/integrated_gradients_baseline/IG_CHUM_CLASS_SUMMARY.csv",
            "SELECT * FROM read_csv_auto('outputs/integrated_gradients_baseline/IG_CHUM_CLASS_SUMMARY.csv')",
            "IG와 CHUM 채널 순위 합의 요약",
        ),
        source(
            "src-hai",
            "Corrected HAI external validation v2",
            "outputs/hai_external_validation_v2/HAI_EXTERNAL_AGGREGATE.csv",
            "SELECT * FROM read_csv_auto('outputs/hai_external_validation_v2/HAI_EXTERNAL_AGGREGATE.csv') WHERE label = 'attack'",
            "HAI 전역 공격 라벨의 모델 변형별 3-seed 평균",
        ),
        source(
            "src-hai-conditional",
            "HAI conditional event-channel summary",
            "outputs/hai_conditional_chum/HAI_CONDITIONAL_TARGETED_SUMMARY.csv",
            "SELECT * FROM read_csv_auto('outputs/hai_conditional_chum/HAI_CONDITIONAL_TARGETED_SUMMARY.csv') WHERE mode = 'loo_sample' AND stable_targeted_cell = true",
            "사전등록 기준을 통과한 HAI 직접 표적 event-channel 셀",
        ),
        source(
            "src-hai-imputer",
            "HAI conditional imputer quality",
            "outputs/hai_conditional_chum/HAI_IMPUTER_QUALITY.csv",
            "SELECT * FROM read_csv_auto('outputs/hai_conditional_chum/HAI_IMPUTER_QUALITY.csv') WHERE reliable = true",
            "HAI 조건부 대치 품질 게이트 통과 채널",
        ),
        source(
            "src-validation",
            "Final evidence validation",
            "outputs/final_evidence_validation/FINAL_EVIDENCE_CHECKS.csv",
            "SELECT * FROM read_csv_auto('outputs/final_evidence_validation/FINAL_EVIDENCE_CHECKS.csv')",
            "raw CSV에서 독립 재계산한 최종 증거 검사",
        ),
    ]

    cards = [
        {
            "id": "card-g3",
            "description": "잠근 primary fault 4개 중 두 새 아키텍처에서 동일 채널 합의를 보인 fault 수",
            "dataset": "card-g3",
            "sourceId": "src-g3",
            "metrics": [{"label": "Architecture-consensus faults", "field": "value", "format": "number"}],
        },
        {
            "id": "card-ig",
            "description": "primary architecture-fault 셀에서 IG와 CHUM의 1위 채널 일치",
            "dataset": "card-ig",
            "sourceId": "src-ig",
            "metrics": [{"label": "IG top-1 agreement", "field": "value", "format": "number", "unit": "/ 8"}],
        },
        {
            "id": "card-hai",
            "description": "HAI 전역 공격 AUROC의 F1-F0 3-seed 평균 차이",
            "dataset": "card-hai",
            "sourceId": "src-hai",
            "metrics": [{"label": "HAI mean ΔAUROC", "field": "value", "format": "number", "signed": True}],
        },
        {
            "id": "card-hai-conditional",
            "description": "직접 표적·대치 품질·3/3 seed·FPR 기준을 모두 통과한 HAI event-channel 셀",
            "dataset": "card-hai-conditional",
            "sourceId": "src-hai-conditional",
            "metrics": [{"label": "Stable targeted HAI cells", "field": "value", "format": "number"}],
        },
    ]

    charts = [
        {
            "id": "chart-g3",
            "title": "Accepted TEP conditional-utility cells",
            "subtitle": "조건부 XMV 대치 후 평균 AUROC 감소, architecture별 5 seeds",
            "intent": "comparison",
            "question": "동일 fault-channel 효용이 TCN과 Transformer에서 반복되는가?",
            "rationale": "Grouped bars expose architecture agreement for each accepted cell.",
            "type": "bar",
            "dataset": "g3-cells",
            "sourceId": "src-g3",
            "encodings": {
                "x": {"field": "fault_channel", "type": "nominal", "label": "Fault / channel"},
                "y": {"field": "mean_delta_auroc", "type": "quantitative", "label": "Mean ΔAUROC", "format": "number"},
                "color": {"field": "architecture_label", "type": "nominal", "label": "Architecture"},
                "tooltip": [
                    {"field": "positive_auroc_seeds", "type": "quantitative", "label": "Positive seeds"},
                    {"field": "max_abs_pre_fpr_shift", "type": "quantitative", "label": "Max |ΔFPR|"},
                ],
            },
            "settings": {"groupMode": "grouped", "showValues": True},
            "layout": "full",
        },
        {
            "id": "chart-hai",
            "title": "HAI global attack performance",
            "subtitle": "F0, capacity-matched F0-C, F1의 3-seed 평균",
            "intent": "comparison",
            "question": "제어 이력 추가 효과가 모델 용량 증가를 넘어서는가?",
            "rationale": "Grouped bars compare the three same-scale performance metrics across variants.",
            "type": "bar",
            "dataset": "hai-chart",
            "sourceId": "src-hai",
            "encodings": {
                "x": {"field": "variant", "type": "nominal", "label": "Variant"},
                "y": {"field": "score", "type": "quantitative", "label": "Score", "format": "number"},
                "color": {"field": "metric", "type": "nominal", "label": "Metric"},
            },
            "settings": {"groupMode": "grouped", "showValues": True},
            "layout": "full",
        },
    ]
    if len(stable_chart):
        charts.append(
            {
                "id": "chart-hai-conditional",
                "title": "Stable targeted HAI event-channel cells",
                "subtitle": "조건부 대치 후 threshold-normalized event score 감소, 3 seeds",
                "intent": "ranking",
                "question": "어떤 직접 공격 control 채널의 관측 이력이 anomaly score에 반복적으로 기여했는가?",
                "rationale": "A ranked bar chart shows the magnitude of accepted targeted cells without implying a time trend.",
                "type": "bar",
                "dataset": "hai-conditional-stable",
                "sourceId": "src-hai-conditional",
                "encodings": {
                    "x": {"field": "event_channel", "type": "nominal", "label": "Attack / control"},
                    "y": {"field": "mean_delta_normalized_score", "type": "quantitative", "label": "Mean normalized-score loss", "format": "number"},
                    "tooltip": [
                        {"field": "positive_seeds", "type": "quantitative", "label": "Positive seeds"},
                        {"field": "max_abs_fpr_shift", "type": "quantitative", "label": "Max |ΔFPR|"},
                    ],
                },
                "settings": {"showValues": True},
                "layout": "full",
            }
        )

    tables = [
        {
            "id": "table-g3",
            "title": "Accepted G3 cells",
            "subtitle": "두 새 아키텍처에서 합의하고 모든 안정성 게이트를 통과한 primary 셀",
            "dataset": "g3-cells",
            "sourceId": "src-g3",
            "defaultSort": {"field": "mean_delta_auroc", "direction": "desc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "fault_channel", "label": "Fault / XMV", "type": "text"},
                {"field": "architecture_label", "label": "Architecture", "type": "text"},
                {"field": "mean_delta_auroc", "label": "Mean ΔAUROC", "format": "number", "movement": True},
                {"field": "run_delta_auroc_ci_low", "label": "Run CI low", "format": "number"},
                {"field": "run_delta_auroc_ci_high", "label": "Run CI high", "format": "number"},
                {"field": "max_abs_pre_fpr_shift", "label": "Max |ΔFPR|", "format": "number"},
            ],
        },
        {
            "id": "table-ig",
            "title": "IG-CHUM rank agreement",
            "subtitle": "primary와 negative/exploratory fault 그룹 비교",
            "dataset": "ig-summary",
            "sourceId": "src-ig",
            "defaultSort": {"field": "top1_rate", "direction": "desc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "fault_group", "label": "Fault group", "type": "text"},
                {"field": "cells", "label": "Cells", "format": "number"},
                {"field": "median_spearman", "label": "Median Spearman", "format": "number"},
                {"field": "top1_agreements", "label": "Top-1 agreements", "format": "number"},
                {"field": "top1_rate", "label": "Top-1 rate", "format": "percent"},
                {"field": "mean_top3_overlap", "label": "Mean top-3 overlap", "format": "number"},
            ],
        },
        {
            "id": "table-hai",
            "title": "HAI variant metrics",
            "subtitle": "global attack label, 3 seeds; 동일 sensor targets와 validation calibration",
            "dataset": "hai-global",
            "sourceId": "src-hai",
            "defaultSort": {"field": "auroc_mean", "direction": "desc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "variant", "label": "Variant", "type": "text"},
                {"field": "parameters", "label": "Parameters", "format": "number"},
                {"field": "auroc_mean", "label": "AUROC", "format": "number"},
                {"field": "auprc_mean", "label": "AUPRC", "format": "number"},
                {"field": "etaf1_mean", "label": "eTaF1", "format": "number"},
                {"field": "fpr_mean", "label": "FPR", "format": "number"},
                {"field": "detected_mean", "label": "Event detected", "format": "percent"},
                {"field": "delay_mean", "label": "Delay (s)", "format": "number"},
            ],
        },
        {
            "id": "table-quality",
            "title": "HAI imputer quality-gated controls",
            "subtitle": "held-out normal validation에서 모든 사전 분포·시계열 기준을 통과한 12 controls",
            "dataset": "hai-imputer-quality",
            "sourceId": "src-hai-imputer",
            "defaultSort": {"field": "r2", "direction": "desc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "feature", "label": "Control", "type": "text"},
                {"field": "r2", "label": "LOO R²", "format": "number"},
                {"field": "sampled_sd_ratio", "label": "SD ratio", "format": "number"},
                {"field": "sampled_mean_shift_sd", "label": "Mean shift (SD)", "format": "number"},
                {"field": "lag1_error", "label": "Lag-1 error", "format": "number"},
                {"field": "outside_train_range_fraction", "label": "Outside range", "format": "percent"},
            ],
        },
        {
            "id": "table-validation",
            "title": "Independent evidence checks",
            "subtitle": "보고서 수치를 raw result tables에서 재계산한 최종 검사",
            "dataset": "validation-checks",
            "sourceId": "src-validation",
            "defaultSort": {"field": "check", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "check", "label": "Check", "type": "text"},
                {"field": "status", "label": "Status", "type": "text"},
                {"field": "evidence", "label": "Evidence", "type": "text"},
            ],
        },
    ]
    if len(stable):
        tables.insert(
            3,
            {
                "id": "table-hai-conditional",
                "title": "Stable targeted HAI event-channel cells",
                "subtitle": "직접 표적, 품질 대치기, 3/3 seed, score materiality, FPR 기준 통과 셀",
                "dataset": "hai-conditional-stable",
                "sourceId": "src-hai-conditional",
                "defaultSort": {"field": "mean_delta_normalized_score", "direction": "desc"},
                "density": "spacious",
                "layout": "full",
                "columns": [
                    {"field": "event_channel", "label": "Attack / control", "type": "text"},
                    {"field": "target_class", "label": "Target class", "type": "text"},
                    {"field": "mean_delta_normalized_score", "label": "Mean score loss", "format": "number", "movement": True},
                    {"field": "min_delta_normalized_score", "label": "Minimum seed loss", "format": "number"},
                    {"field": "positive_seeds", "label": "Positive seeds", "format": "number"},
                    {"field": "mean_detection_drop", "label": "Mean detection drop", "format": "number", "movement": True},
                    {"field": "mean_alarm_delay_increase", "label": "Mean delay Δ", "format": "number", "movement": True},
                    {"field": "max_abs_fpr_shift", "label": "Max |ΔFPR|", "format": "number"},
                ],
            },
        )

    conditional_decision_ko = (
        "외부 채널 수준 재현을 지지했다"
        if conditional_manifest["decision"] == "EXTERNAL_CHANNEL_SUPPORT"
        else "외부 채널 수준 재현 기준을 충족하지 못했다"
    )
    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {TITLE}", "layout": "full"},
        {
            "id": "summary",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## 기술 요약\n\n"
                "**현재 결과는 ‘새로운 대형 모델’보다 신뢰 가능한 AI 실험 프로토콜을 석사논문의 핵심 기여로 지지한다.** "
                f"TEP에서는 잠근 primary fault 4개 중 {consensus_faults}개가 TCN과 Transformer의 동일 채널 합의를 통과했고, "
                f"Integrated Gradients는 primary 셀 8개 중 {int(ig_primary.top1_agreements)}개에서 CHUM과 같은 1위 채널을 선택했다. "
                f"HAI v2에서는 F1이 F0보다 전역 AUROC가 {hai_delta_auroc:+.4f} 높았으며, 조건부 후속 검증은 {conditional_decision_ko}.\n\n"
                "해석 범위는 architecture-robust, event-specific conditional information utility이다. 물리적 인과성, root cause, 또는 모든 산업 시스템으로의 보편적 전이는 주장하지 않는다."
            ),
        },
        {"id": "headline-metrics", "type": "metric-strip", "cardIds": ["card-g3", "card-ig", "card-hai", "card-hai-conditional"], "layout": "full"},
        {
            "id": "g3-finding",
            "type": "markdown",
            "layout": "full",
            "sourceId": "src-g3",
            "body": "## 동일 TEP 채널 효용이 두 새 아키텍처에서 반복됐다\n\nFault 4/XMV10, fault 19/XMV7·8, fault 25/XMV2는 TCN과 Transformer 모두에서 양의 AUROC loss, 5/5 seed 방향 일치, 제한된 pre-fault FPR 이동, 양의 run-bootstrap 하한을 만족했다. 따라서 효과를 GRU 한 구조의 우연으로만 설명하기 어렵다.",
        },
        {"id": "g3-chart", "type": "chart", "chartId": "chart-g3", "layout": "full"},
        {"id": "g3-table", "type": "table", "tableId": "table-g3", "layout": "full"},
        {
            "id": "ig-finding",
            "type": "markdown",
            "layout": "full",
            "sourceId": "src-ig",
            "body": f"## 표준 gradient attribution은 primary fault에서 같은 채널 구조를 부분적으로 확인했다\n\nIG와 CHUM은 추정 대상이 다르지만 primary 셀 8개 중 {int(ig_primary.top1_agreements)}개에서 1위 채널이 일치했다. Negative/exploratory fault에서의 낮은 순위 상관은 이 합의가 모든 fault에 자동으로 나타나는 현상이 아님을 보여준다.",
        },
        {"id": "ig-table", "type": "table", "tableId": "table-ig", "layout": "full"},
        {
            "id": "hai-finding",
            "type": "markdown",
            "layout": "full",
            "sourceId": "src-hai",
            "body": f"## HAI v2는 작지만 일관된 capacity-controlled transfer를 보였다\n\n동일한 29개 sensor targets를 예측할 때 28개 control-history inputs를 추가한 F1은 F0보다 전역 AUROC가 {hai_delta_auroc:+.4f} 높았다. F1과 파라미터 수 차이가 0.26%인 F0-C도 F1의 AUROC·AUPRC·eTaF1을 재현하지 못했다. 세 seed가 모두 같은 방향이었다.",
        },
        {"id": "hai-chart", "type": "chart", "chartId": "chart-hai", "layout": "full"},
        {"id": "hai-table", "type": "table", "tableId": "table-hai", "layout": "full"},
        {
            "id": "conditional-finding",
            "type": "markdown",
            "layout": "full",
            "sourceId": "src-hai-conditional",
            "body": f"## HAI 조건부 후속 검증은 {conditional_decision_ko}\n\n정상 validation 품질 게이트를 통과한 12개 control만 primary 결론에 사용했다. 직접 공격 대상, normalized event-score 감소 0.05 이상, 3/3 seed 양의 방향, 최대 |ΔFPR| 0.005 이하를 동시에 만족한 셀은 {len(stable)}개였다. 사전등록 최소 기준은 3개였다.",
        },
    ]
    if len(stable):
        blocks.extend(
            [
                {"id": "conditional-chart", "type": "chart", "chartId": "chart-hai-conditional", "layout": "full"},
                {"id": "conditional-table", "type": "table", "tableId": "table-hai-conditional", "layout": "full"},
            ]
        )
    blocks.extend(
        [
            {
                "id": "quality-finding",
                "type": "markdown",
                "layout": "full",
                "sourceId": "src-hai-imputer",
                "body": "## HAI 대치 결론은 12개 품질 통과 control로 제한했다\n\nLeave-one-channel-out predictor는 목표 control의 직전값을 제외했고, 20-step 정상 잔차 블록으로 분산과 자기상관을 복원했다. R², 분산비, 평균 이동, lag-1 오차, 훈련 범위 이탈률을 모두 통과하지 못한 채널은 탐색 표에는 남기되 primary 판정에서는 제외했다.",
            },
            {"id": "quality-table", "type": "table", "tableId": "table-quality", "layout": "full"},
            {
                "id": "scope",
                "type": "markdown",
                "layout": "full",
                "body": "## 비교 기준과 측정 단위\n\n- **F0:** sensor/process-measurement history only.\n- **F1:** 같은 sensor target을 예측하면서 control/action history를 추가.\n- **F0-C:** F0 입력을 유지하고 hidden width만 늘린 capacity control.\n- **CHUM effect:** 관측 control history를 정상 조건부 표본으로 바꿨을 때 감소한 탐지 성능 또는 event score.\n- **안정성 단위:** TEP는 5 training seeds와 fault별 matched runs, HAI는 3 training seeds와 50 attacks.\n- **Threshold:** perturbation 조건별 정상 validation score의 고정 99.5 percentile; test label은 calibration에 사용하지 않음.",
            },
            {
                "id": "method",
                "type": "markdown",
                "layout": "full",
                "body": "## 실험 설계는 대안 설명을 단계별로 통제했다\n\n1. F0/F1/F0-C로 control 정보와 단순 모델 용량 증가를 분리했다.\n2. GRU, TCN, compact Transformer에서 동일 split과 alarm rule을 사용했다.\n3. Zero occlusion의 분포 이탈을 확인한 뒤 정상 train-only 조건부 대치와 별도 validation calibration을 사용했다.\n4. Seed 방향, 효과크기, FPR, imputer 품질, run/event grain을 동시에 검사했다.\n5. IG와 HAI로 attribution 방법 및 데이터셋 실패 모드를 달리한 삼각 검증을 수행했다.",
            },
            {
                "id": "validation",
                "type": "markdown",
                "layout": "full",
                "sourceId": "src-validation",
                "body": "## 최종 숫자는 raw result tables에서 독립 재계산했다\n\n최종 validator는 task·event·run 행 수와 복합키 중복을 검사하고, G3 accepted cells, HAI v2 모델 대비, HAI 조건부 판정과 FPR 예외를 원자료에서 다시 계산한다. 열 순서가 잘못된 HAI v1은 명시적으로 격리되어 어떤 최종 계산에도 포함되지 않는다.",
            },
            {"id": "validation-table", "type": "table", "tableId": "table-validation", "layout": "full"},
            {
                "id": "limitations",
                "type": "markdown",
                "layout": "full",
                "body": "## 결론의 경계와 남은 불확실성\n\n- TEP seed 5개에서 two-sided exact sign-flip test의 최소 p-value는 0.0625이므로 effect size, 방향 일치, run-level CI가 주 근거다.\n- HAI는 3 seeds뿐이며 bootstrap interval은 descriptive sensitivity summary다.\n- HAI 50개 attack은 모두 적어도 하나의 control point를 직접 조작하므로 unattacked-control transfer를 검증하지 못한다.\n- 조건부 표본은 관측분포 기반 attribution baseline이지 물리적 intervention이 아니다.\n- CHUM은 현재 새 estimator 이론보다 엄격히 통제된 empirical protocol에 가깝다.",
            },
            {
                "id": "recommendations",
                "type": "markdown",
                "layout": "full",
                "body": "## 다음 단계\n\n1. 최종 제목과 연구질문은 지도교수 승인 후 잠근다.\n2. 논문 본문은 `capacity control → architecture replication → conditional perturbation → attribution triangulation → external transfer`의 검증 사슬로 구성한다.\n3. 가능하면 TEP seed를 1개 이상 추가하고 imputer block length·draw 수 sensitivity를 수행한다.\n4. control을 직접 공격하지 않는 외부 데이터가 확보되면 가장 중요한 일반화 한계를 보완한다.\n5. 인과 제어, root cause, 산업 안전 보장 표현은 사용하지 않는다.",
            },
            {
                "id": "questions",
                "type": "markdown",
                "layout": "full",
                "body": "## 추가 연구 질문\n\n- 채널 효용이 실제 controller topology 또는 물리적 연결성과 어느 정도 일치하는가?\n- 더 유연한 sequence imputer에서도 accepted cells가 유지되는가?\n- control-target이 없는 외부 이상에서도 conditional utility가 반복되는가?\n- CHUM 효과를 conditional mutual information과 연결할 수 있는가?",
            },
        ]
    )

    datasets = {
        "card-g3": [{"value": int(consensus_faults)}],
        "card-ig": [{"value": int(ig_primary.top1_agreements)}],
        "card-hai": [{"value": hai_delta_auroc}],
        "card-hai-conditional": [{"value": int(len(stable))}],
        "g3-cells": records(g3),
        "ig-summary": records(ig),
        "hai-chart": records(hai_chart),
        "hai-global": records(hai_table),
        "hai-conditional-stable": records(stable),
        "hai-imputer-quality": records(quality),
        "validation-checks": records(validation),
    }
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": TITLE,
            "description": "지도교수 검토를 위한 CHUM 기술 보고서: 연구 질문, 실험 설계, 결과, 검산, 한계, 다음 단계",
            "generatedAt": generated_at,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": datasets,
        },
        "sources": sources,
    }
    (OUTPUT / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    markdown_parts: list[str] = []
    table_lookup = {table["id"]: table for table in tables}
    data_lookup = datasets
    for block in blocks:
        if block["type"] == "markdown":
            markdown_parts.append(block["body"])
        elif block["type"] == "table":
            table = table_lookup[block["tableId"]]
            markdown_parts.append(pd.DataFrame(data_lookup[table["dataset"]]).to_markdown(index=False))
    (OUTPUT / "PROFESSOR_RESEARCH_REPORT_KO.md").write_text(
        "\n\n".join(markdown_parts) + "\n", encoding="utf-8"
    )
    (OUTPUT / "REPORT_SOURCE_NOTES.md").write_text(
        """# Report Source Notes

Audience: technical. Primary delivery: portable self-contained HTML.

Required structure mapping: title; technical summary; G3, IG, HAI v2, and HAI conditional findings; scope and metric definitions; experimental design; robustness and limitations; next steps; further questions.

Chart map:
- G3 grouped bar: accepted fault-channel cells by architecture; tests architecture consensus.
- HAI grouped bar: F0/F0-C/F1 global metrics; tests capacity-controlled external transfer.
- HAI conditional ranked bar: accepted targeted event-channel normalized-score loss; tests external channel attribution. Omitted automatically if no cell passes.

IG, imputer quality, and validation evidence remain exact tables because audit lookup is more useful than another chart. The report excludes invalidated HAI v1 evidence.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

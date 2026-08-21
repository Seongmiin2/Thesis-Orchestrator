# CHUM 근거 및 재현 인덱스

이 문서는 `PROFESSOR_BRIEF_KO.md`의 주요 수치가 어디서 생성됐는지와, 기존 대형 산출물을 삭제한 뒤 어떤 코드로 재생성할 수 있는지를 기록한다. 최종 결론은 2026년 8월 21일의 Git 커밋 `bc6166f792e3faceb060a50de6219dddca393fd6`에 있던 raw outputs를 기준으로 작성됐다.

## 1. 근거 우선순위

| 우선순위 | 근거 | 최종 사용 여부 |
| ---: | --- | --- |
| 1 | raw CSV에서 독립 재계산한 final evidence validation | 사용 |
| 2 | 수정된 HAI v2, TEP G3, IG, HAI conditional 결과 | 사용 |
| 3 | 각 실험의 preregistration, config, runner, manifest | 방법 확인에 사용 |
| 4 | 초기 G1/G2 및 탐색 보고서 | 연구 경과와 교차 확인에만 사용 |
| 제외 | `hai_external_validation` v1 | column-order bug로 무효 |
| 제외 | G0 retrieval/FAVE-RAG 후보 | 현재 Reinartz formulation의 thesis 방향에서 폐기 |

## 2. 최종 근거와 남아 있는 실행 코드

### TEP architecture-conditional G3

- 설정: `configs/architecture_chum_g3.yaml`
- 실행: `experiments/run_architecture_chum_g3.py`
- 분석: `experiments/analyze_architecture_chum_g3.py`
- 대치 감사: `experiments/audit_conditional_imputer.py`
- 핵심 완료 규모: 340 tasks, 9,520 fault rows, 190,400 run rows
- 결론: primary faults 4, 19, 25에서 두 architecture 채널 합의, 총 4개 consensus cells

### Integrated Gradients baseline

- 설정: `configs/integrated_gradients_baseline.yaml`
- 실행: `experiments/run_integrated_gradients_baseline.py`
- 분석: `experiments/analyze_integrated_gradients_baseline.py`
- 완료 규모: 880 attribution rows
- 결론: primary architecture-fault cells 8개 중 top channel 7개 일치

### HAI 21.03 데이터 준비 및 감사

- 공격 대상 설정: `configs/hai_2103_attack_targets.csv`
- point 역할 설정: `configs/hai_2103_point_roles.csv`
- 데이터 감사: `experiments/audit_hai_2103.py`
- 데이터 준비: `experiments/prepare_hai_2103.py`
- 역할 검증: `experiments/validate_hai_2103_roles.py`
- 공격 대상 검증: `experiments/validate_hai_2103_attack_targets.py`
- 중요 처리: train-test exact telemetry overlap 43,202 training rows 제외

### 수정된 HAI v2 외부 검증

- 설정: `configs/hai_external_validation.yaml`
- 실행: `experiments/run_hai_external_validation.py`
- 분석: `experiments/analyze_hai_external_validation.py`
- 완료 규모: 9 models, 36 label metrics, 450 event rows
- 결론: global label에서 F1이 F0와 F0-C보다 AUROC/AUPRC/eTaF1 우세, 세 seed 동일 방향
- 주의: 같은 runner의 초기 `hai_external_validation` 출력은 feature column-order bug로 무효이며 인용 금지

### HAI conditional CHUM

- 설정: `configs/hai_conditional_chum.yaml`
- 실행: `experiments/run_hai_conditional_chum.py`
- 분석: `experiments/analyze_hai_conditional_chum.py`
- 대치 모델: `experiments/hai_conditional_imputer.py`
- 대치 감사: `experiments/audit_hai_conditional_imputer.py`
- 완료 규모: 171 tasks, 8,550 event rows
- 결론: 12/28 control channels가 imputer quality gate 통과, 직접 공격된 5개 event-channel cells가 최종 기준 통과

### 최종 독립 검증

- 검증 코드: `experiments/validate_final_evidence.py`
- 보고서 생성 코드: `experiments/build_professor_report_v2.py`
- 결과: completeness, duplication, effect, FPR, decision 관련 18 checks 전체 PASS

## 3. 실험 조건 요약

| 항목 | TEP | HAI 21.03 |
| --- | --- | --- |
| 반복 seed | 42–46, 5개 | 42–44, 3개 |
| 주요 비교 | F1 vs F0/F0-C | F1 vs F0/F0-C |
| 모델 구조 | GRU, TCN, compact Transformer | compact sequence forecaster variants |
| 임계값 | 정상 validation percentile | 정상 validation percentile |
| 주 평가 단위 | fault와 run | global attack event |
| 핵심 FPR 원칙 | 조건별 validation calibration, |ΔFPR| guardrail | 조건별 validation calibration, |ΔFPR| guardrail |

## 4. 최종 수치 체크섬 역할의 요약

다음 값은 향후 재실행 결과가 원래 검증 패키지와 일치하는지 빠르게 확인하는 기준이다.

- TEP G3 consensus primary faults: `[4, 19, 25]`
- TEP G3 consensus cells: `F4/XMV10`, `F19/XMV7`, `F19/XMV8`, `F25/XMV2`
- TEP G3 effect ranges: TCN ΔAUROC `0.05549–0.28994`, Transformer `0.05306–0.26368`
- IG primary top-1 agreement: `7/8`
- HAI v2 F1 global metrics: AUROC `0.851801`, AUPRC `0.504871`, eTaF1 `0.616226`, FPR `0.00270377`
- HAI v2 F1−F0: ΔAUROC `0.018745`, ΔAUPRC `0.030608`, ΔeTaF1 `0.026722`
- HAI v2 F1−F0-C: ΔAUROC `0.021744`, ΔAUPRC `0.038605`, ΔeTaF1 `0.035477`
- HAI quality-gated controls: `12/28`
- HAI conditional accepted targeted cells: `5`
- Final validation: `18 checks PASS`

## 5. 재생성 순서

정확한 CLI 인자는 각 runner의 `--help`와 해당 YAML을 기준으로 확인한다. 절대 경로는 이전 실행 환경을 포함할 수 있으므로 새 환경에 맞게 지정한다.

1. HAI를 사용할 경우 role/attack-target 검증과 exact-overlap 제거를 먼저 수행한다.
2. TEP G3 또는 HAI v2 runner를 해당 config로 실행한다.
3. 각 분석 스크립트로 aggregate report와 decision을 생성한다.
4. imputer audit를 통과한 채널만 conditional decision에 포함한다.
5. `experiments/validate_final_evidence.py`로 raw tables에서 최종 수치를 재계산한다.
6. 새 결과가 위 요약 값과 다른 경우 코드·데이터 버전·column order·split·threshold를 먼저 점검한다.

## 6. 보존 및 복구 정책

- 대형 CSV, NPY, PT checkpoint, log, HTML, 중복 Markdown/JSON 산출물은 저장소 경량화를 위해 제거됐다.
- 삭제된 파일은 Git 이력의 `bc6166f792e3faceb060a50de6219dddca393fd6`에 남아 있어 필요하면 파일 단위로 복구할 수 있다.
- 최종 논문에 숫자를 추가하거나 변경할 때는 이 문서만 수정하지 말고, raw 결과를 재생성한 뒤 final validator를 다시 통과시켜야 한다.
- 무효화된 HAI v1은 복구하더라도 최종 주장에 사용하지 않는다.

## 7. 남은 의사결정

- 최종 제목과 연구질문을 사람과 지도교수가 승인해야 한다.
- TEP seed 추가와 imputer sensitivity는 선택적 robustness 작업이다.
- control channel이 직접 공격되지 않은 외부 데이터 검증은 더 강한 일반화를 위한 선택 작업이다.

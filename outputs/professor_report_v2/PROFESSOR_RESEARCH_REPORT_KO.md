# 제어 이력의 조건부 정보 효용: 석사논문 연구 결과

## 기술 요약

**현재 결과는 ‘새로운 대형 모델’보다 신뢰 가능한 AI 실험 프로토콜을 석사논문의 핵심 기여로 지지한다.** TEP에서는 잠근 primary fault 4개 중 3개가 TCN과 Transformer의 동일 채널 합의를 통과했고, Integrated Gradients는 primary 셀 8개 중 7개에서 CHUM과 같은 1위 채널을 선택했다. HAI v2에서는 F1이 F0보다 전역 AUROC가 +0.0187 높았으며, 조건부 후속 검증은 외부 채널 수준 재현을 지지했다.

해석 범위는 architecture-robust, event-specific conditional information utility이다. 물리적 인과성, root cause, 또는 모든 산업 시스템으로의 보편적 전이는 주장하지 않는다.

## 동일 TEP 채널 효용이 두 새 아키텍처에서 반복됐다

Fault 4/XMV10, fault 19/XMV7·8, fault 25/XMV2는 TCN과 Transformer 모두에서 양의 AUROC loss, 5/5 seed 방향 일치, 제한된 pre-fault FPR 이동, 양의 run-bootstrap 하한을 만족했다. 따라서 효과를 GRU 한 구조의 우연으로만 설명하기 어렵다.

| fault_channel   | architecture_label   |   mean_delta_auroc |   mean_delta_auprc |   positive_auroc_seeds |   max_abs_pre_fpr_shift |   run_delta_auroc_ci_low |   run_delta_auroc_ci_high |
|:----------------|:---------------------|-------------------:|-------------------:|-----------------------:|------------------------:|-------------------------:|--------------------------:|
| Fault 19 / XMV7 | TCN                  |          0.13547   |          0.0924891 |                      5 |                 0.00075 |                0.126955  |                 0.143896  |
| Fault 19 / XMV7 | Transformer          |          0.123656  |          0.0852742 |                      5 |                 0.00075 |                0.113525  |                 0.133368  |
| Fault 19 / XMV8 | TCN                  |          0.0554896 |          0.0335126 |                      5 |                 0.0005  |                0.0505319 |                 0.0604331 |
| Fault 19 / XMV8 | Transformer          |          0.0530555 |          0.0319834 |                      5 |                 0.0005  |                0.0466735 |                 0.0601701 |
| Fault 25 / XMV2 | TCN                  |          0.289938  |          0.235805  |                      5 |                 0.0005  |                0.284973  |                 0.29458   |
| Fault 25 / XMV2 | Transformer          |          0.263676  |          0.196138  |                      5 |                 0.00125 |                0.254727  |                 0.273196  |
| Fault 4 / XMV10 | TCN                  |          0.165285  |          0.121145  |                      5 |                 0.00025 |                0.124141  |                 0.205403  |
| Fault 4 / XMV10 | Transformer          |          0.105502  |          0.0881655 |                      5 |                 0.00075 |                0.0881872 |                 0.119932  |

## 표준 gradient attribution은 primary fault에서 같은 채널 구조를 부분적으로 확인했다

IG와 CHUM은 추정 대상이 다르지만 primary 셀 8개 중 7개에서 1위 채널이 일치했다. Negative/exploratory fault에서의 낮은 순위 상관은 이 합의가 모든 fault에 자동으로 나타나는 현상이 아님을 보여준다.

| fault_group            |   cells |   median_spearman |   top1_agreements |   top1_rate |   mean_top3_overlap |
|:-----------------------|--------:|------------------:|------------------:|------------:|--------------------:|
| Negative / exploratory |       8 |          0.100457 |                 1 |       0.125 |                1.25 |
| Locked primary         |       8 |          0.543379 |                 7 |       0.875 |                1.75 |

## HAI v2는 작지만 일관된 capacity-controlled transfer를 보였다

동일한 29개 sensor targets를 예측할 때 28개 control-history inputs를 추가한 F1은 F0보다 전역 AUROC가 +0.0187 높았다. F1과 파라미터 수 차이가 0.26%인 F0-C도 F1의 AUROC·AUPRC·eTaF1을 재현하지 못했다. 세 seed가 모두 같은 방향이었다.

| variant   |   parameters |   auroc_mean |   auprc_mean |   etaf1_mean |   fpr_mean |   detected_mean |   delay_mean |
|:----------|-------------:|-------------:|-------------:|-------------:|-----------:|----------------:|-------------:|
| F0        |        26909 |     0.833055 |     0.474264 |     0.589504 | 0.00266729 |        0.82     |        47.48 |
| F0-C      |        31021 |     0.830057 |     0.466267 |     0.580749 | 0.00264014 |        0.813333 |        49.18 |
| F1        |        30941 |     0.851801 |     0.504871 |     0.616226 | 0.00270377 |        0.84     |        43.74 |

## HAI 조건부 후속 검증은 외부 채널 수준 재현을 지지했다

정상 validation 품질 게이트를 통과한 12개 control만 primary 결론에 사용했다. 직접 공격 대상, normalized event-score 감소 0.05 이상, 3/3 seed 양의 방향, 최대 |ΔFPR| 0.005 이하를 동시에 만족한 셀은 5개였다. 사전등록 최소 기준은 3개였다.

| event_channel    |   global_event | target_class         |   mean_delta_normalized_score |   min_delta_normalized_score |   positive_seeds |   mean_detection_drop |   mean_alarm_delay_increase |   max_abs_fpr_shift |
|:-----------------|---------------:|:---------------------|------------------------------:|-----------------------------:|-----------------:|----------------------:|----------------------------:|--------------------:|
| A305 / P2_CO_rpm |             30 | control_only         |                     0.214571  |                    0.16632   |                3 |              0        |                       0     |         2.54513e-05 |
| A203 / P1_LCV01D |              8 | control_only         |                     0.18919   |                    0.151941  |                3 |              0        |                       0     |         7.6354e-06  |
| A506 / P2_CO_rpm |             44 | mixed_control_sensor |                     0.107774  |                    0.0827242 |                3 |              0        |                       0     |         2.54513e-05 |
| A215 / P2_CO_rpm |             20 | mixed_control_sensor |                     0.090704  |                    0.0451501 |                3 |              0.666667 |                     100.667 |         2.54513e-05 |
| A216 / P1_FCV03D |             21 | control_only         |                     0.0588121 |                    0.050267  |                3 |              0        |                       0     |         7.6354e-06  |

## HAI 대치 결론은 12개 품질 통과 control로 제한했다

Leave-one-channel-out predictor는 목표 control의 직전값을 제외했고, 20-step 정상 잔차 블록으로 분산과 자기상관을 복원했다. R², 분산비, 평균 이동, lag-1 오차, 훈련 범위 이탈률을 모두 통과하지 못한 채널은 탐색 표에는 남기되 primary 판정에서는 제외했다.

| feature   |       r2 |   sampled_sd_ratio |   sampled_mean_shift_sd |   lag1_error |   outside_train_range_fraction |
|:----------|---------:|-------------------:|------------------------:|-------------:|-------------------------------:|
| P1_B4002  | 0.999989 |           1.00005  |             1.08409e-05 |  1.472e-06   |                    0           |
| P4_ST_GOV | 0.999913 |           1.00006  |             0.000132089 |  7.15213e-05 |                    0           |
| P1_B4022  | 0.99979  |           0.999948 |             0.000110137 |  0.000131138 |                    0           |
| P1_B2004  | 0.999583 |           1.00016  |             0.000193067 |  8.75501e-05 |                    0.0008026   |
| P4_LD     | 0.999474 |           1.00004  |             5.63648e-05 |  0.000601166 |                    2.27688e-05 |
| P1_B2016  | 0.999472 |           1.00014  |             0.000135969 |  0.000339072 |                    0           |
| P4_ST_LD  | 0.998724 |           0.999871 |             0.000276221 |  0.00139842  |                    5.6922e-06  |
| P1_FCV03Z | 0.996449 |           0.997933 |             0.00154743  |  0.000269159 |                    0.00151413  |
| P1_FCV03D | 0.996274 |           0.999977 |             0.00155848  |  0.000319274 |                    0.000244765 |
| P1_LCV01Z | 0.98935  |           1.00818  |             0.0032791   |  0.00116652  |                    0.000330148 |
| P1_LCV01D | 0.98684  |           1.00178  |             0.00177304  |  0.00171883  |                    0.00028461  |
| P2_CO_rpm | 0.53521  |           0.994185 |             0.0297892   |  0.0218567   |                    2.27688e-05 |

## 비교 기준과 측정 단위

- **F0:** sensor/process-measurement history only.
- **F1:** 같은 sensor target을 예측하면서 control/action history를 추가.
- **F0-C:** F0 입력을 유지하고 hidden width만 늘린 capacity control.
- **CHUM effect:** 관측 control history를 정상 조건부 표본으로 바꿨을 때 감소한 탐지 성능 또는 event score.
- **안정성 단위:** TEP는 5 training seeds와 fault별 matched runs, HAI는 3 training seeds와 50 attacks.
- **Threshold:** perturbation 조건별 정상 validation score의 고정 99.5 percentile; test label은 calibration에 사용하지 않음.

## 실험 설계는 대안 설명을 단계별로 통제했다

1. F0/F1/F0-C로 control 정보와 단순 모델 용량 증가를 분리했다.
2. GRU, TCN, compact Transformer에서 동일 split과 alarm rule을 사용했다.
3. Zero occlusion의 분포 이탈을 확인한 뒤 정상 train-only 조건부 대치와 별도 validation calibration을 사용했다.
4. Seed 방향, 효과크기, FPR, imputer 품질, run/event grain을 동시에 검사했다.
5. IG와 HAI로 attribution 방법 및 데이터셋 실패 모드를 달리한 삼각 검증을 수행했다.

## 최종 숫자는 raw result tables에서 독립 재계산했다

최종 validator는 task·event·run 행 수와 복합키 중복을 검사하고, G3 accepted cells, HAI v2 모델 대비, HAI 조건부 판정과 FPR 예외를 원자료에서 다시 계산한다. 열 순서가 잘못된 HAI v1은 명시적으로 격리되어 어떤 최종 계산에도 포함되지 않는다.

| check                                           | status   | evidence                                                                                                                                                                                                                                                                                                                                                                                                                              |
|:------------------------------------------------|:---------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| G3 task completeness                            | PASS     | 340 tasks; 9,520 fault rows and 190,400 run rows; no key duplicates                                                                                                                                                                                                                                                                                                                                                                   |
| G3 locked decision                              | PASS     | consensus primary faults=[4, 19, 25]; required=3                                                                                                                                                                                                                                                                                                                                                                                      |
| G3 raw recomputation tcn/F4/XMV10               | PASS     | {"architecture": "tcn", "fault_id": 4, "channel": 10, "mean_delta_auroc": 0.165284842192691, "mean_delta_auprc": 0.12114509119914889, "positive_seeds": 5, "max_abs_fpr_shift": 0.0002500000000000002, "run_ci_low": 0.1241405274086378}                                                                                                                                                                                              |
| G3 raw recomputation transformer/F4/XMV10       | PASS     | {"architecture": "transformer", "fault_id": 4, "channel": 10, "mean_delta_auroc": 0.10550240863787375, "mean_delta_auprc": 0.08816553069045707, "positive_seeds": 5, "max_abs_fpr_shift": 0.0007499999999999989, "run_ci_low": 0.0881871760797342}                                                                                                                                                                                    |
| G3 raw recomputation tcn/F19/XMV7               | PASS     | {"architecture": "tcn", "fault_id": 19, "channel": 7, "mean_delta_auroc": 0.13546986295681063, "mean_delta_auprc": 0.09248911838907656, "positive_seeds": 5, "max_abs_fpr_shift": 0.0007500000000000007, "run_ci_low": 0.126955361295681}                                                                                                                                                                                             |
| G3 raw recomputation transformer/F19/XMV7       | PASS     | {"architecture": "transformer", "fault_id": 19, "channel": 7, "mean_delta_auroc": 0.12365633720930236, "mean_delta_auprc": 0.08527423326104129, "positive_seeds": 5, "max_abs_fpr_shift": 0.0007500000000000007, "run_ci_low": 0.1135250332225913}                                                                                                                                                                                    |
| G3 raw recomputation tcn/F19/XMV8               | PASS     | {"architecture": "tcn", "fault_id": 19, "channel": 8, "mean_delta_auroc": 0.05548960132890361, "mean_delta_auprc": 0.03351258074923298, "positive_seeds": 5, "max_abs_fpr_shift": 0.0005000000000000004, "run_ci_low": 0.0505318812292358}                                                                                                                                                                                            |
| G3 raw recomputation transformer/F19/XMV8       | PASS     | {"architecture": "transformer", "fault_id": 19, "channel": 8, "mean_delta_auroc": 0.05305553571428576, "mean_delta_auprc": 0.031983406379951895, "positive_seeds": 5, "max_abs_fpr_shift": 0.0004999999999999987, "run_ci_low": 0.0466735008305647}                                                                                                                                                                                   |
| G3 raw recomputation tcn/F25/XMV2               | PASS     | {"architecture": "tcn", "fault_id": 25, "channel": 2, "mean_delta_auroc": 0.2899378363787376, "mean_delta_auprc": 0.2358046337981372, "positive_seeds": 5, "max_abs_fpr_shift": 0.0005000000000000004, "run_ci_low": 0.2849726910299004}                                                                                                                                                                                              |
| G3 raw recomputation transformer/F25/XMV2       | PASS     | {"architecture": "transformer", "fault_id": 25, "channel": 2, "mean_delta_auroc": 0.26367607558139533, "mean_delta_auprc": 0.19613822798311048, "positive_seeds": 5, "max_abs_fpr_shift": 0.0012500000000000011, "run_ci_low": 0.2547268625415282}                                                                                                                                                                                    |
| Integrated Gradients completeness and agreement | PASS     | 880 rows; primary top-1 agreement=7/8                                                                                                                                                                                                                                                                                                                                                                                                 |
| HAI data gates                                  | PASS     | 43,202 overlaps removed; F0=29, added controls=28; 50 target-mapped events                                                                                                                                                                                                                                                                                                                                                            |
| HAI invalid run quarantine                      | PASS     | INVALIDATED_COLUMN_ORDER_BUG                                                                                                                                                                                                                                                                                                                                                                                                          |
| HAI v2 task completeness                        | PASS     | 9 models; 36 label metrics; 450 event rows; no key duplicates                                                                                                                                                                                                                                                                                                                                                                         |
| HAI v2 external support recomputation           | PASS     | [{"contrast":"F1-F0","mean_delta_auroc":0.018745,"positive_seeds_auroc":3,"mean_delta_auprc":0.030608,"positive_seeds_auprc":3,"mean_delta_etaf1":0.026722,"positive_seeds_etaf1":3,"mean_fpr_increase":0.000036},{"contrast":"F1-F0-C","mean_delta_auroc":0.021744,"positive_seeds_auroc":3,"mean_delta_auprc":0.038605,"positive_seeds_auprc":3,"mean_delta_etaf1":0.035477,"positive_seeds_etaf1":3,"mean_fpr_increase":0.000064}] |
| HAI conditional imputer quality gate            | PASS     | 12/28 channels passed all locked imputer-quality criteria                                                                                                                                                                                                                                                                                                                                                                             |
| HAI conditional task completeness               | PASS     | 171 tasks; 8,550 event rows; no composite-key duplicates                                                                                                                                                                                                                                                                                                                                                                              |
| HAI conditional decision recomputation          | PASS     | {"decision": "EXTERNAL_CHANNEL_SUPPORT", "stable_targeted_cells": 5, "fpr_exceptions": {"loo_sample": 0, "zero": 0}}                                                                                                                                                                                                                                                                                                                  |

## 결론의 경계와 남은 불확실성

- TEP seed 5개에서 two-sided exact sign-flip test의 최소 p-value는 0.0625이므로 effect size, 방향 일치, run-level CI가 주 근거다.
- HAI는 3 seeds뿐이며 bootstrap interval은 descriptive sensitivity summary다.
- HAI 50개 attack은 모두 적어도 하나의 control point를 직접 조작하므로 unattacked-control transfer를 검증하지 못한다.
- 조건부 표본은 관측분포 기반 attribution baseline이지 물리적 intervention이 아니다.
- CHUM은 현재 새 estimator 이론보다 엄격히 통제된 empirical protocol에 가깝다.

## 다음 단계

1. 최종 제목과 연구질문은 지도교수 승인 후 잠근다.
2. 논문 본문은 `capacity control → architecture replication → conditional perturbation → attribution triangulation → external transfer`의 검증 사슬로 구성한다.
3. 가능하면 TEP seed를 1개 이상 추가하고 imputer block length·draw 수 sensitivity를 수행한다.
4. control을 직접 공격하지 않는 외부 데이터가 확보되면 가장 중요한 일반화 한계를 보완한다.
5. 인과 제어, root cause, 산업 안전 보장 표현은 사용하지 않는다.

## 추가 연구 질문

- 채널 효용이 실제 controller topology 또는 물리적 연결성과 어느 정도 일치하는가?
- 더 유연한 sequence imputer에서도 accepted cells가 유지되는가?
- control-target이 없는 외부 이상에서도 conditional utility가 반복되는가?
- CHUM 효과를 conditional mutual information과 연결할 수 있는가?

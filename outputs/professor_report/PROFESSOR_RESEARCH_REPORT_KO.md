# 제어 이력의 조건부 정보 효용: 석사논문 연구 결과

## 기술 요약

**현재 결과는 석사논문의 실험적 핵심으로 충분히 강하다.** 연구 질문은 단순히 산업 데이터에서 성능을 높이는 것이 아니라, 부분 관측 시계열 예측에서 *과거 제어 이력이 용량 증가를 넘어 실제로 추가 정보를 제공하는가*, 그리고 그 정보 효용을 특정 채널 수준에서 아키텍처와 귀인 방법이 바뀌어도 재현할 수 있는가이다.

TEP에서는 세 아키텍처의 event-level gain을 확인한 뒤, 분포 보존 조건부 개입에서 locked primary fault 4개 중 3개가 TCN과 Transformer의 동일 채널 합의를 통과했다. Integrated Gradients는 primary 셀 8개 중 7개에서 같은 1위 채널을 골랐다. HAI 21.03에서는 F1이 F0와 용량대조 F0-C보다 AUROC, AUPRC, eTaF1에서 3/3 seed 우위를 보였다.

**논문의 정체성은 새로운 거대 모델이 아니라 신뢰 가능한 AI engineering 실험 방법론이다.** 다만 인과 제어 이론, 새로운 범용 attribution algorithm, top-tier AI 방법론 논문 수준까지 증명된 것은 아니다.

## 문제의식은 산업 응용보다 표현과 검증에 있다

센서만 보는 forecasting detector는 제어 시스템의 상태를 완전히 관측하지 못할 수 있다. 제어 명령의 과거를 추가하면 성능이 오를 수 있지만, 그 상승은 입력 차원과 파라미터 증가, 특정 architecture의 inductive bias, 비현실적인 zero occlusion, 또는 false alarm 변화로도 설명될 수 있다. 이 연구는 이러한 대안 설명을 하나씩 제거해 **control history의 조건부 정보 효용**을 측정한다.

교수님께는 이를 `industrial anomaly detection application`보다 `partial observability에서 action-history representation을 검증하는 model-agnostic evaluation study`로 설명하는 편이 정확하다.

## 같은 채널 효과가 두 아키텍처에서 재현됐다

분포와 시간 구조를 복원하는 leave-one-channel-out residual sampler로 XMV 하나씩 교체했다. Fault 4/XMV10, fault 19/XMV7·8, fault 25/XMV2는 TCN과 Transformer 모두에서 5/5 seed 양의 AUROC loss, 최대 pre-fault FPR 변화 0.0013 이하, 계층적 run bootstrap CI 0 초과를 만족했다. Fault 26/XMV4는 효과가 크지만 imputer 품질 기준을 통과하지 못해 결론에서 제외했다.

## 표준 gradient attribution도 primary fault에서 같은 이야기를 했다

Integrated Gradients와 CHUM은 추정 대상이 다르다. IG는 정상 평균 baseline에서 현재 forecasting-error score의 국소 민감도를, CHUM은 한 채널 정보를 대체했을 때 fault detection performance가 잃는 양을 측정한다. 그럼에도 primary architecture-fault 셀에서는 7/8 top-1 채널 일치와 median Spearman 0.543을 보였다. Negative/exploratory 셀의 median Spearman은 0.101에 그쳐 무차별적 합의로 보기는 어렵다.

| fault_channel   | architecture_label   |   mean_delta_auroc |   mean_delta_auprc |   positive_auroc_seeds |   max_abs_pre_fpr_shift |   run_delta_auroc_ci_low |   run_delta_auroc_ci_high |
|:----------------|:---------------------|-------------------:|-------------------:|-----------------------:|------------------------:|-------------------------:|--------------------------:|
| Fault 19 / XMV7 | TCN                  |             0.1355 |             0.0925 |                      5 |                  0.0008 |                   0.127  |                    0.1439 |
| Fault 19 / XMV7 | Transformer          |             0.1237 |             0.0853 |                      5 |                  0.0008 |                   0.1135 |                    0.1334 |
| Fault 19 / XMV8 | TCN                  |             0.0555 |             0.0335 |                      5 |                  0.0005 |                   0.0505 |                    0.0604 |
| Fault 19 / XMV8 | Transformer          |             0.0531 |             0.032  |                      5 |                  0.0005 |                   0.0467 |                    0.0602 |
| Fault 25 / XMV2 | TCN                  |             0.2899 |             0.2358 |                      5 |                  0.0005 |                   0.285  |                    0.2946 |
| Fault 25 / XMV2 | Transformer          |             0.2637 |             0.1961 |                      5 |                  0.0012 |                   0.2547 |                    0.2732 |
| Fault 4 / XMV10 | TCN                  |             0.1653 |             0.1211 |                      5 |                  0.0002 |                   0.1241 |                    0.2054 |
| Fault 4 / XMV10 | Transformer          |             0.1055 |             0.0882 |                      5 |                  0.0007 |                   0.0882 |                    0.1199 |

## HAI 외부검증은 작지만 일관된 transfer를 보였다

HAI 공식 train/test에서 정확히 겹치는 43,202개 훈련 행을 먼저 제거했다. F0와 F1은 같은 29개 다음 시점 sensor/model signal을 예측하고, F1만 28개 nonconstant control-history input을 추가했다. F0-C는 F0 입력을 유지하면서 F1과 파라미터 차이를 0.26%로 맞췄다.

Global attack label의 3-seed 평균에서 F1은 F0보다 AUROC +0.0187, AUPRC +0.0306, eTaF1 +0.0267 높았고, 평균 FPR 증가는 0.000036이었다. 세 seed 모두 세 성능 지표에서 같은 방향이었다.

## 외부효과는 mixed target 공격에서 더 분명했다

50개 HAI 이벤트 모두 적어도 하나의 control-history point를 직접 공격한다. Control-only 31건에서는 F1의 detection-rate 증가는 0이지만 세 seed 모두 평균 지연이 약 0.39초 줄었다. Control+sensor mixed 19건에서는 F1-F0 detection rate가 +0.0526, censored delay가 9.21초 개선됐고 두 결과 모두 3/3 seed 같은 방향이었다. 따라서 HAI는 transfer stress test이지만, 제어 채널이 전혀 공격되지 않은 상황의 효용을 독립 검증하지는 못한다.

| fault_group            |   cells |   median_spearman |   top1_agreements |   top1_rate |   mean_top3_overlap |
|:-----------------------|--------:|------------------:|------------------:|------------:|--------------------:|
| Negative / exploratory |       8 |            0.1005 |                 1 |       0.125 |                1.25 |
| Locked primary         |       8 |            0.5434 |                 7 |       0.875 |                1.75 |

## 비교 기준과 측정 단위

- **F0:** sensor/process-measurement history only.
- **F1:** 동일한 sensor target을 예측하되 control/action history를 입력에 추가.
- **F0-C:** F0 입력을 유지하고 hidden width만 늘린 capacity control.
- **CHUM effect:** original AUROC에서 특정 control channel의 조건부 대체 후 AUROC를 뺀 값. 양수일수록 해당 채널 정보가 중요하다.
- **안정성 단위:** TEP는 5 training seeds와 fault별 20 matched test runs, HAI는 3 training seeds와 50 attack events.
- **Threshold:** 정상 validation score의 고정 percentile. Test label로 threshold를 조정하지 않았다.

## 실험 설계는 대안 설명을 순서대로 제거했다

1. GRU, TCN, compact Transformer에서 F0/F1/F0-C를 같은 split과 threshold 규칙으로 비교했다.
2. Zero occlusion의 out-of-distribution 문제를 발견한 뒤, target XMV 자신의 직전 값을 predictor에서 제외한 ridge imputer를 정상 train run에만 적합했다.
3. 20-step residual block bootstrap으로 교체 채널의 분산과 자기상관을 복원하고, held-out normal run에서 분포 품질을 gate로 사용했다.
4. Seed 방향, materiality, FPR, imputer quality, paired hierarchical run bootstrap을 모두 만족해야 channel consensus로 인정했다.
5. 표준 IG attribution과 HAI 외부 데이터로 서로 다른 실패 모드에 대한 삼각 검증을 수행했다.

| variant   |   parameters |   auroc_mean |   auprc_mean |   etaf1_mean |   fpr_mean |   detected_mean |   delay_mean |
|:----------|-------------:|-------------:|-------------:|-------------:|-----------:|----------------:|-------------:|
| F0        |        26909 |       0.8331 |       0.4743 |       0.5895 |     0.0027 |          0.82   |        47.48 |
| F0-C      |        31021 |       0.8301 |       0.4663 |       0.5807 |     0.0026 |          0.8133 |        49.18 |
| F1        |        30941 |       0.8518 |       0.5049 |       0.6162 |     0.0027 |          0.84   |        43.74 |

## 최종 숫자는 보고서가 아니라 raw CSV에서 다시 계산됐다

최종 validator는 G3 340개 task와 190,400 run rows의 중복·완전성을 확인하고, accepted G3 cell의 seed mean과 FPR shift를 raw result에서 재계산했다. IG 880행과 HAI 36 metric rows·450 event rows도 독립 검산했다. HAI 초기 실행에서 발견한 train/test column-order bug는 별도 invalid directory로 격리되어 최종 분석기가 사용할 수 없다.

| target_group                        | contrast   |   mean_delta_detected_ratio |   positive_seeds_detected_ratio |   mean_delta_censored_delay |   positive_seeds_delay |
|:------------------------------------|:-----------|----------------------------:|--------------------------------:|----------------------------:|-----------------------:|
| Control-only target (31 events)     | F1-F0      |                      0      |                               0 |                      0.3871 |                      3 |
| Control-only target (31 events)     | F1-F0-C    |                      0      |                               0 |                      0.3763 |                      3 |
| Control + sensor target (19 events) | F1-F0      |                      0.0526 |                               3 |                      9.2105 |                      3 |
| Control + sensor target (19 events) | F1-F0-C    |                      0.0702 |                               3 |                     13.7018 |                      3 |

## 현재 결론의 경계

- 이 결과는 **conditional information utility**를 지지하지만 물리적·제어이론적 인과성을 증명하지 않는다.
- TEP의 seed는 5개라 two-sided exact sign-flip test의 최소 p-value가 0.0625이다. Effect size, 모든 seed 방향, run-level CI가 주 근거다.
- HAI는 3 seeds뿐이며 descriptive external support로 해석해야 한다. Process label은 partial annotation이다.
- HAI 21.03의 모든 attack이 control point를 직접 조작하므로 완전한 indirect-effect external validation은 아니다.
- CHUM은 현재 엄밀한 새 estimator 이론보다 잘 통제된 empirical protocol에 가깝다. 이 점이 석사논문에는 충분하지만 강한 방법론 학회 논문에는 부족하다.

## 교수님께 보고할 때의 권장 프레이밍

1. 첫 문장은 도메인이 아니라 문제로 시작한다: **부분 관측 동적 시스템에서 action history의 추가 정보가 model capacity와 architecture를 넘어 재현되는가?**
2. 핵심 기여는 모델 정확도보다 `capacity control → architecture replication → distribution-preserving intervention → attribution triangulation → external transfer`의 검증 사슬이라고 설명한다.
3. 산업 안전이나 인과 제어를 크게 주장하지 않는다. 현재 가장 강한 표현은 **architecture-robust, fault-specific conditional information utility**다.
4. 다음 실험은 seed를 최소 1개 이상 추가해 exact seed test의 해상도를 확보하고, imputer block length/draw 수 sensitivity와 control-target이 없는 외부 데이터 1개를 보강하는 것이다.
5. 논문 제목과 초록에서는 HAI나 TEP보다 `control-history-aware forecasting`과 `conditional channel utility`를 앞에 둔다.

## 남은 연구 질문

- XMV utility는 실제 controller topology나 물리 연결성과 어느 정도 일치하는가?
- Leave-one-channel-out imputer를 확률적 sequence model로 바꿔도 accepted cell이 유지되는가?
- Fault 26/XMV4처럼 효과는 크지만 imputer 품질이 낮은 셀을 어떤 identification 문제로 다뤄야 하는가?
- 제어 채널이 직접 공격되지 않는 외부 시스템에서도 같은 representation gain이 나타나는가?
- CHUM의 효과량을 정보이론적 conditional mutual information 또는 causal representation 관점으로 정식화할 수 있는가?

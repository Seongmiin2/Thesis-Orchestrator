# 산업 시계열 AI는 추가 Context를 실제로 활용하는가?

지도교수 검토용 연구방향 및 진행 보고
작성일: 2026년 8월 21일
현재 상태: 실험·검증 패키지 완료, 최종 문제정의와 주장 범위 승인 대기

## Executive Summary

- **문제의식:** 산업 이상 탐지에서 센서에 제어 이력을 추가해 평균 성능이 올랐다는 결과만으로는 어느 사건과 채널이 실제 정보를 제공하는지 알 수 없다. 모델 크기 증가, 특정 architecture의 편향, 비현실적인 occlusion, 오경보 증가도 효과처럼 보일 수 있다.
- **해결방안:** Control-History Utility Mapping(CHUM)은 sensor-only(F0), sensor+control(F1), capacity-matched sensor-only(F0-C)를 비교하고, 정상 분포 기반 조건부 대치·validation 임계값·FPR guardrail·seed/run 안정성·architecture consensus를 결합해 event–channel별 추가 예측 정보를 감사한다.
- **핵심 결과:** TEP에서 fault 4, 19, 25의 핵심 channel mapping이 TCN과 compact Transformer 양쪽에서 재현됐다. Integrated Gradients도 primary 셀 8개 중 7개에서 같은 최상위 채널을 선택했다. 수정된 HAI 21.03 v2에서도 F1이 전역 AUROC·AUPRC·eTaF1에서 F0와 F0-C보다 높았고, 5개 직접 공격 event–channel 셀이 조건부 대치 기준을 통과했다.
- **현재 결정:** 증거는 본문 집필을 시작할 수준이다. 오늘 필요한 것은 실험을 더 벌이는 것이 아니라, 논문 기여를 “새 모델”이 아닌 **제어 이력 유용성 감사 프로토콜과 architecture-robust 실증**으로 확정하고 인과·root-cause 주장을 배제하는 것이다.

## 왜 이 연구가 Superintelligence 연구실에 맞는가

이 연구의 중심은 전기·전자 공정 자체가 아니라 **AI가 추가 context를 실제로 활용하는지, 그 활용 근거를 신뢰할 수 있는지 검증하는 문제**다. 산업 시계열은 이를 엄격하게 시험할 수 있는 도메인이다.

| 연구실의 관심 영역 | 공통 AI 문제 | 본 연구의 대응 |
| --- | --- | --- |
| RAG·문서 탐색 | 검색된 문서가 관련 있어도 생성 모델이 실제로 사용하는가 | 추가된 제어 context가 탐지 결정에 실제 정보를 제공하는지 F0/F1/F0-C로 분리 |
| RAG 신뢰성 | 불필요하거나 잘못된 context가 출력을 교란하는가 | event별 이질성, zero perturbation의 FPR 교란, 조건부 대치 비교 |
| Medical Vision AI | 모델이 올바른 근거를 사용하는지 설명을 신뢰할 수 있는가 | IG 단독 설명이 아니라 성능 저하·분포 보존·모델 간 합의로 설명을 검증 |
| 통신·전기전자 AI | 시계열·상태·제어 context가 시스템 판단에 어떤 정보를 주는가 | 센서 상태와 제어 이력을 분리한 industrial multivariate time-series 실증 |

RAG가 “검색 성공”과 “검색 문서 활용”을 구분해야 하듯, 본 연구도 “제어 이력 입력”과 “제어 이력의 실제 활용”을 구분한다. Medical XAI에서 heatmap 하나만으로 근거를 신뢰할 수 없듯, 본 연구도 단일 attribution map이 아니라 **distribution-preserving perturbation, 성능 변화, FPR, architecture consensus**를 함께 요구한다.

따라서 교수님께는 다음과 같이 설명하는 것이 가장 정확하다.

> **CHUM은 산업 시계열을 대상으로 한 trustworthy context-utilization audit다. 추가 context의 유용성을 모델 용량, perturbation-induced distribution shift, false alarms, architecture-specific behavior와 분리한다.**

이 연결은 확장 가능성을 설명하기 위한 것이며, 현재 논문이 RAG나 의료 영상에서 CHUM을 검증했다고 주장하는 것은 아니다.

## 선행연구를 검토한 뒤의 정확한 신규성

조건부 perturbation 자체를 새 원리라고 주장해서는 안 된다. 선행연구는 이미 다음 문제를 다뤘다.

| 선행연구 축 | 이미 알려진 것 | 본 연구가 추가하는 것 |
| --- | --- | --- |
| FIT, TimeSHAP | 시계열 feature·time attribution과 conditional history 고려 | point prediction 설명이 아니라 fault/event 탐지 utility, FPR, run-level 안정성을 평가 |
| Learned perturbation | 고정 perturbation이 시계열 설명을 왜곡할 수 있음 | 정상-only leave-one-channel-out imputer와 residual block sampling을 산업 제어 이력에 적용하고 품질 gate를 둠 |
| OOD explanation 연구 | feature removal이 out-of-distribution 입력을 만들 수 있음 | zero와 조건부 대치의 pre-fault FPR을 직접 비교하고 조건별 validation threshold를 재보정 |
| Model-multiplicity/Rashomon 연구 | 단일 모델 importance는 불안정할 수 있음 | GRU·TCN·Transformer에서 event gain을 확인하고 TCN·Transformer channel consensus를 locked decision으로 사용 |
| Distribution-aware Medical XAI | 설명 방법 자체의 leakage와 신뢰성 평가 필요 | IG를 주 결론으로 쓰지 않고 CHUM과 primary/negative faults에서 일치 정도를 비교 |
| RAG context-utilization 연구 | 관련 context가 있어도 모델 활용과 robustness는 별도 문제 | 같은 문제를 산업 multivariate time series의 sensor/control context에서 정량화 |

따라서 가장 방어 가능한 신규성 문장은 다음과 같다.

> **본 연구는 새로운 범용 attribution 알고리즘을 주장하지 않는다. 산업 이상 탐지에서 control context의 event–channel utility를 capacity-matched baselines, distribution-preserving replacement, condition-specific calibration, FPR guardrails, hierarchical uncertainty, cross-architecture consensus로 판정하는 통합 평가 프로토콜과 실증을 제시한다.**

이 포지셔닝은 과도한 알고리즘 신규성 주장을 피하면서도, 기존 시계열 XAI 논문이 각각 따로 다룬 문제를 산업 이상 탐지의 재현 가능한 decision gate로 결합했다는 기여를 분명히 한다.

### 핵심 1차 문헌

- Tonekaboni et al., *What Went Wrong and When? Instance-wise Feature Importance for Time-Series Black-Box Models*, NeurIPS 2020: https://proceedings.neurips.cc/paper/2020/hash/08fa43588c2571ade19bc0fa5936e028-Abstract.html
- Bento et al., *TimeSHAP: Explaining Recurrent Models through Sequence Perturbations*, KDD 2021: https://arxiv.org/abs/2012.00073
- Hase et al., *The Out-of-Distribution Problem in Explainability and Search Methods for Feature Importance Explanations*, NeurIPS 2021: https://proceedings.neurips.cc/paper/2021/hash/1def1713ebf17722cbe300cfc1c88558-Abstract.html
- Enguehard, *Learning Perturbations to Explain Time Series Predictions*, ICML 2023: https://proceedings.mlr.press/v202/enguehard23a.html
- Donnelly et al., *The Rashomon Importance Distribution*, NeurIPS 2023: https://proceedings.neurips.cc/paper_files/paper/2023/hash/1403ab1a427050538ec59c7f570aec8b-Abstract-Conference.html
- Jethani et al., *Don’t Be Fooled: Label Leakage in Explanation Methods and the Importance of Their Quantitative Evaluation*, AISTATS 2023: https://proceedings.mlr.press/v206/jethani23a.html
- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS 2020: https://proceedings.neurips.cc/paper/2020/hash/6b493230-Abstract.html
- Li et al., *Characterizing Query-Knowledge Relevance for Reliable RAG*, EMNLP 2024: https://aclanthology.org/2024.emnlp-main.353/

## 실험 규모가 보여주는 연구 완성도

현재 패키지는 단순한 단일 모델 ablation이 아니다.

- **두 데이터 환경:** simulated process faults인 TEP와 hardware-in-the-loop attacks인 HAI 21.03
- **세 architecture 계열:** GRU, TCN, compact Transformer
- **세 model controls:** F0, F1, capacity-matched F0-C
- **여러 perturbation controls:** zero, legacy conditional mean, leave-one-channel-out stochastic replacement
- **반복성:** TEP 5 seeds, HAI 3 seeds
- **평가 grain:** sample score가 아니라 fault/event와 run/episode 수준 결과
- **교란 통제:** validation-only threshold, condition-specific recalibration, FPR guardrail
- **대치 품질:** predictive R², variance ratio, mean shift, lag-1 error, train-range violation, Wasserstein distance
- **불확실성:** 2,000회 paired hierarchical seed→run bootstrap
- **설명 교차검증:** Integrated Gradients와 primary/negative fault agreement 비교
- **외부 검증:** HAI 데이터 role·attack-target audit, 43,202 exact-overlap rows 제거, official time-aware metric 사용
- **연구 무결성:** feature-order bug가 있던 HAI v1 전체 무효화, corrected v2만 최종 validator에 포함
- **최종 QA:** completeness, duplicates, effect, FPR, decision 관련 18개 raw-table 재계산 검사 통과

이 구성의 강점은 모델 수가 많다는 사실 자체가 아니라, **가능한 대안 설명을 하나씩 차단했다는 것**이다.

| 대안 설명 | 대응 통제 |
| --- | --- |
| F1이 커서 좋아졌다 | F0-C capacity control |
| GRU에서만 나타났다 | TCN·Transformer 반복 |
| zero masking 충격이다 | 정상 조건부 대치와 FPR 비교 |
| 우연한 seed 결과다 | 5/3 seeds와 방향 filter |
| 평균값만 좋아졌다 | fault/event별 분석 |
| threshold를 test에 맞췄다 | validation-only condition calibration |
| imputer가 비현실적이다 | held-out distribution·lag quality gate |
| 하나의 설명법만 그렇게 말한다 | IG triangulation |
| 데이터 누수다 | run/episode 분리와 HAI exact-overlap 제거 |
| 실패 결과를 숨겼다 | HAI v1 명시적 invalidation과 격리 |

## 추가 실험에 대한 냉정한 결정

### 최종 논문 전에 우선 수행할 좁은 필수 보강

**TEP primary consensus sensitivity**를 수행한다. 전체 340-task grid를 반복하지 않고, 최종 합의 셀 `F4/XMV10`, `F19/XMV7`, `F19/XMV8`, `F25/XMV2`에 한정한다.

- residual block length: 5, 10, 20
- conditional draws: 1, 3, 10
- architecture: TCN, Transformer
- seeds: 기존 42–46 checkpoint 재사용
- 판정: 효과 방향, ΔAUROC 크기, run CI 하한, 최대 절대 ΔFPR, consensus 유지 여부

이 실험은 central method의 두 고정 선택(block length=20, draws=3)에 대한 공격을 직접 막는다. 새 학습이 아니라 기존 checkpoint 추론이므로 범위를 제한하면 현실적이다.

### 교수 피드백에 따라 선택할 보강

- TEP seed 1–2개 추가: seed-level resolution을 개선하지만 새 학습 비용이 든다.
- HAI conditional sensitivity: TEP sensitivity와 같은 방향이면 외부 robustness를 강화한다.
- uncertainty summary 강화: bootstrap repeat 또는 imputer draw 간 분산 보고.

### 4–5주 일정상 하지 말아야 할 것

- 새 대형 architecture 개발
- 세 번째 대규모 데이터셋 전체 파이프라인
- FIT·TimeSHAP·FANS 전체 재구현 경쟁
- RAG나 Medical Vision으로 실제 실험 도메인 확장
- causal/root-cause 검증을 위한 새 intervention 연구

이 항목들은 연구의 초점을 흐리고 논문 집필 시간을 잠식한다. 현재 논문은 **trustworthy context-utilization audit의 산업 시계열 실증**으로 완성하는 것이 맞다.

## 1. 문제의식을 한 문장으로 좁히면

> **어떤 산업 이상 사건에서, 어떤 제어 채널의 과거 이력이, 모델 용량과 오경보율을 통제한 뒤에도 재현 가능한 추가 탐지 정보를 제공하는가?**

기존의 단순한 질문은 “센서와 제어변수를 같이 넣으면 탐지 성능이 오르는가?”였다. 하지만 이 질문은 다음 네 가지를 구분하지 못한다.

1. 모든 이상에 제어 이력이 유용한지, 일부 사건에만 유용한지
2. 정보가 실제 제어 이력에서 오는지, 단순한 모델 파라미터 증가에서 오는지
3. 특정 GRU 구조만의 현상인지, 다른 temporal architecture에서도 반복되는지
4. 채널을 0으로 지워 만든 비현실적 입력이 중요도와 오경보율을 왜곡하는지

따라서 본 연구는 전체 평균 성능 경쟁에서 벗어나 **event–channel별 조건부 정보 유용성**을 측정한다.

## 2. 해결방안: CHUM

CHUM(Control-History Utility Mapping)은 다음 지도를 만든다.

`사건 × 제어 채널 × 시간 블록 × 모델 구조 → 추가 탐지 정보 유용성`

핵심 설계는 다음과 같다.

- **용량 통제:** F0(sensor-only), F1(sensor+control), F0-C(capacity-matched sensor-only)를 비교한다.
- **조건부 대치:** 제어 채널을 0으로 만들지 않고, 정상 training 데이터로 학습한 대치 모델로 해당 이력을 교체한다.
- **오경보 통제:** 각 조건의 threshold는 test label이 아니라 정상 validation 점수로 보정한다.
- **안정성 통제:** 평균 효과 외에 seed 방향, run/event bootstrap, FPR 이동, imputer quality를 확인한다.
- **구조 통제:** GRU, TCN, compact Transformer에서 같은 효과가 반복되는지 확인한다.

이 연구의 방법적 기여는 새로운 거대 모델이 아니라, 제어 이력이 **언제·어디서** 유용한지를 과장 없이 검증하는 audit protocol이다.

## 3. 어젯밤까지 완료한 작업

### 3.1 TEP 사건 수준 architecture robustness

Reinartz TEP의 2,800 runs, 28 faults, 41 XMEAS, 11 XMV를 사용했다. TEP 주 실험은 seed 42–46의 5개 반복이며 run 단위 분할, 정상 training scaler, validation threshold를 사용했다.

TCN과 compact Transformer의 capacity-controlled 비교에서 동일한 7개 GAIN faults가 나타났다: **4, 7, 19, 23, 24, 25, 26**. fault 23은 원래 GRU strict GAIN 집합에는 없던 near-threshold 결과이므로 architecture-dependent로 표시한다.

| Fault | TCN F1−F0-C ΔAUROC | Transformer F1−F0-C ΔAUROC | seed 방향 |
| ---: | ---: | ---: | ---: |
| 4 | 0.1663 | 0.1087 | 양쪽 5/5 |
| 7 | 0.4880 | 0.4845 | 양쪽 5/5 |
| 19 | 0.1783 | 0.1970 | 양쪽 5/5 |
| 23 | 0.0300 | 0.0234 | 양쪽 5/5 |
| 24 | 0.1091 | 0.0877 | 양쪽 5/5 |
| 25 | 0.2916 | 0.2776 | 양쪽 5/5 |
| 26 | 0.3868 | 0.4154 | 양쪽 5/5 |

**의미:** 제어 이력의 사건별 이득은 GRU 고유 현상이나 단순 용량 증가로 설명되지 않는다.

### 3.2 TEP 채널 수준 architecture consensus

조건부 대치 품질과 run-level interval을 함께 적용했을 때, primary faults 중 fault 4, 19, 25가 TCN과 Transformer 양쪽에서 채널 합의를 얻었다. fault 19에 두 채널이 있어 총 4개 consensus cells다.

| Fault / channel | TCN ΔAUROC | Transformer ΔAUROC | TCN CI 하한 | Transformer CI 하한 | 최대 절대 ΔFPR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fault 4 / XMV10 | 0.1653 | 0.1055 | 0.1241 | 0.0882 | 0.00075 |
| Fault 19 / XMV7 | 0.1355 | 0.1237 | 0.1270 | 0.1135 | 0.00075 |
| Fault 19 / XMV8 | 0.0555 | 0.0531 | 0.0505 | 0.0467 | 0.00050 |
| Fault 25 / XMV2 | 0.2899 | 0.2637 | 0.2850 | 0.2547 | 0.00125 |

모든 셀은 각 architecture에서 5/5 seed가 같은 방향이었고 run-level bootstrap 하한이 0보다 컸다. fault 26/XMV4는 raw effect가 컸지만 imputer quality gate를 통과하지 못해 최종 합의에서 제외했다.

**의미:** 큰 효과만 선택한 것이 아니라, 대치 품질이 부족한 결과를 실제로 탈락시키는 보수적인 기준이 작동했다.

### 3.3 Integrated Gradients 교차 확인

CHUM과 estimand가 다른 Integrated Gradients를 보조 baseline으로 적용했다.

| 구분 | 셀 수 | Spearman 중앙값 | top-1 일치 | top-1 비율 |
| --- | ---: | ---: | ---: | ---: |
| Locked primary | 8 | 0.5434 | 7 | 0.875 |
| Negative / exploratory | 8 | 0.1005 | 1 | 0.125 |

**의미:** 모든 fault에서 attribution이 자동으로 같은 채널을 고른 것이 아니라, CHUM 효과가 강한 primary faults에서만 높은 일치가 나타났다.

### 3.4 HAI 21.03 외부 검증

HAI 21.03은 8개 episode, 1,323,608 rows, 50개 global attack events로 구성된다. train-test에 정확히 겹친 telemetry 43,202 training rows를 제거했고, 29 sensor targets와 28 active control histories를 사용했다. 외부 실험은 seed 42–44의 3개 반복이다.

최초 실행은 training/test feature column order가 달라진 버그로 무효화했다. 아래 결과는 수정된 v2만 사용한다.

| Variant | Parameters | AUROC | AUPRC | eTaF1 | FPR | 평균 지연 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F0 | 26,909 | 0.8331 | 0.4743 | 0.5895 | 0.00267 | 47.48 |
| F0-C | 31,021 | 0.8301 | 0.4663 | 0.5807 | 0.00264 | 49.18 |
| F1 | 30,941 | 0.8518 | 0.5049 | 0.6162 | 0.00270 | 43.74 |

F1−F0의 평균 차이는 AUROC +0.018745, AUPRC +0.030608, eTaF1 +0.026722였고 세 seed가 모두 같은 방향이었다. F1−F0-C도 AUROC +0.021744, AUPRC +0.038605, eTaF1 +0.035477이었다. F1과 F0-C의 파라미터 차이는 약 0.26%다.

**의미:** 다른 HIL 데이터에서도 event-level control-history benefit이 관찰됐지만, 모든 process subgroup에서 균일한 것은 아니므로 global-label descriptive support로 한정한다.

### 3.5 HAI channel-level conditional support

28 active controls 중 12개가 imputer quality gate를 통과했다. 직접 공격된 event–channel 셀 중 사전 기준 3개를 넘는 5개 셀이 최종 통과했다.

| Attack / channel | 유형 | 평균 normalized score 감소 | 최소 seed 감소 | seed 방향 |
| --- | --- | ---: | ---: | ---: |
| A305 / P2_CO_rpm | control-only | 0.2146 | 0.1663 | 3/3 |
| A203 / P1_LCV01D | control-only | 0.1892 | 0.1519 | 3/3 |
| A506 / P2_CO_rpm | mixed | 0.1078 | 0.0827 | 3/3 |
| A215 / P2_CO_rpm | mixed | 0.0907 | 0.0452 | 3/3 |
| A216 / P1_FCV03D | control-only | 0.0588 | 0.0503 | 3/3 |

**의미:** HAI에서 직접 공격된 일부 control channel이 탐지 score에 조건부 예측 정보를 제공했다. 그러나 모든 HAI 사건이 control point를 직접 공격하므로, 공격받지 않은 제어 이력의 보편적 유용성을 증명하지는 않는다.

### 3.6 최종 검증과 프로젝트 정리

- G3: 340 tasks, 9,520 fault rows, 190,400 run rows, key 중복 없음
- IG: 880 rows, primary top-1 agreement 7/8
- HAI v2: 9 models, 36 label metrics, 450 event rows, key 중복 없음
- HAI conditional: 171 tasks, 8,550 event rows, 5개 accepted cells
- raw table에서 다시 계산한 completeness·effect·FPR·decision 관련 18개 검증 모두 PASS
- 무효화된 HAI v1은 최종 주장과 검증에서 완전히 제외
- 중복 HTML, CSV, NPY, PT, log, intermediate report를 정리하고 이 문서와 재현 인덱스만 보존

## 4. 현재 프로젝트 상태

| 항목 | 상태 | 판단 |
| --- | --- | --- |
| 문제 후보 탐색 | 완료 | 단순 feature 비교와 현재 형태의 retrieval 방향은 폐기 |
| TEP event-level robustness | PASS | GRU·TCN·Transformer에서 이질적 gain 반복 |
| TEP channel-level G3 | PASS | 3 primary faults, 4 consensus cells |
| IG triangulation | PASS | primary top-1 7/8 일치 |
| HAI 데이터 감사 | PASS with limitation | overlap 제거 및 role/target audit 완료 |
| HAI v2 외부 검증 | EXTERNAL_SUPPORT | global label 중심의 제한적 외부 지지 |
| HAI conditional channel 검증 | EXTERNAL_CHANNEL_SUPPORT | 5개 직접 공격 셀 통과 |
| 최종 raw evidence 검증 | PASS | 18 checks 통과 |
| 최종 제목·연구질문 | 미확정 | 지도교수와 사용자 승인 필요 |
| 논문 본문 | 미착수 | 방향 승인 즉시 시작 가능 |

현재는 **추가 탐색 단계가 아니라 논문 주장과 목차를 잠그는 단계**다.

## 5. 주장 가능한 것과 금지할 것

### 주장 가능한 것

1. 제어 이력의 추가 정보는 이상 사건별로 강하게 이질적이다.
2. TEP의 일부 fault–channel 조합은 용량, threshold, FPR, imputer quality를 통제해도 서로 다른 architecture에서 반복된다.
3. 정상 분포 기반 조건부 대치는 zero occlusion의 비현실적 분포 이동과 FPR 교란을 줄이는 데 유리하다.
4. HAI v2는 다른 HIL 환경에서 event-level benefit과 일부 직접 공격 channel utility에 제한적인 외부 지지를 제공한다.

### 금지하거나 유보할 것

- 선택된 제어 채널이 fault의 물리적 원인이라는 주장
- controller action이 인과적으로 탐지를 개선한다는 주장
- CHUM이 intervention 없이 root cause를 식별한다는 주장
- TEP 결과가 실제 플랜트 전체에 자동 일반화된다는 주장
- 5개 TEP seed 또는 3개 HAI seed를 독립 데이터셋 반복으로 해석하는 주장

TEP의 5개 seed에서는 양측 exact sign-flip test의 최소 p값이 0.0625다. 따라서 효과크기, 방향 일치, run-level interval을 보고하되 seed-level p<0.05를 주장하지 않는다.

## 6. 오늘 지도교수에게 확인받을 결정

1. 문제의식을 **“제어 이력을 추가하면 좋아지는가”에서 “언제·어떤 채널이 추가 정보를 제공하는가”로 확정**해도 되는가?
2. 논문의 주 기여를 새 detector가 아니라 **CHUM audit protocol과 architecture-robust empirical evidence**로 두어도 되는가?
3. 최종 claim을 조건부 예측 정보로 제한하고 causal/root-cause 표현을 명시적으로 배제할 것인가?
4. 현재 evidence package로 본문 집필을 시작하고 추가 seed와 imputer sensitivity는 선택적 보강으로 둘 것인가?

## 7. 오늘부터의 실행 계획

### 오늘 반드시 끝낼 일

1. **한 문장 문제정의 확정**
   “산업 이상 탐지에서 과거 제어 이력의 추가 정보는 사건과 채널에 따라 이질적이며, CHUM은 이를 모델 용량·오경보율·분포 이동을 통제해 측정한다.”를 기본안으로 삼는다.
2. **지도교수 전달본 발송**
   이 문서를 첨부하고 아래 메일 초안으로 방향 승인과 추가 실험의 필수 여부를 묻는다.
3. **승인 전 병행 작업**
   Introduction의 문제정의·contribution 문단과 Methods의 F0/F1/F0-C·conditional replacement·FPR guardrail 초안을 작성한다.
4. **승인 후 목차 잠금**
   문제정의 → CHUM → TEP architecture robustness → channel consensus → IG → HAI → 한계 순으로 본문을 전개한다.

### 선택적 보강 실험

- TEP seed 추가
- imputer block length 및 draw-count sensitivity
- control channel이 직접 공격되지 않은 외부 데이터 검증

이 항목들은 robustness와 일반성을 강화하지만 현재 핵심 결론의 선행조건은 아니다. 지도교수가 필수로 판단하지 않는 한 오늘의 전달자료와 본문 착수를 늦추지 않는다.

## 8. 지도교수에게 보낼 메일 초안

**제목: 석사논문 연구방향 및 CHUM 검증 결과 검토 요청**

교수님 안녕하세요.

석사논문 연구방향을 기존의 단순한 sensor-only 대 sensor+control 성능 비교에서, **산업 이상 탐지에서 제어 이력이 어떤 사건과 채널에 조건부로 유용한지 검증하는 문제**로 구체화했습니다.

이를 위해 Control-History Utility Mapping(CHUM)이라는 분석 프로토콜을 구성했습니다. sensor-only, sensor+control, capacity-matched sensor-only 모델을 비교하고, 정상 데이터 기반 조건부 대치와 오경보율 통제를 적용한 뒤 GRU·TCN·Transformer에서 결과가 반복되는지 확인했습니다.

현재 TEP에서는 fault 4, 19, 25의 핵심 channel mapping이 TCN과 Transformer 양쪽에서 재현됐고, Integrated Gradients도 primary 셀 8개 중 7개에서 같은 최상위 채널을 선택했습니다. HAI 21.03에서도 sensor+control 모델이 global AUROC·AUPRC·eTaF1에서 두 대조군보다 높았으며, 직접 공격된 5개 event–channel 셀이 조건부 대치 기준을 통과했습니다. 최종 validator의 18개 재계산 검사도 모두 통과했습니다.

다만 본 연구의 주장은 물리적 인과나 root cause 식별이 아니라, **사건·채널별 조건부 예측 정보의 유용성**으로 제한하려고 합니다.

첨부 자료를 검토해 주시고 다음 사항에 대한 의견을 부탁드립니다.

1. 이 문제의식과 연구질문을 최종 석사논문 방향으로 확정해도 되는지
2. 주 기여를 새로운 탐지 모델보다 CHUM 검증 프로토콜과 architecture-robust 실증으로 두는 것이 적절한지
3. 현재 결과로 본문 집필을 시작하고 추가 seed 및 imputer sensitivity를 선택적 보강으로 두어도 되는지

감사합니다.

## 9. 제안 제목과 논문 구조

- 국문: **산업 시계열 이상 탐지에서 제어 이력의 조건부 유용성: 모델 구조에 강건한 변수·시간 기여도 분석**
- 영문: **When Does Control History Help? Architecture-Robust Utility Attribution for Industrial Time-Series Anomaly Detection**

제안 목차:

1. Introduction: 평균 성능 비교가 숨기는 event-level heterogeneity
2. Related Work: controller-aware detection, perturbation attribution, architecture robustness
3. Problem Formulation and CHUM
4. Datasets, split, role/provenance audit
5. Event-level capacity-controlled results
6. Channel-level conditional utility and architecture consensus
7. Integrated Gradients triangulation
8. HAI external validation
9. Limitations and claim boundaries
10. Conclusion

## 10. 결론

현재 증거는 **제어 이력이 일부 산업 이상 사건에서만 유용하며, 그 유용성이 특정 채널에 집중되고 서로 다른 모델 구조에서도 반복될 수 있다**는 주장을 지지한다. 가장 중요한 기여는 더 큰 모델이 아니라, capacity control·validation calibration·조건부 대치·FPR guardrail·architecture consensus를 결합해 그 유용성을 감사하는 재현 가능한 방법이다. 오늘의 우선순위는 새 실험을 늘리는 것이 아니라 이 문제의식과 주장 범위를 지도교수와 확정하고 본문 집필을 시작하는 것이다.

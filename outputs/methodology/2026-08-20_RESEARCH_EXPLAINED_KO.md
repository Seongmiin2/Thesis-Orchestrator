# 오늘까지 진행한 석사논문 연구 정리

작성일: 2026년 8월 20일  
대상 독자: 연구자인 본인, 지도교수, 산업 AI에 관심 있는 기술 독자  
현재 단계: Tennessee Eastman Process 내부 검증 완료, HAI 외부 검증 준비 완료

## 1. 먼저 결론부터

현재 논문은 단순히 **“GRU에 제어변수(XMV)를 더 넣으면 성능이 좋아진다”**를 주장하는 연구가 아니다. 그 주장은 기존 연구와 쉽게 겹치고, 모델 파라미터 증가나 특정 모델 구조 때문에 나타난 결과일 수 있기 때문이다.

이 논문이 실제로 다루는 문제는 다음과 같다.

> 산업 공정의 과거 제어 이력은 모든 이상 상황에서 똑같이 유용한가? 그렇지 않다면 어떤 이상 사건, 어떤 제어 채널, 어떤 시간 구간에서 유용하며, 그 현상은 GRU뿐 아니라 서로 다른 시계열 모델에서도 반복되는가?

오늘까지의 실험에서 얻은 가장 중요한 결과는 다음과 같다.

- 과거 XMV 정보의 효과는 모든 fault에서 균일하지 않았다.
- 효과는 일부 fault에 매우 강하게 집중됐다.
- 이 fault별 효과는 GRU뿐 아니라 TCN과 compact Transformer에서도 반복됐다.
- 모델 파라미터 수를 맞춘 sensor-only 대조군으로도 효과가 설명되지 않았다.
- 비현실적인 zero occlusion 대신 정상 데이터 기반 conditional replacement를 사용했을 때에도 fault 4, 19, 25, 26에서 안정적인 제어채널 효과가 남았다.
- 따라서 현재 논문의 가장 강한 기여는 **새로운 거대 신경망**이 아니라, 제어 이력의 조건부 유용성을 엄밀하게 측정하는 방법과 architecture-robust 실증 결과다.

현재 가장 적절한 영문 가제는 다음과 같다.

> **When Does Control History Help? Architecture-Robust Utility Attribution for Industrial Time-Series Anomaly Detection**

국문 가제는 다음과 같다.

> **산업 시계열 이상 탐지에서 제어 이력의 조건부 유용성: 모델 구조에 강건한 변수·시간 기여도 분석**

---

## 2. 논문에서 정의한 문제

### 2.1 산업 공정 데이터에는 센서와 제어 이력이 함께 존재한다

산업 공정에서는 크게 두 종류의 시계열이 발생한다.

- `XMEAS`: 압력, 온도, 유량, 레벨 등 공정 상태를 관측하는 센서값
- `XMV`: 밸브 개도, 유량 조작값 등 공정을 조절하기 위해 사용된 manipulated variable 이력

일반적인 이상 탐지 모델은 센서만 사용하거나 센서와 제어변수를 전부 하나의 feature 집합으로 합쳐 사용한다. 센서와 제어 이력을 함께 사용했을 때 성능이 오르면 흔히 “제어변수가 도움이 됐다”고 결론을 내린다.

그러나 이 결론에는 다음 문제가 있다.

1. 입력 변수가 늘면서 모델 파라미터도 늘어났기 때문에 성능이 오른 것일 수 있다.
2. 전체 평균 성능은 좋아져도 실제로는 몇 개 fault에서만 효과가 나타날 수 있다.
3. XMV는 독립적인 인과적 조치가 아니라 이미 발생한 fault에 대한 controller 반응일 수 있다.
4. 특정 GRU 구조에서만 우연히 XMV를 잘 사용했을 수 있다.
5. 변수를 0으로 지우는 일반적인 occlusion은 실제 공정에서 발생하지 않는 입력을 만들어 잘못된 중요도를 낼 수 있다.
6. 정상 구간의 false alarm이 함께 증가했다면 탐지 성능 향상이라고 보기 어렵다.

따라서 이 논문은 “XMV를 넣을 것인가?”보다 더 좁고 검증 가능한 문제를 정의한다.

### 2.2 최종 연구 문제

이 논문의 중심 문제는 **Control-History Utility**, 즉 과거 제어 이력이 이상 탐지에 제공하는 추가 정보의 유용성이다.

이를 네 개의 연구질문으로 나눈다.

#### RQ1. 제어 이력의 효과는 fault마다 다른가?

센서+제어 모델이 센서-only 및 capacity-matched sensor-only 모델보다 좋은 fault가 일부에만 집중되는지 확인한다.

#### RQ2. 어떤 제어 채널과 시간 구간이 효과를 만드는가?

특정 XMV 채널을 제거하거나 과거 시간 구간을 제한했을 때 성능이 얼마나 감소하는지 측정한다.

#### RQ3. 이 현상은 모델 구조와 무관하게 반복되는가?

GRU, TCN, compact Transformer에서 같은 fault별 효과가 재현되는지 확인한다.

#### RQ4. 시뮬레이션이 아닌 다른 제어 시스템에서도 나타나는가?

HAI hardware-in-the-loop 데이터에서 sensor-only와 sensor+control의 사건별 차이를 재현한다.

현재 RQ1과 RQ3은 강하게 지지됐고, RQ2는 GRU에서 일부 fault에 대해 지지됐다. RQ4는 데이터 취득과 적합성 감사까지 완료됐으며 실제 학습이 다음 단계다.

---

## 3. 왜 이 연구 방향을 선택했는가

### 3.1 처음 검토한 두 후보

처음에는 다음 두 방향을 비교했다.

1. 센서와 제어 이력을 함께 사용하는 predictive anomaly detection
2. 과거 사례를 검색해 사용하는 applicability-aware retrieval 또는 FAVE-RAG

두 번째 방향은 아이디어 자체는 흥미로웠지만 현재 데이터에서 논문의 핵심 변수를 객관적으로 정의할 수 없었다.

- operating mode가 없었다.
- XMV12가 없었다.
- 검증된 fault mechanism label이 없었다.
- 미래 제어 계획 또는 적용 가능성의 객관적인 ground truth가 없었다.
- 실행 가능한 oracle proxy도 sensor-only retrieval보다 유의미한 downstream headroom을 만들지 못했다.

따라서 retrieval 방향을 계속 진행하면 “applicability”라는 이름은 있지만 실제로는 단순 XMV 유사도를 측정하는 연구가 될 위험이 컸다. 이 방향은 현재 데이터 구성에서 폐기했다.

### 3.2 제어 이력 방향이 살아남은 이유

반면 센서+제어 예측 모델은 capacity control, 여러 seed, run-level 불확실성 분석 후에도 fault별로 강한 구조를 보였다.

특히 초기 GRU 실험에서 fault 3, 4, 7, 19, 24, 25, 26이 strict GAIN으로 분류됐다. 이 중 다수는 파라미터 수를 맞춘 F0-C보다도 큰 차이를 유지했다.

이 결과는 다음 질문으로 이어질 충분한 근거가 됐다.

> 왜 일부 fault에서만 XMV 이력이 큰 정보를 제공하는가?

즉 단순 feature 비교를 논문의 기여로 삼은 것이 아니라, feature 비교에서 발견된 **이질성(heterogeneity)**을 새로운 연구 문제로 승격했다.

---

## 4. 왜 GRU를 사용했고, 왜 TCN과 Transformer를 추가했는가

### 4.1 GRU는 최고 모델이 아니라 통제된 분석 도구였다

GRU를 처음 사용한 이유는 다음과 같다.

- 길이 20의 짧은 시계열 window를 처리하기에 충분하다.
- 모델이 작아 5개 seed와 많은 ablation을 반복할 수 있다.
- sensor-only와 sensor+control 구조를 거의 동일하게 만들 수 있다.
- 파라미터 수를 맞춘 대조군을 만들기 쉽다.
- 순차적 inductive bias가 있어 산업 시계열의 기준 모델로 사용하기 적절하다.

그러나 GRU 하나만 사용하면 “이 결과는 GRU에서만 나타난 것 아닌가?”라는 비판을 피할 수 없다.

### 4.2 서로 다른 세 모델 계열을 선택했다

따라서 다음 세 계열로 확장했다.

| 모델 | 시간정보를 처리하는 방식 | 논문에서의 역할 |
| --- | --- | --- |
| GRU | recurrent hidden state | 최초 발견 및 상세 CHUM 분석 |
| TCN | temporal convolution | 순환 구조가 없는 지역·다중 시간패턴 대조군 |
| Compact Transformer | self-attention | attention 기반 구조에서의 재현성 확인 |

세 모델의 절대 성능을 겨루는 것이 목적은 아니다. 핵심은 모델이 달라도 동일한 fault에서 제어 이력의 추가 정보가 나타나는지 확인하는 것이다.

### 4.3 모델 크기 효과를 어떻게 통제했는가

각 architecture에서 다음 세 조건을 비교했다.

- `F0`: XMEAS sensor-only
- `F1`: XMEAS + XMV
- `F0-C`: XMEAS-only이지만 F1과 파라미터 수를 최대한 맞춘 capacity control

예를 들어 TCN의 파라미터 수는 다음과 같다.

| 조건 | 파라미터 수 |
| --- | ---: |
| TCN F0 | 8,425 |
| TCN F1 | 9,481 |
| TCN F0-C | 9,526 |

Transformer는 F0/F0-C가 20,489개, F1이 20,841개로 차이가 약 1.7%였다.

따라서 F1이 F0-C를 이기면 단순히 모델이 커서 좋아졌다는 설명은 약해진다.

---

## 5. 논문이 제안하는 솔루션: CHUM

### 5.1 CHUM이란 무엇인가

이 논문에서 제안하는 최소 방법론 기여의 이름은 다음과 같다.

> **Control-History Utility Mapping (CHUM)**

CHUM은 새로운 anomaly detector 하나가 아니라, 기존 detector가 과거 제어 이력을 언제 어떻게 활용하는지 검증하는 audit 방법이다.

CHUM이 만드는 결과는 다음과 같은 map이다.

```text
이상 사건 × 제어 채널 × 과거 시간 구간 × 모델 구조
                         ↓
               추가적인 탐지 유용성
```

### 5.2 CHUM의 핵심 구성

#### 1단계: 전체 제어 이력의 추가 효과 측정

같은 architecture 안에서 sensor+control 모델과 두 sensor-only 대조군을 비교한다.

```text
Event utility = 성능(sensor + control) - 성능(capacity-matched sensor-only)
```

전체 평균만 보지 않고 fault 또는 attack event별로 계산한다.

#### 2단계: 제어 채널별 conditional replacement

단순 zero occlusion은 정상 운전에서 불가능한 값을 만들 수 있다. 이를 보완하기 위해 정상 train 구간만 사용해 다음 관계를 학습했다.

```text
현재 XMV ≈ f(현재 XMEAS, 직전 XMV)
```

특정 XMV 채널을 0으로 바꾸는 대신 정상 조건에서 예상되는 값으로 교체하고 detector 성능이 얼마나 감소하는지 측정한다.

성능이 크게 감소하면 그 채널의 실제 과거 이력이 detector에 중요한 추가 정보를 제공했다는 뜻이다.

#### 3단계: 시간구간 교란

- 최근 1개 XMV 시점만 유지
- 최근 5개 유지
- 최근 10개 유지
- XMV 시간 순서를 반전

이를 통해 모델이 단순 최신값만 사용하는지, 더 긴 temporal context를 활용하는지 확인한다.

#### 4단계: architecture consensus

GRU에서 발견된 event-level gain이 TCN과 Transformer에서도 반복되는지 확인한다. 향후에는 개별 XMV channel mapping까지 세 architecture에서 반복되는지 검증한다.

### 5.3 CHUM이 기존 feature importance와 다른 점

일반적인 feature importance는 한 모델의 gradient, attention, SHAP 또는 zero masking 결과를 보여주는 경우가 많다. CHUM은 다음 통제를 함께 요구한다.

- sensor-only baseline
- capacity-matched baseline
- validation-only false-alarm threshold
- 정상분포 기반 conditional replacement
- 여러 training seed
- run/event 단위 통계
- 여러 architecture 간 consensus

즉 “모델이 이 변수를 중요하다고 표시했다”가 아니라 “이 변수를 분포를 존중하며 대체했을 때 실제 탐지 유용성이 재현 가능하게 사라졌다”를 측정하려 한다.

---

## 6. 사용한 데이터와 평가 기준

### 6.1 주 데이터: Reinartz Tennessee Eastman Process

현재 주 실험 데이터는 다음 규모다.

- 2,800개 run
- run당 2,000 sample
- 총 560만 행
- 28개 fault
- 41개 XMEAS
- 사용 가능한 11개 XMV
- fault onset: sample 600

각 fault는 train, validation, test run으로 분리했다. 하나의 run에서 만들어진 window가 서로 다른 split에 섞이지 않도록 run-level split을 사용했다.

### 6.2 평가 단위

- 5개 training seed: 42, 43, 44, 45, 46
- fault당 test run 20개
- 전체 test run 560개
- anomaly threshold: validation normal score의 99번째 percentile
- alarm: threshold를 3개 sample 연속 초과할 때 발생

주요 metric은 다음과 같다.

- `AUROC`: 정상과 fault score의 전체 순위 분리 능력
- `AUPRC`: fault sample이 많은 순위 상단에 위치하는 정도
- detected-run ratio: 실제 run 중 alarm이 발생한 비율
- censored delay: 미탐지를 최장 지연으로 포함한 탐지 지연
- pre-fault FPR: fault가 시작되기 전 잘못 발생한 alarm 비율

### 6.3 외부 검증 데이터: HAI 21.03

HAI는 실제 controller와 hardware-in-the-loop simulator를 결합한 산업제어시스템 데이터다.

- 공식 압축 CSV 8개
- 압축 크기 약 178.6 MiB
- 79개 공정·제어 point
- global attack label과 P1/P2/P3 process별 label
- setpoint, process variable, controller output, valve/pump command, 실제 상태를 구분할 수 있는 기술문서 제공

HAI는 TEP fault와 같은 현상을 주장하기 위한 데이터가 아니다. 다음의 더 일반적인 질문을 검증하는 용도다.

> 물리/HIL 제어 시스템에서도 control-history utility가 attack event별로 불균일한가?

---

## 7. 오늘까지 수행한 실험 과정

### 7.1 Experiment 1: GRU에서 fault별 gain 확인

F0, F1, F0-C를 5개 seed에서 비교했다. strict GAIN fault는 다음과 같았다.

```text
3, 4, 7, 19, 24, 25, 26
```

이는 “모든 fault에서 XMV가 유용하다”는 주장을 기각하고, fault-specific utility라는 연구 문제를 만들었다.

### 7.2 Experiment 2: zero occlusion과 temporal perturbation

다음 16개 조건을 평가했다.

- original
- XMV time reversal
- 최근 1/5/10개 시점만 유지
- XMV01부터 XMV11까지 개별 zero occlusion

5 seed × 16 조건 × 28 fault × 20 run을 평가했다.

초기 결과는 강했지만 일부 no-gain fault의 pre-FPR 차이가 사전 기준 0.005를 넘었다. 따라서 결과를 PASS로 만들기 위해 기준을 완화하지 않고 `MODIFY`로 판정했다.

### 7.3 G1: conditional replacement로 FPR 문제 해결

정상 train 구간으로 학습한 조건부 XMV 값을 사용하자 다음과 같이 결과가 정제됐다.

- 최대 pre-FPR 변화: 0.002
- 0.005를 넘은 조건: 0개
- material effect가 남은 주요 조합:

| Fault | Conditional replacement | 평균 AUROC 손실 | 방향 일치 |
| ---: | --- | ---: | ---: |
| 4 | XMV10 | 0.4191 | 5/5 seed |
| 19 | XMV08 | 0.0289 | 5/5 seed |
| 25 | XMV02 | 0.0217 | 5/5 seed |
| 26 | XMV04 | 0.0840 | 5/5 seed |

반면 fault 7과 24의 개별 channel 효과는 conditional replacement에서 0.02 기준을 넘지 못했다. 따라서 이 두 fault의 zero-occlusion channel attribution은 논문의 주 근거로 사용하지 않는다.

### 7.4 G2: TCN과 Transformer architecture robustness

TCN 15개 모델과 Transformer 15개 모델을 학습하고 전체 test cohort에서 평가했다.

두 architecture가 동일하게 다음 fault를 GAIN으로 분류했다.

```text
4, 7, 19, 23, 24, 25, 26
```

GAIN 기준은 다음을 동시에 요구했다.

- F1의 평균 AUROC 개선이 F0와 F0-C 모두에 대해 0.02 이상
- 평균 AUPRC 개선이 두 대조군 모두에 대해 0.01 이상
- AUROC 방향이 두 대조군 모두에 대해 최소 4/5 seed에서 양수

실제 결과는 다음과 같다.

| Fault | TCN ΔAUROC vs F0-C | Transformer ΔAUROC vs F0-C | Seed 방향 |
| ---: | ---: | ---: | ---: |
| 4 | 0.1663 | 0.1087 | 두 모델 모두 5/5 |
| 7 | 0.4880 | 0.4845 | 두 모델 모두 5/5 |
| 19 | 0.1783 | 0.1970 | 두 모델 모두 5/5 |
| 23 | 0.0300 | 0.0234 | 두 모델 모두 5/5 |
| 24 | 0.1091 | 0.0877 | 두 모델 모두 5/5 |
| 25 | 0.2916 | 0.2776 | 두 모델 모두 5/5 |
| 26 | 0.3868 | 0.4154 | 두 모델 모두 5/5 |

fault 23은 TCN과 Transformer에서는 기준을 통과했지만 최초 strict GRU GAIN에는 포함되지 않았다. 따라서 이를 사후에 GRU gain으로 바꾸지 않고 **architecture-dependent 또는 near-threshold fault**로 별도 보고한다.

---

## 8. 결과를 어떻게 해석해야 하는가

### 8.1 현재 확인된 사실

다음은 데이터와 실험으로 직접 확인한 사실이다.

- XMV를 추가했을 때의 효과는 fault별로 크게 다르다.
- fault 4, 19, 25, 26의 전체 control-history gain은 GRU, TCN, Transformer에서 반복된다.
- TCN과 Transformer의 gain은 파라미터를 맞춘 sensor-only 모델로 설명되지 않는다.
- 정상 조건부 replacement에서도 특정 GRU XMV channel 효과가 5/5 seed에서 유지된다.
- conditional replacement는 zero occlusion보다 pre-FPR shift를 크게 줄였다.

### 8.2 합리적이지만 아직 제한된 해석

다음은 확인된 결과에 기반한 해석이다.

- 일부 fault에서는 controller response 또는 manipulated-variable pattern이 센서만으로는 부족한 추가적인 fault 분리 정보를 제공한다.
- 모델은 단순한 최신 XMV 값뿐 아니라 더 긴 과거 구간을 사용하는 경우가 있다.
- 이 정보효과는 GRU의 recurrent 구조에만 의존하지 않는다.

### 8.3 아직 확인하지 못한 주장

다음은 현재 논문에서 주장하면 안 된다.

- XMV가 fault를 발생시켰다.
- controller action이 인과적으로 탐지를 개선했다.
- 선택된 XMV가 실제 root cause다.
- GRU에서 확인된 XMV10/XMV08/XMV02/XMV04 매핑이 TCN과 Transformer에서도 동일하다.
- TEP 결과가 실제 산업 플랜트에 그대로 일반화된다.

XMV는 fault 이후 controller가 반응한 결과일 수 있다. 따라서 현재 연구는 **causal control effect**가 아니라 **incremental predictive information**을 다룬다.

---

## 9. 이 논문이 제공하는 솔루션

이 논문은 사용자가 새로운 anomaly detector 하나를 반드시 사용해야 한다고 제안하지 않는다. 대신 산업 시계열 모델을 개발하거나 검증할 때 다음 절차를 제공한다.

1. sensor-only와 sensor+control 모델을 비교한다.
2. sensor-only 모델의 용량을 맞춰 파라미터 효과를 제거한다.
3. 전체 평균이 아니라 event별 utility를 계산한다.
4. false-alarm threshold를 validation normal에서 조건별로 다시 맞춘다.
5. zero masking뿐 아니라 정상분포 기반 conditional replacement를 사용한다.
6. channel과 temporal block을 개별 교란한다.
7. 여러 seed와 run-level uncertainty를 평가한다.
8. 서로 다른 architecture에서 반복되는 결과와 특정 모델에만 나타나는 결과를 구분한다.

실무적으로는 다음 질문에 답하는 도구가 된다.

- 이 공정에서 actuator/control history를 수집하고 모델에 넣을 가치가 있는가?
- 어떤 이상 사건에서만 효과가 있는가?
- 어떤 control channel이 실제 detector 성능에 기여하는가?
- 센서 또는 제어 채널이 누락됐을 때 어떤 fault 탐지가 취약해지는가?
- 모델의 attention 또는 gradient 설명을 실제 성능 변화로 검증할 수 있는가?

---

## 10. 이 논문의 novelty는 무엇인가

novelty를 과장하지 않기 위해 세 종류로 구분한다.

### 10.1 새 모델 구조 novelty: 낮음

GRU, TCN, Transformer 자체는 새로운 모델이 아니다. 단순히 이 모델들을 조합하거나 파라미터를 키우는 것을 핵심 novelty로 주장하지 않는다.

### 10.2 방법론 novelty: 중간 이상

CHUM은 다음을 하나의 통합된 protocol로 묶는다.

- event-level control utility
- capacity matching
- matched FPR calibration
- conditional control-history replacement
- temporal block perturbation
- seed/run stability
- architecture consensus

각 요소는 개별적으로 완전히 새로운 개념이 아닐 수 있다. 그러나 산업 이상 탐지에서 **제어 이력의 유용성 자체를 대상으로 이 통제를 일관되게 결합한 audit framework**는 충분히 논문 기여가 될 가능성이 있다. 최종 novelty 주장은 최신 선행연구의 full-text 비교를 통해 문장 수준으로 더 좁혀야 한다.

### 10.3 실증 novelty: 강함

현재 가장 강한 novelty는 다음 실증 결과다.

> 제어 이력의 탐지 유용성은 보편적이지 않고 특정 fault에 집중되며, 그 event-level 구조가 GRU, TCN, Transformer와 capacity-matched control에서 반복된다.

여기에 HAI 외부 검증과 architecture별 channel consensus가 추가되면 다음과 같이 더 강하게 주장할 수 있다.

> 제어 이력의 조건부 유용성은 하나의 데이터셋 또는 모델 구조에 한정된 artifact가 아니며, 시뮬레이션 fault와 HIL control-system attack에서 공통적으로 관찰되는 검증 가능한 현상이다.

### 10.4 석사논문으로서의 가치

이 연구는 다음 이유로 석사논문에 적합하다.

- 명확하고 falsifiable한 연구질문이 있다.
- 단순 성능 향상이 아니라 왜·언제 효과가 나타나는지를 분석한다.
- negative result와 실패 조건을 숨기지 않는다.
- 모델 용량, seed, false alarm, 분포 밖 perturbation이라는 주요 반론을 직접 다룬다.
- 560만 행의 주 데이터와 HIL 외부 데이터를 사용한다.
- 재현 가능한 코드, checkpoint, 사전등록 문서, CSV 결과가 존재한다.
- AI 모델링, explainability, 산업 시계열, 통계적 검증을 하나의 연구로 연결한다.

---

## 11. 현재 논문의 한계

### 11.1 seed 5개의 통계적 해상도

5개 seed를 사용한 양측 exact sign-flip test의 최소 가능한 p-value는 0.0625다. 따라서 모든 seed가 같은 방향이어도 seed-level `p < 0.05`는 수학적으로 만들 수 없다.

이를 숨기거나 일반적인 t-test로 과장하지 않고 다음을 함께 보고한다.

- 효과크기
- 5개 seed 전체 값
- 방향 일치 수
- run-level hierarchical bootstrap
- seed-level 검정의 해상도 한계

최종 논문에서 seed-level `p < 0.05`가 필수라면 핵심 조건에 한해 seed를 추가해야 한다.

### 11.2 conditional replacement는 물리적 counterfactual이 아니다

현재 conditional imputer는 정상 데이터에서 현재 XMEAS와 직전 XMV를 이용해 XMV의 조건부 평균을 예측한다. zero보다 현실적인 값이지만, 실제 controller가 동일 상황에서 반드시 선택할 물리적 action을 생성하는 것은 아니다.

따라서 이를 “정상 조건부 attribution baseline”으로 부르고 “causal intervention”이라고 부르지 않는다.

### 11.3 TEP 데이터의 한계

- simulation benchmark다.
- XMV12가 없다.
- operating mode 정보가 없다.
- 공식 physical mechanism mapping이 충분하지 않다.
- controller reaction과 fault effect를 완전히 분리할 수 없다.

이 한계를 보완하기 위해 HAI 외부 검증을 준비했다.

### 11.4 channel-level architecture consensus 미완료

현재 architecture consensus는 event-level gain에 대해 확인됐다. 개별 XMV channel attribution은 GRU에서만 conditional replacement로 검증됐다.

따라서 다음 단계에서 TCN과 Transformer checkpoint에도 conditional CHUM을 적용해야 한다.

---

## 12. 지도교수님께 설명할 때의 핵심 문장

다음처럼 설명하는 것이 가장 정확하다.

> 기존 산업 이상 탐지 연구는 센서와 제어변수를 함께 넣고 전체 성능을 비교하는 경우가 많습니다. 제 연구는 제어 이력의 효과가 모든 fault에 보편적인지 의문을 제기합니다. 센서-only, sensor+control, capacity-matched sensor-only 모델을 5개 seed에서 비교한 결과, 효과가 특정 fault에 집중됐고 이 구조가 GRU뿐 아니라 TCN과 Transformer에서도 반복됐습니다. 또한 정상분포 기반 conditional replacement를 이용해 어떤 제어 채널이 실제 탐지 성능에 기여하는지 false-alarm shift를 통제하며 분석했습니다. 최종적으로는 이를 CHUM이라는 재현 가능한 control-history utility audit protocol로 정리하고 HAI HIL 데이터에서 외부 검증하려고 합니다.

교수님이 “새로운 모델이 무엇이냐”고 물으면 다음처럼 답한다.

> 새로운 backbone을 제안하는 논문은 아닙니다. 기존 backbone의 성능 경쟁보다, 산업 AI가 control history를 언제 신뢰하고 사용하는지를 검증하는 방법론과 architecture-robust empirical finding이 기여입니다.

교수님이 “그게 그냥 controller reaction 아니냐”고 물으면 다음처럼 답한다.

> 맞습니다. 그래서 causal control effect라고 주장하지 않습니다. controller response를 포함한 과거 XMV가 제공하는 incremental predictive information을 측정하며, root cause와는 구분합니다. 향후 HAI의 SP/PV/CO attack target과 command/state 구분을 이용해 이 경계를 더 분석할 예정입니다.

---

## 13. 제안하는 논문 목차

### 1장. 서론

- 산업 이상 탐지에서 sensor와 control history의 역할
- 단순 feature addition과 pooled performance 비교의 한계
- 연구질문과 기여

### 2장. 관련 연구

- 산업 공정 monitoring과 prediction residual
- TEP 기반 deep fault detection
- controller-aware 또는 input-output monitoring
- 시계열 feature attribution과 perturbation
- GRU, TCN, Transformer 기반 anomaly detection

### 3장. 문제 정의와 CHUM

- sensor/control 시계열 표기
- event-level incremental utility
- conditional replacement
- temporal utility
- architecture consensus
- claim boundary

### 4장. 데이터와 실험설계

- Reinartz TEP
- leakage-safe run split
- F0/F1/F0-C
- threshold와 alarm rule
- 통계 및 decision gate
- HAI point-role schema

### 5장. TEP 실험 결과

- GRU fault heterogeneity
- zero perturbation과 failure mode
- conditional CHUM
- TCN/Transformer robustness
- negative controls와 architecture-dependent fault

### 6장. HAI 외부 검증

- sensor/control/SP/PV/CO role mapping
- attack group별 utility
- 외부 architecture replication
- TEP와 HAI에서 공통되는 현상과 다른 점

### 7장. 논의

- 왜 일부 사건에서만 control history가 유용한가
- predictive information과 causality의 차이
- attribution reliability
- 산업적 의미와 한계

### 8장. 결론

- 확인된 사실
- 방법론 기여
- 일반화 범위
- 후속 연구

---

## 14. 앞으로 해야 할 일

우선순위는 다음과 같다.

1. HAI 79개 point의 공식 role table을 완성한다.
2. HAI train 파일의 attack-label purity, timestamp continuity, missing value, constant channel을 검사한다.
3. HAI에서 sensor-only, sensor+control, capacity-matched control을 3개 seed로 먼저 실행한다.
4. HAI에서 event별 utility heterogeneity가 확인되면 conditional CHUM을 적용한다.
5. TCN과 Transformer checkpoint에 conditional channel replacement를 적용한다.
6. Integrated Gradients 또는 classical contribution method를 attribution baseline으로 추가한다.
7. 핵심 결과가 닫힌 뒤에만 최종 제목과 abstract를 확정한다.
8. 그 후 Method와 Results 장부터 논문 원고를 작성한다.

논문을 빨리 쓰기 위해 지금 결과를 과장하는 것보다, HAI와 channel-level architecture consensus 두 가지를 먼저 닫는 편이 최종 심사에서 훨씬 안전하다.

---

## 15. 용어 정리

| 용어 | 쉬운 의미 |
| --- | --- |
| XMEAS | 공정 상태를 관측하는 센서값 |
| XMV | 공정을 조절하기 위해 사용된 manipulated variable 이력 |
| Fault | 공정에 주입된 이상 또는 고장 조건 |
| Residual | 모델의 예측값과 실제 센서값의 차이 |
| F0 | 센서만 사용하는 예측 모델 |
| F1 | 센서와 제어 이력을 함께 사용하는 예측 모델 |
| F0-C | F1과 파라미터 수를 맞춘 센서-only 대조군 |
| Occlusion | 입력 변수 일부를 지워 중요도를 측정하는 방법 |
| Conditional replacement | 지우는 대신 정상 조건에서 예상되는 값으로 교체하는 방법 |
| Utility | 해당 정보가 실제 탐지 성능에 추가로 제공하는 가치 |
| Architecture consensus | 서로 다른 모델 구조에서 같은 현상이 반복되는 상태 |
| Capacity control | 모델 크기 증가 효과를 분리하기 위한 대조 실험 |
| FPR | 정상인데 이상이라고 잘못 판단한 비율 |
| Seed | 모델 초기값과 학습 순서의 무작위성을 통제하기 위한 반복 번호 |

---

## 16. 근거 자료와 재현 산출물

이 문서의 수치와 판단은 다음 저장 산출물에 근거한다.

- `outputs/final_gate_exp1/FINAL_GATE_EXP1_REPORT.md`
- `outputs/mechanism_gate_exp2/MECHANISM_GATE_EXP2_REPORT.md`
- `outputs/mechanism_gate_exp2/G1_STATISTICAL_SUMMARY.csv`
- `outputs/conditional_chum_g1/CONDITIONAL_CHUM_G1_REPORT.md`
- `outputs/conditional_chum_g1/CONDITIONAL_CHUM_RESULTS.csv`
- `outputs/architecture_gate_g2_full/ARCHITECTURE_GATE_G2_REPORT.md`
- `outputs/architecture_gate_g2_full/G2_METRICS.csv`
- `outputs/architecture_gate_g2_full/G2_FAULT_RESULTS.csv`
- `outputs/architecture_gate_g2_full/G2_FAULT_SUMMARY.csv`
- `outputs/methodology/THESIS_DIRECTION_V2.md`
- `outputs/methodology/HAI_21_03_EXTERNAL_VALIDATION_AUDIT.md`

외부 데이터 및 선행근거:

- HAI 공식 저장소: https://github.com/icsdataset/hai
- SWaT 공식 데이터 안내: https://www.sutd.edu.sg/itrust/itrust-labs/datasets/dataset-characteristics/swat/
- Controller-model/LSTM-AE monitoring 연구: https://doi.org/10.1021/acs.iecr.4c01980
- TEP temporal deep-learning 비교 연구: https://doi.org/10.1016/j.jii.2021.100216

---

## 17. 최종 현재 판단

현재 연구는 **단순 GRU 성능 비교 단계는 넘어섰다.**

확인된 핵심은 다음과 같다.

> 과거 제어 이력이 제공하는 이상 탐지 정보는 fault별로 강하게 다르며, 이 event-level 구조는 GRU, TCN, Transformer와 capacity-matched 대조군에서 반복된다.

논문의 가장 적절한 기여 형태는 다음 두 가지의 결합이다.

1. **방법론 기여:** CHUM이라는 FPR·capacity·distribution·architecture를 통제한 control-history utility audit protocol
2. **실증 기여:** 제어 이력의 유용성이 보편적이지 않고 특정 fault에 집중되며 여러 시계열 architecture에서 재현된다는 결과

HAI 외부 검증과 channel-level architecture consensus가 완료되면, 석사논문으로서 novelty와 방어력은 현재보다 한 단계 더 강해질 것이다.

# 유튜브 분야 상승 예측 프로젝트

![status](https://img.shields.io/badge/status-final%20presentation-2F5D62)
![model](https://img.shields.io/badge/model-BiGRU-355E63)
![task](https://img.shields.io/badge/task-Top--N%20category%20forecast-C2A88D)
![scope](https://img.shields.io/badge/scope-core%2010%20categories-7DA39D)

유튜브 분야별 과거 흐름, 최근 12주 시계열 반응, Google Trends 검색량, 캘린더 변수를 함께 활용해 **핵심 10개 분야 중 앞으로 4주 동안 상승 가능성이 높은 Top-5 분야를 예측**하려 한 딥러닝 프로젝트다.

이 저장소에는 최종 발표 자료뿐 아니라,  
문제를 어떻게 다시 정의했는지, 데이터 정합성 문제를 어떻게 고쳤는지, 그리고 왜 최종적으로 BiGRU 기반 구조를 선택했는지까지 함께 정리해두었다.

> 최종 발표 자료: [딥러닝 발표 PPT.pdf](./딥러닝%20발표%20PPT.pdf)
>
> 예상 질문 표: [QnA_Report.pdf](./QnA_Report.pdf)
>
> 발표용 답변 가이드: [PRESENTATION_QNA.md](./PRESENTATION_QNA.md)
>
> English readers can start with [PROJECT_JOURNEY_EN.md](./PROJECT_JOURNEY_EN.md), [FINAL_MODEL.md](./FINAL_MODEL.md), and [PRESENTATION_QNA_EN.md](./PRESENTATION_QNA_EN.md).

## 한눈에 보기

- **문제**: 유튜브 핵심 10개 분야 중 어떤 분야가 앞으로 4주 동안 더 상승할까?
- **핵심 아이디어**: 개별 영상만 보지 않고, **카테고리 흐름 + 최근 12주 시계열 + 외부 변수**를 함께 본다.
- **최종 모델**: RNN 계열의 표준 시계열 딥러닝 모델인 **BiGRU**
- **최종 목표**: 단순 조회수 예측이 아니라 **Top-5 상승 분야 선별**

## 프로젝트 개요 및 목표

이 프로젝트의 출발점은 단순했다.

“현재 조회수가 높은 분야”를 보는 것보다,  
**앞으로 상승할 가능성이 높은 분야를 미리 파악할 수 없을까?**

최종 발표에서는 아래 세 가지를 중심 목표로 정리했다.

1. **미래 상승 분야 예측**
   - 과거 카테고리 흐름과 최근 시계열 반응을 결합해, 향후 4주 동안 상승 가능성이 높은 분야를 예측한다.
2. **Top-N 선별 문제로 재정의**
   - 모든 분야를 동일하게 맞히는 문제보다, 실제로 주목할 분야를 상위권으로 골라내는 문제에 집중했다.
3. **카테고리 맥락 반영**
   - 영상 반응을 개별 영상만의 문제가 아니라, 해당 카테고리의 흐름 안에서 해석한다.

## 문제 정의 및 필요성

기존 virality 예측은 보통 개별 영상의 조회수, 좋아요, 댓글, 제목 정보에 더 집중한다.  
하지만 실제 영상 성과는 영상 자체만으로 결정되지 않고, **그 영상이 속한 카테고리 전체의 관심도 흐름**에도 크게 영향을 받는다.

그래서 이 프로젝트는 질문을 바꿨다.

- “어떤 영상이 잘 될까?”를 바로 묻기보다
- **“앞으로 어떤 분야가 더 관심을 받을까?”를 먼저 예측하고**
- 영상 반응은 그 위에서 보조적으로 활용하는 방식으로 접근했다.

## 데이터 구성 및 분석

발표 자료 기준으로 데이터는 세 흐름으로 정리된다.

1. **YouTube Data API 기반 영상 데이터**
   - 제목, 채널, 업로드 시점, 태그, 카테고리
   - 조회수, 좋아요, 댓글 등 반응 지표
2. **과거 카테고리 추세 데이터**
   - 2020~2025년 주차별 분야 흐름
   - 장기적인 변화 패턴 반영
3. **최근 시계열 및 외부 변수 데이터**
   - 최근 12주 카테고리 반응 추세
   - Google Trends 검색량
   - 공휴일, 연휴, 방학, 시험기간 등 캘린더 변수

최종 데이터 규모는 다음과 같다.

- 전체 통합 영상: 18,958개
- 중복 제거 영상: 16,641개
- 주간 추세 데이터: 1,711행
- 시계열 지원 데이터: 5,770행

최종 학습 대상은 활동이 충분한 16개 분야였고,  
발표와 최종 평가는 핵심 10개 분야를 기준으로 진행했다.

## 예측 모델 설계

발표에서 설명한 최종 구조는 크게 네 단계다.

1. **데이터 입력**
   - 과거 카테고리 성과 데이터
   - 최근 12주 카테고리 반응 추세
   - Google Trends 및 캘린더 변수
2. **Category Trend Model**
   - 카테고리별 과거 흐름과 최근 모멘텀을 읽는다.
3. **BiGRU 기반 시계열 모델**
   - RNN → GRU → BiGRU 흐름 위에서 최근 12주 패턴을 양방향으로 학습한다.
   - category embedding을 함께 사용해 분야별 고유 특성을 반영한다.
4. **Final Prediction**
   - 상승 확률과 순위 상승 확률을 함께 산출하고, 이를 바탕으로 최종 Top-N을 선별한다.

핵심은 “복잡한 딥러닝 모델을 쓰는 것”보다,  
**카테고리 흐름을 먼저 읽고, 영상·시계열·외부 변수를 그 위에서 결합하는 구조**를 세우는 데 있었다.

## 핵심 파생변수 전략

발표 자료에서는 아래 변수들을 핵심으로 설명했다.

- **avg_virality**: 현재 반응 규모
- **engagement_rate**: 참여도 지표
- **rolling_4week_mean**: 최근 4주 평균 반응 수준
- **momentum_ratio**: 최근 반응 가속도
- **competition_score**: 분야 내 경쟁 강도
- **opportunity_score**: 상승 가능성 지표
- **rise_label**: 향후 4주 상승 여부
- **rank_up**: 향후 순위 상승 여부

즉, 이 프로젝트는 단순 규모 비교가 아니라  
**최근 변화 방향과 상대적 상승 신호**를 수치화하는 방향으로 파생변수를 설계했다.

## 최종 성능 및 예측 결과

핵심 10개 분야 기준 최종 BiGRU 모델 성능은 아래와 같다.

### 분류 성능

- Accuracy: `0.767`
- Balanced Accuracy: `0.798`
- Precision: `0.944`
- Recall: `0.739`
- F1-score: `0.829`
- ROC AUC: `0.845`

### Top-5 선별 성능

- Precision@5: `0.900`
- Recall@5: `0.801`
- HitRate@5: `1.000`
- NDCG@5: `0.881`

### 최종 Top-5 상승 예측 분야

1. 반려동물
2. 먹방
3. 경제
4. 브이로그
5. 교육

이 결과는 단순 조회수 크기가 아니라,  
상승 확률, 순위 상승 확률, 최근 반응 추세, 검색 관심도, 상대 순위 신호를 함께 반영한 결과다.

## 이 프로젝트가 의미하는 것

이 프로젝트가 남기는 중요한 결론은 단순하다.

- 유튜브 분야의 상승 가능성은 **단순 규모만으로 설명되지 않는다**
- 장기 흐름, 최근 반응 속도, 외부 관심도, 캘린더 요인을 함께 봐야 한다
- 딥러닝 모델 자체보다 **문제를 어떻게 재정의하고 데이터를 어떻게 맞췄는지**가 성능과 해석에 큰 영향을 준다

즉, 이 저장소는 “모델 하나를 돌려본 결과”보다  
**상승 분야 예측 문제를 실제로 설계하고, 정리하고, 구현해 본 과정**을 남긴 기록에 더 가깝다.

## 프로젝트 스토리

이 저장소는 결과만 정리한 곳이 아니라,  
**어떤 질문에서 출발했고, 어디서 막혔고, 무엇을 고쳤고, 결국 무엇을 남겼는지**를 함께 기록한 프로젝트다.

- 한국어 스토리: [PROJECT_JOURNEY.md](./PROJECT_JOURNEY.md)
- English story: [PROJECT_JOURNEY_EN.md](./PROJECT_JOURNEY_EN.md)

두 문서는 같은 프로젝트를 설명하지만,  
한국어 문서는 발표와 작업 맥락을 더 자세히 담고 있고, 영어 문서는 외부 사람이 저장소만 보고도 이해할 수 있도록 다시 썼다.

## 발표 Q&A 자료

최종 발표 자료와 함께 질의응답 대비 자료도 같이 남겨두었다.

- [QnA_Report.pdf](./QnA_Report.pdf): 최종 발표 기준 예상 질문 표
- [PRESENTATION_QNA.md](./PRESENTATION_QNA.md): 코드와 최종 성능 수치를 대조해 다시 정리한 한국어 답변 가이드
- [PRESENTATION_QNA_EN.md](./PRESENTATION_QNA_EN.md): 영어권 독자를 위한 짧은 FAQ 정리

특히 `PRESENTATION_QNA.md`는 원본 표에 있던 표현을 그대로 옮기지 않고,  
실제 최종 모델 설정과 실험 결과에 맞게 다시 다듬은 버전이다.

## 저장소 안의 추가 자료

- [PROJECT_JOURNEY.md](./PROJECT_JOURNEY.md): 한국어 프로젝트 일대기
- [PROJECT_JOURNEY_EN.md](./PROJECT_JOURNEY_EN.md): English project story
- [QnA_Report.pdf](./QnA_Report.pdf): 최종 발표용 예상 질문 표
- [PRESENTATION_QNA.md](./PRESENTATION_QNA.md): 발표용 답변 가이드
- [PRESENTATION_QNA_EN.md](./PRESENTATION_QNA_EN.md): English FAQ
- [TROUBLESHOOTING.md](./TROUBLESHOOTING.md): 데이터와 실험 과정에서 겪은 문제
- [FINAL_MODEL.md](./FINAL_MODEL.md): BiGRU 기반 최종 모델 설명
- [RESULTS.md](./RESULTS.md): 최종 성능과 예측 결과 정리
- [RUN_PROJECT.md](./RUN_PROJECT.md): 실행 안내
- [REPRODUCIBILITY.md](./REPRODUCIBILITY.md): 재현 범위와 한계

## 공개 저장소 구조

```text
.
├─ 딥러닝 발표 PPT.pdf                           # 최종 발표 PDF
├─ QnA_Report.pdf                                # 최종 발표용 예상 질문 표
├─ youtube_trend_project_pipeline.executed.ipynb # 메인 프로젝트 노트북
├─ train_active_category_rank_bigru.py           # 최종 BiGRU 학습 코드
├─ run_core10_top_predictions.py                 # 최종 Top-5 예측 재생성 코드
├─ make_paper_visualization_suite.py             # 논문형 시각화 생성 코드
├─ PROJECT_JOURNEY.md
├─ PROJECT_JOURNEY_EN.md
├─ TROUBLESHOOTING.md
├─ FINAL_MODEL.md
├─ RESULTS.md
├─ RUN_PROJECT.md
├─ REPRODUCIBILITY.md
├─ requirements.txt
└─ docs/assets/
```

## 어디부터 보면 좋은가

처음 보는 사람에게는 아래 순서를 추천한다.

1. 이 `README.md`
2. [딥러닝 발표 PPT.pdf](./딥러닝%20발표%20PPT.pdf)
3. [QnA_Report.pdf](./QnA_Report.pdf)
4. [PRESENTATION_QNA.md](./PRESENTATION_QNA.md)
5. [PROJECT_JOURNEY.md](./PROJECT_JOURNEY.md)
6. [FINAL_MODEL.md](./FINAL_MODEL.md)
7. [RESULTS.md](./RESULTS.md)
8. 외부 공유용이면 [PROJECT_JOURNEY_EN.md](./PROJECT_JOURNEY_EN.md)

## 빠르게 실행하고 싶다면

```bash
pip install -r requirements.txt
```

그다음 아래 문서를 보는 편이 가장 안전하다.

- 실행 방법: [RUN_PROJECT.md](./RUN_PROJECT.md)
- 재현 범위: [REPRODUCIBILITY.md](./REPRODUCIBILITY.md)

## 한 줄 요약

이 저장소는 **유튜브 핵심 10개 분야의 향후 4주 상승 가능성을 최근 12주 시계열, 외부 변수, BiGRU 기반 딥러닝 모델로 예측해본 프로젝트의 최종 발표 자료와 그 뒤의 실험 기록을 함께 남겨둔 저장소**다.

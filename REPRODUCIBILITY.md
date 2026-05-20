# REPRODUCIBILITY

이 문서는 이 저장소를 받은 사람이 **어디까지 재현할 수 있는지**와, **어디부터는 원본 산출물이나 공유 패키지가 필요한지**를 더 엄밀하게 설명합니다.

## 1. 재현 가능한 수준

이 저장소는 아래 세 단계로 이해하는 것이 가장 안전합니다.

### 수준 A. 문서/결과 확인

아무 것도 다시 생성하지 않고, 아래 파일만 읽어도 프로젝트의 핵심은 이해할 수 있습니다.

- `README.md`
- `PROJECT_JOURNEY.md`
- `TROUBLESHOOTING.md`
- `FINAL_MODEL.md`
- `RESULTS.md`
- `youtube_trend_project_pipeline.executed.ipynb`

이 수준은 **가장 안정적**이며, 외부 공유 시에도 가장 먼저 권장되는 방식입니다.

### 수준 B. 최종 예측/시각화 재생성

이미 정리된 `project_ready_data`가 로컬에 존재한다면 아래 코드를 다시 실행할 수 있습니다.

- `python run_core10_top_predictions.py`
- `python make_paper_visualization_suite.py`

이 수준은 **최종 결과 확인과 발표용 그림 재생성**에 적합합니다.

### 수준 C. 완전한 원천 데이터 재구성

이 저장소는 발표/회고 중심 버전이기 때문에, 원천 수집 단계부터 완전히 다시 수행하는 것은 공개 저장소 단독으로는 제한이 있습니다.

이유:
- 원천 수집에는 외부 API와 수집 시점 의존성이 있음
- Google Trends는 live feed가 아니라 복구/정규화된 캐시 기반 파일을 사용
- 일부 대용량 산출물은 Git에서 제외될 수 있음

즉, 이 저장소는 **완전한 수집 재현**보다 **최종 파이프라인과 결과 해석 재현**에 더 적합합니다.

## 2. 필요한 환경

- Python 3.11+
- Windows 환경 기준 정리
- GPU는 있으면 좋지만 필수는 아님

## 3. 설치

기본 설치:

```bash
pip install -r requirements.txt
```

참고:
- `requirements.txt`가 공개용 기준 파일입니다.
- `requirements-youtube.txt`는 과거 로컬 실행 흐름을 유지하기 위한 **legacy compatibility file** 입니다.

## 4. 어떤 파일이 있어야 하는가

### 문서/결과 확인만 할 때

필수:
- Markdown 문서들
- 메인 노트북
- `docs/assets/` 안 대표 그림

### 최종 예측과 시각화를 다시 만들 때

필수:
- `project_ready_data/external_features/`
- `project_ready_data/model_outputs/`
- `project_ready_data/ready_category_weekly_trend.csv`

즉, `project_ready_data/`가 없으면 **최종 재생성 코드 전체가 바로 돌지는 않을 수 있습니다.**

## 5. 공개 저장소에서 바로 안 될 수 있는 부분

아래는 저장소만 클론해서는 바로 완전히 재현되지 않을 수 있는 요소입니다.

- 대용량 산출물이 `.gitignore`에 의해 제외된 경우
- `project_ready_data/` 전체가 없는 경우
- 외부 API나 live Trends 수집이 막히는 경우
- 폰트/경로 차이로 일부 그림 레이아웃이 달라지는 경우

## 6. 가장 안전한 사용 전략

1. 먼저 `README.md`와 `RESULTS.md`로 전체 개요를 파악한다.
2. `FINAL_MODEL.md`로 최종 모델 구조를 이해한다.
3. `PROJECT_JOURNEY.md`와 `TROUBLESHOOTING.md`로 실제 문제 해결 과정을 읽는다.
4. 전체 산출물이 필요하면 `프로젝트-공유패키지`를 함께 확인한다.
5. 로컬에 `project_ready_data`가 준비되어 있을 때만 최종 예측과 시각화 재생성을 시도한다.

## 7. 한 줄 요약

이 저장소는 **원천 수집부터 100% 다시 만드는 완전 재현 저장소**라기보다, **정리된 데이터와 결과를 바탕으로 최종 파이프라인, 모델, 시각화, 발표 스토리를 재현하는 저장소**로 보는 것이 가장 정확합니다.

# 재현 범위

이 문서는 저장소를 받은 사람이 **어디까지 재현할 수 있는지**, 그리고 **어디부터는 로컬 원본 산출물이 필요한지**를 조금 더 엄밀하게 설명한다.

## 1. 재현 가능한 범위

이 저장소는 아래 세 단계로 이해하는 편이 가장 안전하다.

### 단계 A. 문서와 결과 확인

아무것도 다시 만들지 않고, 아래 파일만 읽어도 프로젝트의 전체 흐름은 충분히 이해할 수 있다.

- `README.md`
- `PROJECT_JOURNEY.md`
- `TROUBLESHOOTING.md`
- `FINAL_MODEL.md`
- `RESULTS.md`
- `youtube_trend_project_pipeline.executed.ipynb`

이 단계는 **가장 안정적**이고, 팀 공유나 발표 준비에서는 보통 여기서부터 시작하는 것이 좋다.

### 단계 B. 최종 예측과 시각화 재생성

로컬에 `project_ready_data`가 준비되어 있다면 아래 코드를 다시 실행할 수 있다.

- `python run_core10_top_predictions.py`
- `python make_paper_visualization_suite.py`

이 단계는 **최종 결과 확인과 발표용 그림 재생성**에 해당한다.

### 단계 C. 전체 수집 파이프라인 재현

공개 저장소만으로는 이 단계까지 완전하게 재현하기 어렵다.

이유는 아래와 같다.

- 원본 수집은 외부 API 상태와 수집 시점에 영향을 받는다.
- Google Trends는 live feed가 아니라 복구·정규화된 캐시 파일을 함께 사용했다.
- 대용량 산출물과 일부 로컬 데이터는 Git에서 제외되어 있다.

즉 이 저장소는 **원본 수집 파이프라인 전체를 100% 재현하는 저장소**라기보다,  
**최종 문제 정의와 결과를 이해하고 필요한 범위 안에서 다시 확인하는 저장소**에 가깝다.

## 2. 필요한 환경

- Python 3.11+
- Windows 환경 기준 정리
- GPU가 있으면 좋지만 필수는 아님

## 3. 설치

```bash
pip install -r requirements.txt
```

참고:

- `requirements.txt`가 공개용 기준 파일이다.
- 공개 저장소 설명은 이 파일을 기준으로 맞춰 두었다.

## 4. 어떤 파일이 있어야 하는가

### 문서와 결과만 확인할 경우

필수:

- Markdown 문서
- 메인 노트북
- `docs/assets/` 안 대표 그림

### 최종 예측과 시각화를 다시 생성할 경우

필수:

- `project_ready_data/ready_category_weekly_trend.csv`
- `project_ready_data/external_features/`
- `project_ready_data/model_outputs/`

즉 `project_ready_data/`가 없으면 **최종 재생성 코드 전체가 바로 동작하지 않을 수 있다.**

## 5. 공개 저장소만으로 바로 안 되는 부분

아래는 공개 저장소만으로는 바로 안 될 수 있는 항목들이다.

- Git에서 제외된 대용량 산출물이 필요한 경우
- `project_ready_data/` 전체가 빠진 경우
- 외부 API 상태에 의존하는 재수집 작업
- 로컬 폴더 구조나 폰트 차이로 시각화 레이아웃이 조금 달라지는 경우

## 6. 가장 안전한 사용 순서

1. `README.md`와 `RESULTS.md`로 전체 개요를 본다.
2. `FINAL_MODEL.md`로 최종 모델 구조를 본다.
3. `PROJECT_JOURNEY.md`와 `TROUBLESHOOTING.md`로 실제 문제 해결 흐름을 본다.
4. 전체 산출물이 필요하다면 공개 저장소만으로는 부족할 수 있으며, 로컬 작업 폴더에 남아 있는 원본 산출물 유무를 먼저 확인한다.
5. 로컬에 `project_ready_data`가 준비된 경우에만 예측/시각화 재생성을 시도한다.

## 7. 한 줄 정리

이 저장소는 **원본 수집 전 과정을 100% 다시 만드는 저장소**라기보다,  
**최종 문제 정의, 모델, 결과, 시각화를 중심으로 프로젝트를 이해하고 필요한 범위 안에서 다시 실행해볼 수 있는 저장소**다.

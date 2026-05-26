# 실행 가이드

이 문서는 저장소를 처음 받은 사람이 **무엇을 어디까지 재현할 수 있는지**를 빠르게 이해하도록 정리한 실행 안내다.

## 1. 먼저 구분할 것: 두 가지 사용 방식

### A. 결과 확인 중심

이미 만들어진 결과와 문서를 읽는 방식이다.

- 최종 발표 자료: `딥러닝 발표 PPT.pdf`
- 메인 노트북: `youtube_trend_project_pipeline.executed.ipynb`
- 최종 성능 문서: `RESULTS.md`
- 최종 모델 설명: `FINAL_MODEL.md`

이 방식은 데이터를 다시 수집하지 않아도 프로젝트 전체 흐름을 파악할 수 있어서 가장 먼저 권장한다.

### B. 코드 실행 중심

최종 예측과 시각화를 다시 생성해보는 방식이다.

- 최종 예측 재생성: `run_core10_top_predictions.py`
- 논문형 시각화 재생성: `make_paper_visualization_suite.py`

이 방식은 로컬에 `project_ready_data`가 준비되어 있을 때를 전제로 한다.

공개 저장소만 단독으로 받은 경우에는 `project_ready_data`가 비어 있거나 일부만 포함되어 있을 수 있으니,  
그럴 때는 결과 확인 중심으로 먼저 읽고, 재생성이 필요하면 로컬 작업 폴더에 원본 산출물이 있는지부터 확인하는 편이 안전하다.

재현 범위와 한계는 [REPRODUCIBILITY.md](./REPRODUCIBILITY.md)에 따로 정리했다.

## 2. 권장 환경

- Python 3.11+
- Windows 환경 기준 정리
- GPU가 있으면 더 빠르지만 필수는 아님

## 3. 설치

```bash
pip install -r requirements.txt
```

## 4. 가장 먼저 볼 파일

1. `README.md`
2. `딥러닝 발표 PPT.pdf`
3. `PROJECT_JOURNEY.md`
4. `FINAL_MODEL.md`
5. `youtube_trend_project_pipeline.executed.ipynb`
6. `RESULTS.md`

## 5. 최종 예측 결과 다시 만들기

```bash
python run_core10_top_predictions.py
```

생성되거나 갱신되는 주요 파일:

- `project_ready_data/model_outputs/dl_core10_category_rank_bigru_top_categories.csv`
- `project_ready_data/model_outputs/dl_core10_category_rank_bigru_future_probs.csv`
- `project_ready_data/model_outputs/dl_core10_category_rank_bigru_reproduced_test_metrics.json`

## 6. 논문형 시각화 다시 만들기

```bash
python make_paper_visualization_suite.py
```

대표적으로 아래 그림들이 갱신된다.

- `paper_eda_core10_recent_trends.png`
- `paper_eda_core10_intensity_heatmap.png`
- `paper_results_core10_performance.png`
- `paper_results_core10_prediction_rank_heatmap.png`

## 7. 실행할 때 주의할 점

- 이 저장소에는 실험 중간 산출물과 최종 산출물의 흔적이 함께 남아 있다.
- 처음에는 `core10`으로 시작하는 최종 결과 파일을 우선 보는 편이 덜 헷갈린다.
- 공개 저장소에는 대용량 생성 결과가 모두 포함되지 않을 수 있다.
- 따라서 문서 이해와 결과 확인은 저장소만으로 가능하지만, 전체 재생성은 로컬 산출물 유무에 따라 달라질 수 있다.
- 재현보다 결과 이해가 목적이라면 `README.md → 딥러닝 발표 PPT.pdf → PROJECT_JOURNEY.md → FINAL_MODEL.md` 순서로 보는 것이 가장 빠르다.
- 공개 저장소에서 가능한 범위와 불가능한 범위는 [REPRODUCIBILITY.md](./REPRODUCIBILITY.md)에 정리해두었다.

## 8. 가장 안전한 읽는 순서

1. `README.md`
2. `PROJECT_JOURNEY.md`
3. `TROUBLESHOOTING.md`
4. `FINAL_MODEL.md`
5. `RESULTS.md`
6. `REPRODUCIBILITY.md`
7. `youtube_trend_project_pipeline.executed.ipynb`

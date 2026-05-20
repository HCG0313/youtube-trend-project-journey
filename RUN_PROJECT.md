# RUN PROJECT

이 문서는 이 저장소를 처음 받은 사람이 **무엇을 어디까지 재현할 수 있는지**를 빠르게 이해할 수 있도록 정리한 실행 안내입니다.

## 1. 재현 수준

이 저장소는 크게 두 가지 방식으로 볼 수 있습니다.

### A. 결과 확인 중심

이미 생성된 결과와 문서를 확인하는 방식입니다.

- 메인 노트북: `youtube_trend_project_pipeline.executed.ipynb`
- 최종 성능 문서: `RESULTS.md`
- 최종 모델 설명: `FINAL_MODEL.md`
- 발표 자료: `review_deeplearning_proposal.pptx`

이 방식은 전체 데이터 재수집 없이도 프로젝트 흐름을 가장 빠르게 이해할 수 있습니다.

### B. 코드 실행 중심

최종 예측과 시각화를 다시 생성해 보는 방식입니다.

- 최종 예측 재생성: `run_core10_top_predictions.py`
- 논문형 시각화 재생성: `make_paper_visualization_suite.py`

이 방식은 이미 준비된 `project_ready_data` 구조를 전제로 합니다.
즉, 공개 저장소만 단독으로 받은 경우에는 `project_ready_data`가 비어 있거나 아예 없을 수 있으므로, 이때는 결과 확인 중심으로 먼저 읽고 전체 산출물은 `프로젝트-공유패키지`를 함께 확인하는 것이 안전합니다.
더 엄밀한 재현 범위는 [REPRODUCIBILITY.md](./REPRODUCIBILITY.md) 문서를 같이 확인하는 것이 좋습니다.

## 2. 권장 환경

- Python 3.11+
- Windows 환경 기준
- GPU가 있으면 좋지만 필수는 아님

## 3. 필수 패키지 설치

```bash
pip install -r requirements.txt
```

## 4. 가장 먼저 볼 파일

1. `README.md`
2. `youtube_trend_project_pipeline.executed.ipynb`
3. `FINAL_MODEL.md`
4. `RESULTS.md`

## 5. 최종 예측 결과 다시 생성

```bash
python run_core10_top_predictions.py
```

생성/갱신되는 대표 파일:

- `project_ready_data/model_outputs/dl_core10_category_rank_bigru_top_categories.csv`
- `project_ready_data/model_outputs/dl_core10_category_rank_bigru_future_probs.csv`
- `project_ready_data/model_outputs/dl_core10_category_rank_bigru_reproduced_test_metrics.json`

## 6. 논문형 시각화 다시 생성

```bash
python make_paper_visualization_suite.py
```

대표적으로 아래 그림들이 갱신됩니다.

- `paper_eda_core10_recent_trends.png`
- `paper_eda_core10_intensity_heatmap.png`
- `paper_results_core10_performance.png`
- `paper_results_core10_prediction_rank_heatmap.png`

## 7. 주의할 점

- 이 저장소는 실험 중간 산출물과 최종 산출물이 함께 존재합니다.
- 처음에는 `core10`으로 시작하는 최종 결과 파일을 우선 보는 것이 좋습니다.
- 공개 저장소에 대용량 생성 결과가 모두 포함되지 않을 수 있으므로, 전체 산출물이 필요하면 `프로젝트-공유패키지`를 함께 확인해야 합니다.
- 완전 재현보다 “결과 확인”이 먼저라면 `README.md → RESULTS.md → FINAL_MODEL.md → 메인 노트북` 순으로 보는 것이 가장 덜 헷갈립니다.
- 공개 저장소만으로 가능한 범위와 불가능한 범위는 [REPRODUCIBILITY.md](./REPRODUCIBILITY.md)에 따로 정리되어 있습니다.

## 8. 가장 안전한 읽는 순서

1. `README.md`
2. `PROJECT_JOURNEY.md`
3. `TROUBLESHOOTING.md`
4. `FINAL_MODEL.md`
5. `RESULTS.md`
6. `REPRODUCIBILITY.md`
7. `youtube_trend_project_pipeline.executed.ipynb`

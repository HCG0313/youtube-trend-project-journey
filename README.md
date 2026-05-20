# 유튜브 트렌드 분야 예측 프로젝트

과거 카테고리 추세, 최근 12주 시계열 반응, Google Trends 검색량, 캘린더 변수를 결합해 **유튜브 핵심 10개 분야의 향후 4주 상승 가능성**을 예측한 딥러닝 프로젝트입니다.

이 저장소는 단순히 “최종 성능이 얼마였다”를 정리한 결과 보관소가 아니라, **데이터 문제를 어떻게 발견하고 수정했는지까지 남기는 경험 기록형 저장소**를 목표로 합니다.

## 한눈에 보기

- **문제**: 어떤 유튜브 분야가 앞으로 더 상승할 것인가?
- **최종 목표**: 핵심 10개 분야 안에서 **Top-5 상승 분야 선별**
- **최종 모델**: `BiGRU` 기반 시계열 딥러닝 모델
- **최종 예측 기간**: 다음 4주
- **최종 예측 대상**: `게임`, `경제`, `교육`, `뉴스시사`, `먹방`, `반려동물`, `뷰티`, `브이로그`, `요리`, `운동`

## 이 프로젝트가 흥미로운 이유

이 프로젝트는 모델 성능보다도 **문제 정의와 데이터 정합성 수정**이 더 중요했던 사례입니다.

- Google Trends 수집 실패를 복구했고
- `%Y-%U`와 ISO week 충돌 문제를 수정했고
- 전체 20개 분야 예측에서 핵심 10개 분야 Top-5 선별 문제로 다시 정의했고
- 최종적으로 BiGRU 기반 모델로 발표 가능한 수준까지 정리했습니다

즉, 이 저장소는 결과와 함께 **문제를 어떻게 다루고 고쳐 나갔는지**를 보여줍니다.

## 최종 성능

핵심 10개 분야 기준 최종 성능은 다음과 같습니다.

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

이 결과는 모든 유튜브 분야를 맞히는 범용 모델이라기보다, **핵심 10개 분야 안에서 앞으로 상승할 가능성이 높은 분야를 고르는 모델**로 해석하는 것이 가장 적절합니다.

## 최종 Top-5 상승 예측 분야

1. 반려동물  
2. 먹방  
3. 경제  
4. 브이로그  
5. 교육

이 순위는 단순 인기 순위가 아니라, 상승 확률과 순위 상승 확률을 결합한 `final_score` 기준입니다.

## 빠르게 이해하는 방법

- **프로젝트 흐름부터 보고 싶다면**: [PROJECT_JOURNEY.md](./PROJECT_JOURNEY.md)
- **최종 모델 구조부터 보고 싶다면**: [FINAL_MODEL.md](./FINAL_MODEL.md)
- **최종 성능과 예측 결과부터 보고 싶다면**: [RESULTS.md](./RESULTS.md)
- **실행/재현 방법이 필요하다면**: [RUN_PROJECT.md](./RUN_PROJECT.md)
- **재현 가능 범위와 한계를 정확히 알고 싶다면**: [REPRODUCIBILITY.md](./REPRODUCIBILITY.md)

## 대표 시각화

### 1. 핵심 10개 분야 최근 추세

![핵심 10개 분야 최근 추세](./docs/assets/paper_eda_core10_recent_trends.png)

최근 2년 구간에서 분야별 반응 흐름이 동일하지 않음을 보여줍니다.  
즉 유튜브 분야 변화는 전체 평균이 아니라 **카테고리별 시계열 패턴**으로 봐야 한다는 점을 확인했습니다.

### 2. 최종 예측 결과 요약

![최종 예측 결과](./docs/assets/paper_results_core10_prediction_rank_heatmap.png)

최종 예측 결과는 단순 조회수 규모가 아니라, 최근 반응, 검색량, 순위 상승 신호를 함께 반영한 결과입니다.

## 저장소 구조

```text
.
├─ youtube_trend_project_pipeline.executed.ipynb   # 메인 프로젝트 노트북
├─ 1.ipynb                                         # EDA 보조 노트북
├─ train_active_category_rank_bigru.py             # 최종 BiGRU 학습 코드
├─ run_core10_top_predictions.py                   # 최종 예측 재생성 코드
├─ review_deeplearning_proposal.pptx               # 발표 슬라이드
├─ 프로젝트 대본.pdf                               # 발표 대본 PDF
├─ RUN_PROJECT.md                                  # 재현/실행 안내
├─ REPRODUCIBILITY.md                              # 재현 가능 범위와 한계
├─ requirements.txt                                # 공개용 기본 실행 패키지
├─ docs/assets/                                    # 문서용 대표 이미지
├─ project_ready_data/                             # 로컬 산출물 폴더(공개 저장소에는 없을 수 있음)
│  ├─ external_features/                           # Google Trends, 캘린더 변수
│  ├─ model_outputs/                               # 최종 모델 성능 및 예측 결과
│  ├─ notebook_figures/                            # 노트북 기반 시각화 결과
│  └─ ppt_figures/                                 # PPT/논문형 시각화 결과
└─ 프로젝트-공유패키지/                              # 팀 공유용 분류 패키지
```

## 빠르게 재현하는 방법

### 환경

- Python 3.11+ 권장
- Windows 환경 기준으로 정리
- GPU가 있으면 더 빠르지만, 최종 결과 확인 정도는 CPU로도 가능

### 패키지 설치

```bash
pip install -r requirements.txt
```

### 먼저 볼 파일

- 전체 흐름: `youtube_trend_project_pipeline.executed.ipynb`
- 최종 모델: `train_active_category_rank_bigru.py`
- 최종 예측 재생성: `run_core10_top_predictions.py`

### 최종 예측 결과 다시 만들기

```bash
python run_core10_top_predictions.py
```

### 논문형 시각화 다시 만들기

```bash
python make_paper_visualization_suite.py
```

### 재현 시 알아둘 점

- 저장소 공개 시점에는 대용량 생성 결과가 Git에서 제외될 수 있습니다.
- `project_ready_data/`는 로컬 산출물 폴더라서 공개 저장소에 항상 포함되지 않을 수 있습니다.
- 전체 산출물이 필요한 경우 `프로젝트-공유패키지` 폴더를 함께 확인하는 것이 가장 안전합니다.
- 실행 순서와 파일 의존성은 [RUN_PROJECT.md](./RUN_PROJECT.md)에 더 자세히 정리했습니다.
- 재현 가능 범위와 공개 저장소의 한계는 [REPRODUCIBILITY.md](./REPRODUCIBILITY.md)에 별도로 정리했습니다.

## 읽는 순서

1. **README.md**: 프로젝트 전체 개요
2. **PROJECT_JOURNEY.md**: 처음부터 끝까지의 진행 과정
3. **TROUBLESHOOTING.md**: 실제 문제와 해결 방식
4. **FINAL_MODEL.md**: 최종 모델 구조와 설계
5. **RESULTS.md**: 최종 성능과 Top-5 예측 결과

## 같이 보면 좋은 문서

- [PROJECT_JOURNEY.md](./PROJECT_JOURNEY.md)
- [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)
- [FINAL_MODEL.md](./FINAL_MODEL.md)
- [RESULTS.md](./RESULTS.md)
- [REPRODUCIBILITY.md](./REPRODUCIBILITY.md)

## 문서 사용 원칙

- 아래 문서들은 **최종 공개용 문서**입니다.
  - `README.md`
  - `PROJECT_JOURNEY.md`
  - `TROUBLESHOOTING.md`
  - `FINAL_MODEL.md`
  - `RESULTS.md`
  - `RUN_PROJECT.md`
  - `REPRODUCIBILITY.md`
- 그 외 루트의 다른 Markdown 파일들은 프로젝트 진행 중 만들어진 **작업 메모/보조 기록**일 수 있으므로, 처음 보는 사람은 위 최종 문서부터 읽는 것이 가장 안전합니다.

## 프로젝트 성격

- 팀 프로젝트 기반
- 이 저장소는 최종 정리와 경험 기록에 초점을 맞춘 버전
- 주요 기여 영역: 데이터 정리, 문제 재정의, 최종 모델 정리, 시각화, 발표 자료 통합

## 주의할 점

- 저장소에는 실험 중간 산출물도 많기 때문에, 처음에는 메인 노트북과 최종 모델 결과부터 보는 것이 좋습니다.
- 공개 저장소로 옮기기 전에는 API 키, 대용량 파일, 개인 로그 파일이 포함되지 않았는지 반드시 다시 확인해야 합니다.
- Git 기반 업로드에서는 `.gitignore`에 의해 로컬 키 파일과 로그가 제외되지만, **폴더 전체를 수동 업로드하는 방식은 권장하지 않습니다.**
- GitHub 공개 전에는 문서와 노트북이 UTF-8 기준으로 정상 렌더링되는지 한 번 더 확인하는 것이 좋습니다.

## 한 줄 요약

이 프로젝트는 **유튜브 핵심 10개 분야의 향후 4주 상승 가능성을 예측하기 위해, 데이터 오류를 직접 수정하고 문제를 재정의하며 최종적으로 BiGRU 기반 Top-5 선별 모델까지 완성한 경험 기록형 딥러닝 프로젝트**입니다.

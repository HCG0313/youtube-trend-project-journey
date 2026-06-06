# 데이터 파이프라인 핵심 코드 인덱스

이 문서는 공개 저장소에 추가한 핵심 데이터 파이프라인 코드가 각각 어디에 쓰이는지 빠르게 확인하기 위한 안내서다.

## 이번에 추가한 핵심 파일

### 1. `week_utils.py`
- ISO week 기준으로 주차를 계산하고 정렬하는 보조 함수 모음이다.
- `year_week`를 단순 문자열이 아니라 실제 ISO 주차 기준으로 맞추는 데 사용했다.
- 발표와 문서에서 설명한 `주차 축 정렬 문제`를 해결할 때 직접 사용된 파일이다.

### 2. `build_category_trend_dataset.py`
- historical video CSV들을 묶어서 카테고리-주차 단위의 초기 추세 테이블을 만드는 스크립트다.
- 과거 카테고리 흐름을 하나의 주간 데이터셋으로 정리하는 데 사용했다.

### 3. `integrate_project_ready_data.py`
- 프로젝트에서 가장 중요한 전처리/통합 스크립트다.
- 여러 원천 영상 데이터, 시계열 지원 데이터, 외부 데이터 소스를 합쳐서 `project_ready_data` 안의 최종 학습용 테이블을 만든다.
- 파생변수 생성, 카테고리-주차 집계, 주차 정렬, 태그/쇼츠 관련 보조 변수 생성도 이 파일에서 처리한다.

### 4. `collect_google_trends_apify.py`
- Google Trends를 Apify 기반으로 다시 수집하는 스크립트다.
- active category 기준으로 주차별 검색량을 chunk 단위로 수집하고, 겹치는 구간을 보정한 뒤 정규화한다.
- 발표와 문서에서 설명한 `Google Trends 복구` 과정과 직접 연결된다.

### 5. `merge_apify_google_trends_batches.py`
- 여러 번에 나눠 수집된 Apify raw 결과를 하나로 합치는 보조 스크립트다.
- chunk별 겹침 구간을 이용해 스케일을 맞추고, 최종 normalized 테이블을 다시 만든다.

## 이 다섯 개가 중요한 이유

현재 공개 저장소에는 최종 모델 코드인 `train_active_category_rank_bigru.py`와 예측 코드인 `run_core10_top_predictions.py`가 이미 올라가 있다.  
하지만 그 전 단계에서

- 주차를 어떻게 맞췄는지
- 카테고리-주차 데이터셋을 어떻게 만들었는지
- Google Trends를 어떻게 복구하고 합쳤는지
- 최종 학습용 테이블과 파생변수를 어떻게 만들었는지

를 보여주는 핵심 코드가 빠져 있었다.

이번에 추가한 다섯 개 파일은 그 빈 부분을 메운다.

## 권장 읽는 순서

1. `week_utils.py`
2. `build_category_trend_dataset.py`
3. `collect_google_trends_apify.py`
4. `merge_apify_google_trends_batches.py`
5. `integrate_project_ready_data.py`
6. `train_active_category_rank_bigru.py`
7. `run_core10_top_predictions.py`

이 순서로 보면 `주차 정렬 → 카테고리 주간 데이터 생성 → 외부 검색량 수집/병합 → 최종 학습 테이블 생성 → 모델 학습/예측` 흐름을 자연스럽게 따라갈 수 있다.

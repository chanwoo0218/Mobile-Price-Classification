# Mobile Price Classification

스마트폰 하드웨어 사양으로 가격대 `0–3`을 예측한 다중분류 프로젝트입니다. 첨부된 전처리·모델 비교·최종 최적화 노트북과 발표자료를 기준으로 저장소를 수정했습니다.

## Data

Kaggle Mobile Price Classification 데이터셋을 사용했습니다.

- train: 2,000 rows, 20 input variables + `price_range`
- test: 1,000 rows, 20 input variables + `id`
- target: 4개 가격 등급이 균형 있게 분포
- 원본 데이터에 결측값 없음

## Preprocessing & EDA

- 배터리, RAM, 화면 해상도, 카메라, 통신 기능 등 변수 점검
- `sc_total = sc_h × sc_w`, `px_total = px_height × px_width` 파생변수 생성
- 이진 범주형 변수와 가격대: Chi-square test
- 연속형 변수와 가격대: Kruskal–Wallis test
- 중복 정보와 설명력이 낮은 변수를 검토한 뒤 학습 데이터 생성

## Modeling

1. Random Forest baseline과 KFold/StratifiedKFold 비교
2. Logistic Regression, SVM, KNN, Decision Tree, Random Forest 등 후보 비교
3. 1차 선택 기준은 macro F1
4. 실제 활용 관점에서 가장 비싼 class 3의 recall을 핵심 보조 지표로 설정
5. Logistic Regression의 `C`와 `class_weight`를 GridSearch하여 class 3 false negative를 줄임

## Results

발표자료에 기록된 최종 결과:

| Metric | Before | Final |
|---|---:|---:|
| Macro F1 | 0.9405 | **0.9445** |
| Class 3 recall | - | **0.9670** |
| Class 3 false negatives | 15 | **12** |

오류는 대부분 실제 등급과 인접한 등급으로의 혼동이었고, 특히 `3 → 2` 오분류를 줄이는 데 초점을 맞췄습니다.

## Repository Structure

```text
src/train.py
notebooks/01_preprocessing.ipynb
notebooks/02_model_comparison_and_tuning.ipynb
data/raw/train.csv
data/raw/test.csv
data/processed/train_preprocessed.csv
docs/presentation.pdf
```

## Run

```bash
pip install -r requirements.txt
python src/train.py --data data/processed/train_preprocessed.csv
```

원본 노트북의 실험 흐름과 seed를 유지하면서, 저장소용 실행 스크립트는 재사용하기 쉽게 정리했습니다.
# 스마트폰 가격 등급 분류

> 스마트폰 하드웨어 사양으로 가격 등급 `0~3`을 분류하고, 고가 제품을 낮은 가격대로 오분류하는 손실을 줄이기 위해 **3등급 재현율**을 함께 최적화한 다중분류 프로젝트입니다.

## 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 수행 기간 | 2026.01.02 ~ 2026.02.25 |
| 문제 유형 | 다중분류·비즈니스 지표 최적화 |
| 데이터 | 2,000개 스마트폰, 4개 균형 등급 |
| 최종 모델 | Logistic Regression |
| 주요 지표 | Macro F1, 3등급 Recall |

## 분석 과정

- 자료형·결측치·등급 균형과 변수 분포 점검
- 화면 면적 `sc_total`, 총 픽셀량 `px_total` 파생변수 생성
- 범주형 변수는 카이제곱 검정, 수치형 변수는 Kruskal-Wallis 검정으로 관련성 검토
- Logistic Regression, SVM, KNN, Decision Tree, Random Forest 비교
- 교차검증 Macro F1과 고가 등급 Recall을 함께 고려해 최종 모델 선택
- 혼동행렬로 `3 → 2` 오분류를 집중 분석하고 class weight 조정

## 주요 결과

| 지표 | 기본 Logistic Regression | 최종 조정 모델 |
|---|---:|---:|
| Macro F1 | 0.9405 | **0.9445** |
| 3등급 Recall | - | **0.9670** |
| 3등급 False Negative | 15 | **12** |

전체 정확도만 높이는 대신, 프리미엄 제품을 저평가하는 오류의 비용을 모델 선택에 반영했습니다.

## 저장소 구성

```text
.
├── README.md
├── requirements.txt
├── data
│   ├── README.md
│   ├── raw/train.csv
│   ├── raw/test.csv
│   ├── processed/train_preprocessed.csv
│   └── sample/train_preprocessed_sample.csv
├── docs
│   └── 발표자료_요약.md
└── notebooks
    ├── 01_데이터_전처리.ipynb
    ├── 02_RandomForest_모델링.ipynb
    ├── 03_RandomForest_평가.ipynb
    ├── 04_분류모델_비교.ipynb
    └── 05_최종_모델링_및_최적화.ipynb
```

## 실행 방법

```bash
pip install -r requirements.txt
python scripts/reassemble_data.py
jupyter notebook
```

노트북은 번호 순서대로 실행할 수 있습니다. `03_RandomForest_평가.ipynb`는 앞선 모델링 노트북에서 생성한 모델 파일이 필요할 수 있습니다.

## 한계와 개선 방향

- 데이터가 작고 인위적으로 균형화되어 실제 상품 카탈로그 분포와 다를 수 있습니다.
- 가격 등급은 순서형이지만 기본 모델은 일반 다중분류로 처리합니다.
- 실제 사업 손실을 기반으로 한 비용행렬, 순서형 분류, 확률 보정이 필요합니다.

## 포트폴리오

프로젝트 배경과 학습 회고는 [노션 포트폴리오](https://app.notion.com/p/b3f82d8994c2833094278155ac67d45d)에서 확인할 수 있습니다.

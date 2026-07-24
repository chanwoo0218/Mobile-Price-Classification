# 데이터 안내

원본 데이터는 Kaggle의 **Mobile Price Classification** 데이터셋입니다.

- `data/raw/train.csv`: 학습 원본(2,000행, target=`price_range`)
- `data/raw/test.csv`: 예측용 테스트 원본
- `data/processed/train_preprocessed.csv`: 통계 검정과 파생변수 생성 후 모델링에 사용한 데이터
- `data/sample/train_preprocessed_sample.csv`: 스키마 확인용 표본

현재 GitHub 연결에서는 로컬 CSV를 직접 전달하는 기능이 없어 원본·전체 전처리 CSV는 별도 업로드가 필요합니다. `notebooks/01_데이터_전처리.ipynb`를 실행하면 `train_preprocessed.csv`를 다시 생성할 수 있습니다.

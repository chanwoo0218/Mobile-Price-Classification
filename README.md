# Mobile Price Classification

모바일 기기의 하드웨어 스펙을 이용해 가격 범위 `0–3`을 예측한 다중분류 프로젝트입니다. 단순 정확도보다 **고가 제품(class 3)을 놓치지 않는 recall**을 핵심 비즈니스 지표로 설정했습니다.

## Workflow

1. 결측값·변수 타입·클래스 분포 확인
2. `sc_total`, `px_total` 파생변수 생성
3. 범주형 변수: Chi-square test
4. 수치형 변수: Kruskal–Wallis test
5. Logistic Regression과 SVM을 macro F1로 비교
6. class 3 recall을 우선하고 macro F1을 보조 기준으로 튜닝
7. Confusion Matrix로 `3 → 2` 오분류 분석
8. class weight 조정으로 고가 제품의 false negative 감소

## Documented Result

발표 자료에 기록된 최종 결과:

- Macro F1: `0.9405 → 0.9445`
- Class 3 recall: `0.9670`
- Class 3 false negatives: `15 → 12`

## Run

```bash
pip install -r requirements.txt
python src/train.py --data data/train.csv
```

## Repository Note

공개용 코드는 노션 포트폴리오와 최종 발표 자료에 기록된 분석 절차를 재현하도록 정리했습니다. 원본 노트북을 확보하면 동일 데이터 분할과 seed로 결과를 재검증해야 합니다.

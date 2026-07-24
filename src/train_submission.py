from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model(class3_weight: float = 1.3) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=10,
            penalty="l2",
            solver="lbfgs",
            max_iter=3000,
            multi_class="multinomial",
            class_weight={0: 1.0, 1: 1.0, 2: 1.0, 3: class3_weight},
            random_state=121,
        )),
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/train_preprocessed.csv"))
    parser.add_argument("--model-out", type=Path, default=Path("models/logistic_regression.joblib"))
    parser.add_argument("--metrics-out", type=Path, default=Path("outputs/metrics.json"))
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    X = df.drop(columns="price_range")
    y = df["price_range"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=121)
    oof = cross_val_predict(build_model(), X, y, cv=cv)
    metrics = {
        "oof_macro_f1": f1_score(y, oof, average="macro"),
        "oof_class3_recall": recall_score(y, oof, labels=[3], average=None)[0],
        "oof_class3_precision": precision_score(y, oof, labels=[3], average=None, zero_division=0)[0],
        "oof_class3_false_negatives": int(((y == 3) & (oof != 3)).sum()),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=121, stratify=y
    )
    model = build_model()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics["holdout_classification_report"] = classification_report(y_test, pred, output_dict=True)
    metrics["holdout_confusion_matrix"] = confusion_matrix(y_test, pred).tolist()

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_out)
    args.metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

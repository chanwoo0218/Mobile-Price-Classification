"""Business-metric-aware Mobile Price Classification pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from scipy.stats import chi2_contingency, kruskal
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

TARGET = "price_range"
BINARY_CANDIDATES = ["blue", "dual_sim", "four_g", "three_g", "touch_screen", "wifi"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"sc_h", "sc_w"}.issubset(out.columns):
        out["sc_total"] = out["sc_h"] * out["sc_w"]
    if {"px_height", "px_width"}.issubset(out.columns):
        out["px_total"] = out["px_height"] * out["px_width"]
    return out


def select_features(df: pd.DataFrame, alpha: float = 0.05) -> list[str]:
    selected: list[str] = []
    for col in df.columns:
        if col == TARGET or col in {"sc_h", "sc_w", "px_height", "px_width"}:
            continue
        try:
            if col in BINARY_CANDIDATES or df[col].nunique() <= 4:
                table = pd.crosstab(df[col], df[TARGET])
                p = chi2_contingency(table)[1]
            else:
                groups = [g[col].dropna().to_numpy() for _, g in df.groupby(TARGET)]
                p = kruskal(*groups).pvalue
            if p < alpha:
                selected.append(col)
        except Exception:
            selected.append(col)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train.csv")
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()

    df = add_features(pd.read_csv(args.data)).dropna()
    features = select_features(df)
    X, y = df[features], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scaler = ColumnTransformer([("num", StandardScaler(), features)], remainder="drop")

    candidates = {
        "logistic": Pipeline([("scale", scaler), ("model", LogisticRegression(max_iter=5000))]),
        "svm": Pipeline([("scale", scaler), ("model", SVC())]),
    }
    cv_scores = {}
    for name, model in candidates.items():
        pred = cross_val_predict(model, X_train, y_train, cv=cv, n_jobs=-1)
        cv_scores[name] = {
            "macro_f1": float(f1_score(y_train, pred, average="macro")),
            "recall_class_3": float(recall_score(y_train, pred, labels=[3], average="macro")),
        }

    rows = []
    for params in ParameterGrid({
        "C": [0.01, 0.1, 1, 10, 100],
        "class_weight": [None, {3: 1.1}, {3: 1.2}, {2: 1.05, 3: 1.15}],
    }):
        model = Pipeline([
            ("scale", scaler),
            ("model", LogisticRegression(max_iter=5000, C=params["C"], class_weight=params["class_weight"])),
        ])
        pred = cross_val_predict(model, X_train, y_train, cv=cv, n_jobs=-1)
        rows.append({
            **params,
            "recall_class_3": recall_score(y_train, pred, labels=[3], average="macro"),
            "macro_f1": f1_score(y_train, pred, average="macro"),
        })
    top_recall = max(r["recall_class_3"] for r in rows)
    shortlist = [r for r in rows if r["recall_class_3"] >= top_recall - 0.01]
    best = max(shortlist, key=lambda r: r["macro_f1"])

    final_model = Pipeline([
        ("scale", scaler),
        ("model", LogisticRegression(max_iter=5000, C=best["C"], class_weight=best["class_weight"])),
    ])
    final_model.fit(X_train, y_train)
    pred = final_model.predict(X_test)
    report = classification_report(y_test, pred, output_dict=True)
    metrics = {
        "selected_features": features,
        "candidate_cv": cv_scores,
        "best_params": best,
        "test_macro_f1": float(f1_score(y_test, pred, average="macro")),
        "test_recall_class_3": float(recall_score(y_test, pred, labels=[3], average="macro")),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": report,
    }

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump(final_model, output / "model.joblib")
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))


if __name__ == "__main__":
    main()

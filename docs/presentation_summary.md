# Presentation Summary

## Objective

Predict one of four mobile-phone price ranges from hardware specifications while minimizing missed high-price devices.

## Analysis flow

1. Inspect data quality, types, and balanced target distribution.
2. Engineer screen-area and pixel-area features.
3. Use Chi-square tests for binary categorical features and Kruskal–Wallis tests for continuous features.
4. Compare Logistic Regression, SVM, KNN, Decision Tree, and Random Forest with macro F1.
5. Tune Logistic Regression with `C` and class weights, prioritizing class-3 recall.

## Final reported outcome

- Macro F1: 0.9405 → 0.9445
- Class-3 recall: 0.9670
- Class-3 false negatives: 15 → 12

Most mistakes occurred between adjacent price classes. The final decision therefore emphasized reducing class `3 → 2` errors rather than optimizing accuracy alone.

## Binary source note

The submitted presentation is retained in the downloadable reviewed package. This repository contains a Markdown summary because the connector used for this update did not provide a safe direct binary-file handoff from the uploaded archive.

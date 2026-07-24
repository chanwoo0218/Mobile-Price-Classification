# Mobile Price Classification

> **스마트폰 하드웨어 사양으로 가격 등급 0-3을 분류하고, 고가 제품을 낮은 가격대로 오분류하는 손실을 줄이기 위해 class 3 recall을 중심으로 최적화한 다중분류 프로젝트입니다.**

[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://www.python.org/) ![Model](https://img.shields.io/badge/Final%20Model-Logistic%20Regression-orange) ![Metric](https://img.shields.io/badge/Business%20Metric-Class%203%20Recall-green)

## At a Glance

| Item | Description |
|---|---|
| Project type | Multiclass classification / Business-metric optimization |
| Period | 2026.01.02 - 2026.02.25 |
| Activity | DF winter short-term program |
| Dataset | Kaggle Mobile Price Classification |
| Observations | 2,000 devices |
| Classes | Four balanced price ranges, 500 observations each |
| Final model | Logistic Regression |
| Primary metrics | Macro F1 and class 3 recall |
| Core stack | Python, Pandas, scikit-learn, SciPy |

## Problem

The dataset contains battery capacity, RAM, internal memory, camera specifications, screen size, pixel resolution, and connectivity features. The task is to classify each device into one of four price ranges.

A model with high overall accuracy can still be costly if it systematically classifies premium products as cheaper products. The project therefore frames the problem as:

> **How can we maintain balanced multiclass performance while reducing false negatives for the highest price class?**

## Dataset

The original training data contains **2,000 rows and 21 columns**, including the target. All four target classes are perfectly balanced.

| Class | Meaning | Count |
|---:|---|---:|
| 0 | Lowest price range | 500 |
| 1 | Lower-middle price range | 500 |
| 2 | Upper-middle price range | 500 |
| 3 | Highest price range | 500 |

Feature groups include:

- Battery and performance: `battery_power`, `clock_speed`, `ram`, `n_cores`
- Storage: `int_memory`
- Camera: `fc`, `pc`
- Display: `px_height`, `px_width`, `sc_h`, `sc_w`
- Physical properties: `mobile_wt`, `m_dep`
- Connectivity: Bluetooth, dual SIM, 3G, 4G, touch screen, Wi-Fi

## Exploratory Analysis and Feature Engineering

- Verified data types, missing values, class balance, and feature distributions.
- Created `sc_total` to represent screen-area information.
- Created `px_total = px_height × px_width` to represent total pixel capacity.
- Examined categorical features using the Chi-square test.
- Examined numerical features using the Kruskal-Wallis test.
- Removed weak or redundant variables based on statistical and modeling evidence.

The final submitted preprocessed table retained:

```text
battery_power, int_memory, mobile_wt, ram, px_total, price_range
```

## Key Decisions

### Why Macro F1 instead of accuracy alone?

Macro F1 computes the F1 score independently for each class and then averages them, preventing strong performance on one class from hiding weak performance on another.

### Why prioritize class 3 recall?

Class 3 recall measures how many true premium devices the model successfully recognizes. A class 3 false negative represents a premium device assigned to a lower price category, which was defined as the more costly business error.

### Why Logistic Regression?

Multiple classifiers were compared, including Logistic Regression, SVM, KNN, Decision Tree, and Random Forest. Logistic Regression provided strong balanced performance, stable cross-validation behavior, and transparent class-weight adjustment.

## Modeling Pipeline

```text
EDA and class-balance check
        ↓
Derived display features
        ↓
Chi-square / Kruskal-Wallis tests
        ↓
Feature selection and scaling
        ↓
Logistic Regression, SVM, KNN, tree models
        ↓
Cross-validated Macro F1 comparison
        ↓
Class-weight and hyperparameter tuning
        ↓
Confusion-matrix error analysis
```

## Results

| Metric | Baseline Logistic Regression | Tuned model |
|---|---:|---:|
| Macro F1 | 0.9405 | **0.9445** |
| Class 3 recall | - | **0.9670** |
| Class 3 false negatives | 15 | **12** |

The final tuning produced a modest Macro F1 improvement while reducing the number of premium devices missed by the model.

## Error Analysis

The confusion matrix was used to inspect not only how many predictions were wrong, but **which direction the errors occurred**. Particular attention was paid to class `3 → 2` errors because they directly correspond to underestimating premium products.

This project demonstrates that final-model selection should reflect the cost structure of errors rather than rely on a single aggregate score.

## Project Work

- Conducted EDA and verified the balanced four-class structure.
- Designed `sc_total` and `px_total` derived variables.
- Used Chi-square and Kruskal-Wallis tests to support feature-selection decisions.
- Compared multiple linear, kernel, distance-based, and tree-based classifiers.
- Selected the final model using Macro F1 and business-oriented class 3 recall.
- Tuned class weights and analyzed false negatives with a confusion matrix.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── data
│   ├── README.md
│   └── sample
│       └── train_preprocessed_sample.csv
├── docs
│   ├── README.md
│   └── presentation_summary.md
└── src
    ├── train.py
    └── train_submission.py
```

## How to Run

```bash
git clone https://github.com/chanwoo0218/Mobile-Price-Classification.git
cd Mobile-Price-Classification
pip install -r requirements.txt
```

Run the submission-derived pipeline:

```bash
python src/train_submission.py --data data/train.csv
```

The script reproduces feature engineering, preprocessing, cross-validation, and final evaluation. See `data/README.md` for the required schema.

## Limitations

- The dataset is small and artificially balanced, unlike many real product catalogs.
- Price classes are ordinal, but the primary model treats them as nominal classes.
- The business cost assigned to class 3 errors is a project assumption rather than a measured financial value.
- Market price can also depend on brand, release timing, design, and demand variables absent from the dataset.

## Future Work

- Ordinal classification or cost-sensitive learning
- Explicit cost matrix derived from business impact
- Probability calibration and uncertainty-aware decisions
- External validation using newer device data
- Feature-attribution analysis for individual products

## Portfolio

The Korean-language project explanation and learning reflections are available on the [Notion portfolio page](https://app.notion.com/p/b3f82d8994c2833094278155ac67d45d).
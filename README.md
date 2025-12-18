# Supervised Learning for Email Classification
## A Comparative Study of CART, k-NN, Random Forest, and Oblique Decision Trees

**Course:** Introduction to Artificial Intelligence – FEUP  
**Academic Year:** 2025/2026  
**Assignment:** Supervised Learning (Assignment 2)

**Authors:**
- Paolo Pascarelli
- João Filipe
- Diogo Teixeira

---

## Table of Contents

1. [Problem Description](#1-problem-description)
2. [Dataset](#2-dataset)
3. [Installation & Requirements](#3-installation--requirements)
4. [How to Run](#4-how-to-run)
5. [Notebook Structure](#5-notebook-structure)
6. [Preprocessing Pipeline](#6-preprocessing-pipeline)
7. [Algorithms Implemented](#7-algorithms-implemented)
8. [Evaluation Methodology](#8-evaluation-methodology)
9. [Results Summary](#9-results-summary)
10. [File Structure](#10-file-structure)

---

## 1. Problem Description

**Task:** Binary classification for email spam detection.

**Objective:** Automatically classify emails as **Spam (1)** or **Not Spam (0)** based on word frequencies, character frequencies, and capital letter statistics.

**Target Variable:** Binary class label
- `1` = Spam (unsolicited email)
- `0` = Non-spam (legitimate email)

**Goal:** Compare four supervised learning algorithms evaluating trade-offs between:
- Predictive accuracy
- Model interpretability  
- Training time
- Model complexity

---

## 2. Dataset

**Name:** Spambase  
**Source:** UCI Machine Learning Repository / OpenML (ID: 44)  
**Reference:** Hopkins, M., Reeber, E., Forman, G., & Suermondt, J. (1999)

| Property | Value |
|----------|-------|
| Instances | 4,601 |
| Features | 57 (continuous) |
| Classes | 2 (binary) |
| Missing Values | 0 |
| Class Balance | 60.6% non-spam / 39.4% spam |

**Feature Groups:**
- **Word Frequencies (48):** Percentage of words matching specific terms (e.g., 'make', 'money', 'free', 'credit')
- **Character Frequencies (6):** Percentage of special characters (`;`, `(`, `[`, `!`, `$`, `#`)
- **Capital Letter Statistics (3):** Average length, longest sequence, total count of capital letters

---

## 3. Installation & Requirements

### Prerequisites

- Python 3.9+
- Jupyter Notebook or JupyterLab

### Required Libraries

```bash
pip install numpy pandas scikit-learn matplotlib seaborn openml scipy
```

### Dependencies List

| Library | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.21 | Numerical operations |
| pandas | ≥1.3 | Data manipulation |
| scikit-learn | ≥1.0 | ML algorithms, preprocessing, evaluation |
| matplotlib | ≥3.4 | Visualization |
| seaborn | ≥0.11 | Statistical plots |
| openml | ≥0.12 | Dataset loading |
| scipy | ≥1.7 | Statistical functions |

---

## 4. How to Run

### Option 1: Jupyter Notebook (Recommended)

```bash
# 1. Clone or download the repository
# 2. Navigate to the project directory
cd path/to/project

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook tree_ai_extended-3.ipynb
```

### Option 2: JupyterLab

```bash
jupyter lab tree_ai_extended-3.ipynb
```

### Execution Instructions

1. **Run All Cells:** Use `Kernel → Restart & Run All` to execute the entire notebook sequentially
2. **Expected Runtime:** ~5-10 minutes (depending on hardware)
3. **Internet Required:** First run requires internet to download the Spambase dataset from OpenML

### Troubleshooting

| Issue | Solution |
|-------|----------|
| OpenML connection error | Check internet connection; dataset is cached after first download |
| Memory error | Close other applications; reduce `n_jobs` parameter to 1 |
| Convergence warnings | These are suppressed by default; safe to ignore |

---

## 5. Notebook Structure

The notebook is organized into four main parts:

### Part 1: Algorithm Implementation
- Custom ISTA (Iterative Shrinkage-Thresholding Algorithm) optimizer
- Oblique Decision Tree class with hybrid splitting strategy
- Soft thresholding for L1 regularization

### Part 2: Data Understanding & Preprocessing
- Dataset loading from OpenML
- Exploratory Data Analysis (class balance, feature distributions)
- Correlation analysis
- Preprocessing pipeline construction

### Part 3: Model Training & Evaluation
- Nested cross-validation (5 outer folds × 3 inner folds)
- Hyperparameter tuning via GridSearchCV
- Performance metrics computation

### Part 4: Visualization & Analysis
- ROC curves comparison
- Confusion matrices
- Learning curves
- Feature importance analysis
- Training time comparison

---

## 6. Preprocessing Pipeline

We use scikit-learn's `Pipeline` and `ColumnTransformer` to ensure **no data leakage**:

```
┌─────────────────────────────────────────────────────────┐
│                  Preprocessing Pipeline                  │
├─────────────────────────────────────────────────────────┤
│ Step 1: Imputation     │ SimpleImputer (median)         │
│ Step 2: Encoding       │ OneHotEncoder (handle_unknown) │
│ Step 3: Scaling        │ StandardScaler                 │
└─────────────────────────────────────────────────────────┘
```

**Key Design Choices:**
- **Median imputation:** Robust to outliers (Spambase has no missing values, but ensures generalizability)
- **Global standardization:** Essential for k-NN distance calculations and Oblique Tree gradient optimization
- **Pipeline wrapping:** Ensures preprocessing is fitted only on training data during cross-validation

---

## 7. Algorithms Implemented

### 7.1 CART (Classification and Regression Trees)
- **Type:** Axis-aligned decision tree
- **Split criterion:** Gini impurity
- **Hyperparameters tuned:** `max_depth`, `min_samples_split`, `ccp_alpha`

### 7.2 k-Nearest Neighbors (k-NN)
- **Type:** Instance-based lazy learning
- **Distance metric:** Minkowski (p=1 Manhattan, p=2 Euclidean)
- **Hyperparameters tuned:** `n_neighbors`, `weights`, `p`

### 7.3 Random Forest
- **Type:** Ensemble of bagged decision trees
- **Randomization:** Bootstrap sampling + feature subsampling
- **Hyperparameters tuned:** `n_estimators`, `max_depth`, `max_features`

### 7.4 Oblique Decision Tree (Custom Implementation)
- **Type:** Decision tree with linear (oblique) splits
- **Split form:** wᵀx + b < 0
- **Optimization:** Hybrid two-phase strategy

**Hybrid Optimization Strategy:**
1. **Phase 1 (Selection):** `sklearn.LogisticRegression` with `lbfgs` solver, `penalty=None` to rapidly evaluate all 2^(K-1)-1 class bipartitions
2. **Phase 2 (Sparsification):** Custom ISTA solver with L1 regularization drives uninformative weights to zero

**Hyperparameters tuned:** `max_depth`, `l1_lambda`, `min_impurity_decrease`

---

## 8. Evaluation Methodology

### Cross-Validation Strategy

**Nested Stratified Cross-Validation:**
- **Outer loop:** 5 folds (performance estimation)
- **Inner loop:** 3 folds (hyperparameter tuning)

This prevents optimistic bias from hyperparameter selection on test data.

### Metrics Computed

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct predictions |
| **Balanced Accuracy** | Average per-class recall |
| **Precision** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |
| **F1-Score (macro)** | Harmonic mean of precision and recall |
| **ROC-AUC** | Area under ROC curve |
| **Training Time** | Wall-clock time for model fitting |
| **Tree Nodes** | Model complexity (for tree-based methods) |

---

## 9. Results Summary

### Performance Comparison (5-Fold Nested CV)

| Model | Accuracy | F1 (macro) | Precision | Recall | Time (s) | Nodes |
|-------|----------|------------|-----------|--------|----------|-------|
| CART | 0.917 ± 0.011 | 0.912 | 0.915 | 0.910 | 2.25 | 97.8 |
| **Oblique Tree** | **0.931 ± 0.006** | **0.928** | **0.928** | **0.928** | 33.25 | **13.4** |
| k-NN | 0.920 ± 0.009 | 0.915 | 0.920 | 0.912 | 0.55 | — |
| Random Forest 🏆 | 0.953 ± 0.008 | 0.950 | 0.953 | 0.948 | 14.32 | 582 |

### Key Findings

1. **Random Forest** achieves highest accuracy (95.3%) but sacrifices interpretability
2. **Oblique Tree** outperforms CART (+1.4% accuracy) with **7× fewer nodes** (13 vs 98)
3. **k-NN** provides fast training but defers computation to prediction time
4. **Trade-off:** Oblique Tree training time (33s) is higher due to iterative ISTA optimization

### ROC-AUC Scores

| Model | AUC |
|-------|-----|
| Random Forest | 0.98 |
| k-NN | 0.96 |
| Oblique Tree | 0.96 |
| CART | 0.95 |

---

## 10. File Structure

```
project/
│
├── tree_ai_extended-3.ipynb    # Main Jupyter notebook (source code)
├── README.md                    # This file
├── presentation.pdf             # 10-slide presentation summary
│
└── (generated at runtime)
    ├── figures/                 # Saved plots (ROC, confusion matrices, etc.)
    └── results/                 # Exported metrics tables
```

---

## References

- Rusland, N.F., et al. (2017). A Comparative Study for Spam Classifications. https://www.researchgate.net/publication/340096186
- Int. Journal of Information Security (2023). Ensemble Methods for Spam Detection. https://link.springer.com/article/10.1007/s10207-023-00756-1
- Knowledge-Based Systems (2008). Content-Based Dynamic Spam Classification. https://www.sciencedirect.com/science/article/abs/pii/S0950705108000026
- T. Hastie, R. Tibshirani, and J. Friedman, The Elements of Statistical Learning: Data Mining, Inference, and Prediction. Springer Science & Business Media, 2nd ed., 2009.
- L. Hyafil and R. L. Rivest, “Constructing optimal binary decision trees is npcomplete,” Information Processing Letters, vol. 5, no. 1, pp. 15–17, 1976.
- L. Breiman, J. Friedman, C. J. Stone, and R. A. Olshen, Classification and Regression Trees. CRC press, 1984.
- J. R. Quinlan, C4.5: Programs for Machine Learning. Morgan Kaufmann, 1993.
- J. R. Quinlan, “Induction of decision trees,” Machine Learning, vol. 1, no. 1, pp. 81106, 1986.
- C. E. Shannon, “A mathematical theory of communication,” The Bell System Technical Journal, vol. 27, no. 3, pp. 379–423, 1948.
- S. K. Murthy, S. Kasif, and S. Salzberg, “A system for induction of oblique decision trees,” Journal of artificial intelligence research, vol. 2, pp. 1–32, 1994.

---

## License

This project was developed for educational purposes as part of the Artificial Intelligence course at FEUP.

---

**Last Updated:** December 2025

# Credit Card Fraud Detection

A production-oriented machine learning pipeline for detecting fraudulent transactions on a severely imbalanced dataset. Built to reflect the modeling decisions a data scientist encounters in fintech — where false negatives cost real money and false positives erode customer trust.

---

## Business Context

Payment fraud cost the global financial industry over $30 billion in 2023. At a company like Ramp — processing billions in corporate card spend — a fraud model needs to answer two questions simultaneously:

1. **How often do we catch fraud before it lands?** (Recall)
2. **How often do we wrongly block a legitimate transaction?** (Precision)

These goals are in tension. Tightening the fraud threshold catches more fraud but declines more legitimate spend, frustrating cardholders and finance teams. The right operating point is a **business decision**, not a model decision — which is why this project focuses on building a calibrated probability scorer and exposing the full precision-recall curve rather than optimizing a single threshold.

The dataset reflects a related challenge: **class imbalance**. Only 0.172% of transactions are fraudulent. A naive model that flags nothing achieves 99.83% accuracy — and is completely useless. This is why accuracy is not reported anywhere in this project.

---

## Dataset

**Source:** [ULB Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — Kaggle
**Size:** 284,807 transactions from European cardholders, September 2013
**Fraud rate:** 0.172% (492 fraudulent transactions)

| Feature | Description |
|---|---|
| `V1`–`V28` | PCA-transformed features (original features are confidential) |
| `Time` | Seconds elapsed since the first transaction in the dataset |
| `Amount` | Transaction amount in EUR |
| `Class` | Target — `1` = fraud, `0` = legitimate |

The V-features are the result of a PCA transformation applied by the data provider to protect cardholder privacy. High-importance components would be traced back to original features (e.g., merchant category, transaction velocity) in a production system where those loadings are available.

---

## Methodology

### 1. Exploratory Data Analysis
- Class distribution, transaction amount and timing patterns
- Visual comparison of fraud vs. legitimate distributions across top features

### 2. Preprocessing
- `StandardScaler` applied to `Amount` and `Time` (V-features are already PCA-normalized)
- Stratified train/test split (80/20) to preserve the real-world fraud rate in the held-out set

### 3. Handling Class Imbalance — Sample Weights
Rather than resampling, we use `compute_sample_weight('balanced')` to assign each training sample a weight inversely proportional to its class frequency. Fraud samples receive a weight ~578x higher than legitimate ones, so the model is penalized heavily for missing fraud. The training set stays at its original 228k rows — no synthetic data generated.

> **Key design choice:** The test set is never reweighted — it reflects the true real-world distribution, so evaluation metrics are meaningful.

### 4. Model — Gradient Boosting Classifier (Fixed Configuration)
Gradient Boosting builds an ensemble of shallow trees sequentially, where each tree corrects the residual errors of the previous one. It is well-suited for tabular fraud data because:
- Non-parametric: captures non-linear feature interactions natively
- Robust to scale differences and outliers
- Outputs calibrated probability scores — essential for threshold tuning

| Parameter | Value | Rationale |
|---|---|---|
| `n_estimators` | `50` | Sufficient signal for a baseline; expand for production |
| `max_depth` | `3` | Shallow trees — standard for boosting, guards against overfit |
| `learning_rate` | `0.1` | Standard shrinkage |
| `sample_weight` | `~578x fraud` | Passed to `fit()` — GBC's equivalent of `class_weight='balanced'` |

### 5. Evaluation
All evaluation is performed on the held-out test set at the original fraud rate (~0.172%).

| Metric | Why it's used |
|---|---|
| **AUPRC** | Threshold-independent; measures quality on the minority class; random baseline ≈ 0.0017 |
| **Precision-Recall Curve** | Full operating curve — lets the business pick its own threshold |
| **F1 Score** | Harmonic mean of precision/recall at the default threshold |
| **Confusion Matrix** | TP/FP/TN/FN counts translated into business impact |
| **Threshold Analysis** | Sweeps P(fraud) thresholds to find the cost-optimal operating point |

---

## Key Results

| Metric | Value |
|---|---|
| Metric | Default threshold (0.5) | Optimized threshold (0.98) |
|---|---|---|
| AUPRC | **0.6961** (random baseline: 0.0017) | — |
| F1 Score | 0.1800 | **0.7813** |
| Precision | — | 0.7979 |
| Recall | 0.9100 (catches 90.8% of fraud) | 0.7653 |

The low default-threshold F1 reflects a deliberate high-recall operating point — at 0.5 the model flags broadly to catch as much fraud as possible (91% recall), and the threshold analysis shows how to dial in precision-recall balance for a specific production cost structure.

---

## Visualizations

The notebook generates and saves the following charts:

| File | Contents |
|---|---|
| `eda_overview.png` | Class distribution, amount histogram, transaction timing by class |
| `evaluation.png` | Precision-Recall curve (vs. random baseline) + Confusion Matrix |
| `threshold_analysis.png` | Precision, Recall, and F1 vs. decision threshold |
| `feature_importance.png` | Top 15 V-components by Gini importance |
| `feature_distributions.png` | Density plots of top 6 features split by class |

---

## Tech Stack

| Category | Tool |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| ML modeling | `scikit-learn` — `GradientBoostingClassifier`, `compute_sample_weight` |
| Environment | Python 3.8+, Jupyter Notebook |

---

## Project Structure

```
fraud-detection/
├── creditcard.csv              # Raw dataset (download from Kaggle — not committed)
├── fraud_detection.ipynb       # Full pipeline: EDA → preprocessing → modeling → evaluation
└── README.md
```

> The dataset is not committed to this repo due to its size (~150MB). Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root before running the notebook.

---

## How to Run

```bash
# 1. Clone the repo and install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# 2. Add creditcard.csv to the project root (download from Kaggle)

# 3. Launch the notebook
jupyter notebook fraud_detection.ipynb
```

---

## Production Extensions

This notebook is scoped as a rigorous baseline. In a production fintech setting, the natural next steps would be:

- **Faster boosting**: Replace `sklearn.GradientBoostingClassifier` with LightGBM or XGBoost for 10–100x training speedup and built-in `scale_pos_weight` for imbalance
- **Bayesian hyperparameter search**: Optuna or Ray Tune over a larger search space, more efficiently than grid search
- **Probability calibration**: `CalibratedClassifierCV` to ensure `predict_proba` scores reflect true probabilities — critical for cost-based threshold setting
- **Feature engineering**: Transaction velocity (# charges in last 60 min), time-since-last-transaction, merchant category patterns, cardholder spend baseline deviation
- **Concept drift monitoring**: Fraud patterns evolve; retrain on a rolling window, monitor AUPRC over time, alert when model performance degrades
- **Cost-sensitive threshold**: Formalize the threshold decision as `threshold = cost_FP / (cost_FP + cost_FN)` using actual dollar loss estimates

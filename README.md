# 🏠 Smart Home Energy Prediction

An end-to-end machine learning pipeline that forecasts household energy consumption using weather conditions, time-based features, and autoregressive lag signals. Three models are benchmarked — Linear Regression, Random Forest, and XGBoost — with full evaluation and visualisation.

---

## Project Structure

```
Smart Home Energy Prediction/
│
├── smart_home_energy.ipynb           # Main notebook — run this
├── smart_home_energy_usage_dataset.csv  # Dataset
├── requirements.txt                  # Python dependencies
│
└── outputs/                          # Generated after running the notebook
    ├── eda_overview.png
    ├── model_comparison.png
    ├── actual_vs_predicted.png
    ├── feature_importance.png
    └── residuals.png
```

---

## Setup

**1. Clone or download the project, then create a virtual environment:**

```bash
python -m venv .venv
```

**2. Activate it:**

```bash
# Windows
.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate
```

**3. Install dependencies:**

```bash
pip install -r requirements.txt
```

**4. Place the dataset in the project root** (same folder as the notebook), then open the notebook:

```bash
jupyter notebook smart_home_energy.ipynb
```

Run all cells top to bottom.

---

## Pipeline Overview

| Step | Description |
|------|-------------|
| **1 — Preprocessing** | Load CSV, parse datetime, sort per home, strip column names |
| **2 — EDA** | Correlation heatmap, diurnal load cycle, day-of-week distributions, temperature scatter |
| **3 — Feature Engineering** | Cyclical time encoding, lag features (1h, 2h), rolling mean |
| **4 — Model Training** | Linear Regression, Random Forest, XGBoost trained on chronological 80/20 split |
| **5 — Evaluation** | MAE, RMSE, R² comparison; actual vs predicted plots; feature importance; residual analysis |

---

## Models

### Linear Regression *(Baseline)*
A simple benchmark with no hyperparameters. Uses scaled features. Establishes a floor — if the more complex models can't beat this, something is wrong.

### Random Forest *(Primary)*
A tree-based ensemble that handles non-linear feature interactions naturally. Does not require feature scaling. Feature importances reveal which signals (lag vs weather vs time) drive predictions.

```
n_estimators    = 200
max_features    = "sqrt"
min_samples_leaf = 2
```

### XGBoost *(High-Achiever)*
Sequential gradient boosting — typically the most accurate model on tabular data. Uses early stopping to prevent overfitting automatically.

```
n_estimators          = 200
learning_rate         = 0.05
max_depth             = 4
early_stopping_rounds = 20
```

---

## 📊 Evaluation Metrics

| Metric | What it measures |
|--------|-----------------|
| **MAE** | Average error in kWh — easy to interpret |
| **RMSE** | Penalises large errors more heavily — good for detecting spikes |
| **R²** | Proportion of energy variance explained by the model (1.0 = perfect) |

---

## ⚠️ Key Design Decisions

**Chronological train/test split** — The dataset is split 80/20 by time, not randomly. Using `train_test_split()` on time-series data causes data leakage (training on the future to predict the past) and produces artificially inflated metrics.

**No `Lag_24h` feature** — The dataset contains ~5 rows per home (max 11). A 24-step lag would produce zero non-null values and wipe the entire dataset after `dropna()`. Lags are limited to 1h and 2h.

**Cyclical time encoding** — Raw hour values (0–23) imply hour 23 and hour 0 are maximally different. Sine/cosine encoding wraps the cycle so midnight is correctly adjacent to 11 PM.

**No `groupby().apply()` for imputation** — Pandas 3.x drops the group key column after `.apply()`. The dataset has zero missing values anyway, so imputation is skipped entirely.

---

## 📦 Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
```

Install with:
```bash
pip install -r requirements.txt
```
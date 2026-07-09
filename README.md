# Telco Customer Churn Prediction — MLOps Pipeline

**GitHub Repository: https://github.com/sina-moghadam/churn-prediction-flow**

---

## Overview

This project implements a standard, reproducible MLOps pipeline for predicting customer churn using the IBM Telco Customer Churn dataset. Data is managed through versioned stages, multiple machine learning models are trained and compared with full experiment tracking in MLflow, and the best-performing model is packaged as a containerized FastAPI service for inference.

The dataset contains records for roughly 7,000 customers of a telecom company. The goal is to predict `Churn Value`:
- `1` → the customer left (churned)
- `0` → the customer stayed

---

## Project Structure

```
final_prgg/
├── data/
│   ├── v1/
│   │   └── Telco_customer_churn.xlsx   # raw dataset (added manually, not generated)
│   ├── v2/                             # cleaned + encoded data (generated)
│   └── v3/                             # feature-engineered + scaled data (generated)
├── src/
│   ├── __init__.py
│   ├── config.py            # central configuration values
│   ├── logger.py            # shared logging setup
│   ├── mlflow_utils.py      # MLflow tracking URI / experiment setup
│   ├── data_loader.py       # loads the raw dataset (v1)
│   ├── preprocessing.py     # cleaning, imputation, categorical encoding -> v2
│   ├── features.py          # domain feature engineering, scaling -> v3
│   ├── train.py             # model tuning, training, MLflow logging
│   ├── evaluate.py          # metrics, threshold search, final test evaluation
│   ├── utils.py             # small IO helpers (CSV/JSON read/write)
│   └── api.py               # FastAPI prediction service
├── run_pipeline.py          # single entry point: runs the full pipeline end to end
├── requirements.txt
├── Dockerfile
└── .gitignore
```

---

## Step-by-Step: What This Pipeline Does

### Step 1 — Load raw data (v1)

`src/data_loader.py` reads the raw Excel file from `data/v1/Telco_customer_churn.xlsx`. This is the only dataset version that is not generated — it must be supplied manually and is treated as the immutable source of truth for the whole pipeline.

### Step 2 — Preprocess and clean (v2)

`src/preprocessing.py` takes the raw data and:
- Drops identifier, geography, and leakage-prone columns (customer ID, `Churn Label`, `Churn Score`, `CLTV`, etc.) — these either don't help prediction or would leak information about the target that wouldn't be available at prediction time.
- Coerces numeric columns and imputes missing values (median for numeric columns, mode for categorical columns), storing the exact imputation values used so the same values can be reapplied later at inference time without recomputing them.
- One-hot encodes categorical columns using a fixed, saved vocabulary, so training and inference always produce dataframes with identical columns.
- Writes the result to `data/v2/cleaned_data.csv`, along with a metadata JSON file (`data/v2/preprocessing_metadata.json`) describing every imputation value and category used.

### Step 3 — Feature engineering (v3)

`src/features.py` builds on top of v2 by:
- Adding domain-driven features: total active services, count of account-protection services, count of streaming services, numeric contract length in months, and average charge per tenure.
- Scaling continuous numeric columns with `StandardScaler`, fitting the scaler only on this stage and saving it to `data/v3/scaler.joblib` for reuse at inference time.
- Writing the result to `data/v3/featured_data.csv` with its own metadata file (`data/v3/feature_metadata.json`).

### Step 4 — Train and compare models

`src/train.py` trains four model families — Logistic Regression, Random Forest, XGBoost, and CatBoost — on **both** dataset versions (v2 and v3), so 8 total model/version combinations are evaluated.

For each combination:
- The data is split into train / validation / test sets with a fixed random seed, stratified on the target so class balance is preserved across splits.
- `GridSearchCV` with stratified k-fold cross-validation tunes each model's hyperparameters on the training split only.
- The tuned model is scored once on the validation split. Instead of assuming a default 0.5 decision threshold, the threshold that maximizes F1-score on the validation set is searched and stored.
- A weighted combination of ROC-AUC, F1, accuracy, and recall produces a single "overall score" used to compare every model/version combination on equal footing.
- Every run — model name, dataset version, hyperparameters, metrics, confusion matrix, feature importances, and the trained model artifact itself — is logged to MLflow.
- The single best-scoring combination across all 8 runs is recorded in `src/best_model_info.json`.

The test split is **never touched** during this stage — it's set aside for the final evaluation only.

### Step 5 — Final evaluation

`src/evaluate.py` loads the best model recorded in `src/best_model_info.json`, runs it against the untouched test split, computes final metrics (accuracy, precision, recall, F1, ROC-AUC), logs them to the corresponding MLflow run, and registers the model in the MLflow Model Registry under `telco-churn-classifier`.

### Step 6 — Serve predictions via API

`src/api.py` exposes the trained model as a FastAPI service:
- On startup, it loads the best model (using the correct MLflow flavor loader for CatBoost/XGBoost/sklearn), the preprocessing metadata, the feature metadata, and the fitted scaler — all cached so this only happens once per process.
- `GET /health` reports which model and dataset version are currently loaded.
- `POST /predict` accepts one or more raw customer records (same raw column names as the original dataset), runs them through the exact same cleaning → encoding → feature engineering → scaling pipeline used during training, and returns the predicted class plus churn/stay probabilities for each record.

---

## Setup and Local Execution

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows (PowerShell):
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the raw dataset

Place the raw Excel file at:
```
data/v1/Telco_customer_churn.xlsx
```

### 4. Run the full pipeline

```bash
python run_pipeline.py
```

This runs all five stages in order: load → preprocess → feature engineering → train & compare → evaluate. Console output logs progress for every stage, including per-model validation scores and the final selected model.

### 5. Serve the trained model

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Then check `http://localhost:8000/health`, or send a POST request to `http://localhost:8000/predict` with one or more raw customer records.

---

## Running with Docker

The Dockerfile packages the project into a container with all dependencies pre-installed.

Build the image:
```bash
docker build -t churn-app .
```

Run the full training pipeline (default container command):
```bash
docker run churn-app
```

Serve the API instead of training:
```bash
docker run -p 8000:8000 churn-app uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Note: the raw dataset (`data/v1/Telco_customer_churn.xlsx`) must be present in the build context before running `docker build`, since it gets copied into the image and is required for Stage 1 to succeed inside the container.

---

## Experiment Tracking with MLflow

Every training run — across every model family and every dataset version — is logged locally under `mlruns/`. To browse experiments visually:

```bash
mlflow ui
```

Then open `http://localhost:5000` in a browser to compare runs, metrics, parameters, and artifacts (confusion matrices, feature importances) side by side.

---

## Version Control

All source code, configuration, and generated dataset versions (`data/v2`, `data/v3`) are tracked in this Git repository, with `.venv/` and `mlruns/` excluded via `.gitignore` since they are either environment-specific or regenerable/local tracking data.

---

## Notes

- `data/v2` and `data/v3` are generated by the pipeline and can always be regenerated by rerunning `run_pipeline.py` against `data/v1`.
- The decision threshold used at inference time is the one that maximized F1-score on the validation split during training, not a fixed 0.5 cutoff — this is stored per model and applied consistently between training, evaluation, and the live API.
- Because preprocessing and feature-engineering metadata (imputation values, category vocabularies, the fitted scaler) are saved during training and only ever reused — never refit — at inference time, predictions made through the API are guaranteed to be transformed identically to how the training data was transformed.

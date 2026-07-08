from __future__ import annotations

RANDOM_SEED = 42

TARGET_COLUMN = "Churn Value"

DROP_COLUMNS = [
    "CustomerID",
    "customerID",
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
    "Churn Reason",
    "Churn Label",
    "Churn Score",
    "CLTV",
]

NUMERIC_COLUMNS = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
]

SERVICE_COLUMNS = [
    "Phone Service",
    "Multiple Lines",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
]

PROTECTION_COLUMNS = [
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
]

STREAMING_COLUMNS = [
    "Streaming TV",
    "Streaming Movies",
]

CONTRACT_LENGTH_MONTHS = {
    "Month-to-month": 1,
    "One year": 12,
    "Two year": 24,
}

DATA_PATHS = {
    "raw": "data/v1/Telco_customer_churn.xlsx",
    "v2": "data/v2/cleaned_data.csv",
    "v2_metadata": "data/v2/preprocessing_metadata.json",
    "v3": "data/v3/featured_data.csv",
    "v3_metadata": "data/v3/feature_metadata.json",
    "v3_scaler": "data/v3/scaler.joblib",
}

BEST_MODEL_INFO_PATH = "src/best_model_info.json"

REGISTERED_MODEL_NAME = "telco-churn-classifier"

DATASET_VERSIONS = {
    "v2": DATA_PATHS["v2"],
    "v3": DATA_PATHS["v3"],
}

TRAIN_TEST_SPLIT = {
    "test_size": 0.15,
    "validation_size": 0.1765,
}

CV_SPLITS = 5

MODEL_SELECTION_METRIC_WEIGHTS = {
    "roc_auc": 0.30,
    "f1": 0.30,
    "accuracy": 0.20,
    "recall": 0.20,
}

DECISION_THRESHOLD_SEARCH = {
    "min": 0.05,
    "max": 0.95,
    "step": 0.01,
}

PARAM_GRIDS = {
    "LogisticRegression": {
        "C": [0.1, 1.0, 10.0],
        "class_weight": [None, "balanced"],
    },
    "RandomForest": {
        "n_estimators": [200, 400],
        "max_depth": [None, 10],
        "class_weight": [None, "balanced"],
    },
    "XGBoost": {
        "n_estimators": [200, 400],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
    },
    "CatBoost": {
        "iterations": [200, 400],
        "depth": [4, 6],
        "learning_rate": [0.05, 0.1],
    },
}

API_HOST = "0.0.0.0"
API_PORT = 8000
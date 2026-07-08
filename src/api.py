"""Prediction endpoint for the champion churn model, served via FastAPI.

Start the service with:

    uvicorn src.api:app --host 0.0.0.0 --port 8000

Send raw customer records (same column names as the source dataset, minus
any identifier columns) as a POST body to /predict. See README.md for a
worked example of the request and response shapes.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import joblib
import mlflow.catboost
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, HTTPException

from src import config
from src.features import transform_encoded_record_for_inference
from src.logger import configure_logging
from src.mlflow_utils import setup_mlflow
from src.preprocessing import clean_dataframe, encode_categorical_features
from src.utils import read_json

configure_logging()
LOGGER = logging.getLogger(__name__)

app = FastAPI(title="Telco Customer Churn Prediction API", version="1.0.0")


_FLAVOR_LOADERS = {
    "CatBoost": mlflow.catboost.load_model,
    "XGBoost": mlflow.xgboost.load_model,
}


def _load_native_model(model_name: str, run_id: str) -> Any:

    model_uri = f"runs:/{run_id}/model"
    loader = _FLAVOR_LOADERS.get(model_name, mlflow.sklearn.load_model)
    return loader(model_uri)


@lru_cache(maxsize=1)
def _runtime_resources() -> dict[str, Any]:

    setup_mlflow()

    best_info = read_json(config.BEST_MODEL_INFO_PATH)
    preprocessing_metadata = read_json(config.DATA_PATHS["v2_metadata"])
    feature_metadata = read_json(config.DATA_PATHS["v3_metadata"])

    scaler = (
        joblib.load(config.DATA_PATHS["v3_scaler"])
        if feature_metadata.get("scaled_columns")
        else None
    )

    model = _load_native_model(best_info["model_name"], best_info["run_id"])

    LOGGER.info(
        "Loaded model %s (version=%s, run_id=%s) for serving.",
        best_info["model_name"],
        best_info["dataset_version"],
        best_info["run_id"],
    )

    return {
        "model": model,
        "model_name": best_info["model_name"],
        "dataset_version": best_info["dataset_version"],
        "decision_threshold": float(best_info["decision_threshold"]),
        "preprocessing_metadata": preprocessing_metadata,
        "feature_metadata": feature_metadata,
        "scaler": scaler,
    }


def _parse_records(payload: Any) -> list[dict[str, Any]]:

    if isinstance(payload, dict) and "records" in payload:
        payload = payload["records"]

    if isinstance(payload, dict):
        records = [payload]
    elif isinstance(payload, list):
        records = payload
    else:
        raise HTTPException(status_code=422, detail="Request body must be a JSON object or list.")

    if not records or not all(isinstance(record, dict) for record in records):
        raise HTTPException(status_code=422, detail="Each record must be a JSON object.")

    return records


def _build_feature_matrix(records: list[dict[str, Any]], resources: dict[str, Any]) -> pd.DataFrame:
    raw_dataframe = pd.DataFrame.from_records(records)

    cleaned, _ = clean_dataframe(
        raw_dataframe,
        fit_metadata=False,
        metadata=resources["preprocessing_metadata"],
        include_target=False,
    )
    encoded = encode_categorical_features(cleaned, resources["preprocessing_metadata"])

    return transform_encoded_record_for_inference(
        encoded, resources["feature_metadata"], resources["scaler"]
    )


def _score(features: pd.DataFrame, resources: dict[str, Any]) -> list[dict[str, Any]]:
    """Run inference and package each row into a prediction dict."""
    model = resources["model"]
    if not hasattr(model, "predict_proba"):
        raise HTTPException(status_code=500, detail="Loaded model does not expose predict_proba.")

    probabilities = np.asarray(model.predict_proba(features))
    threshold = resources["decision_threshold"]
    churned_probability = probabilities[:, 1]
    predicted_classes = (churned_probability >= threshold).astype(int)

    return [
        {
            "predicted_class": int(predicted_classes[row]),
            "decision_threshold": float(threshold),
            "probability_stayed": float(probabilities[row, 0]),
            "probability_churned": float(probabilities[row, 1]),
        }
        for row in range(len(predicted_classes))
    ]


@app.get("/health")
def health() -> dict[str, str]:
    resources = _runtime_resources()
    return {
        "status": "ok",
        "model_name": resources["model_name"],
        "dataset_version": resources["dataset_version"],
    }


@app.post("/predict")
def predict(payload: Any = Body(...)) -> dict[str, Any]:
    resources = _runtime_resources()
    records = _parse_records(payload)
    features = _build_feature_matrix(records, resources)
    predictions = _score(features, resources)
    return {"predictions": predictions}
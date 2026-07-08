from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src import config
from src.logger import configure_logging
from src.mlflow_utils import setup_mlflow
from src.utils import read_json

configure_logging()
LOGGER = logging.getLogger(__name__)


def predict_positive_class_scores(model: Any, features: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(features))[:, 1]
    return np.asarray(model.predict(features)).astype(float)


def compute_classification_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def compute_overall_score(metrics_with_roc_auc: dict[str, float]) -> float:
    weights = config.MODEL_SELECTION_METRIC_WEIGHTS
    return float(sum(weights[name] * metrics_with_roc_auc[name] for name in weights))


def optimize_decision_threshold(y_true: pd.Series, scores: np.ndarray) -> tuple[float, float]:
    search = config.DECISION_THRESHOLD_SEARCH
    thresholds = np.arange(search["min"], search["max"], search["step"])

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)
        if len(np.unique(predictions)) < 1:
            continue
        score = f1_score(y_true, predictions, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, best_f1


def _log_json_artifact(payload: dict[str, Any], file_name: str) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, file_name)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        mlflow.log_artifact(path)


def save_confusion_matrix_artifact(y_true: pd.Series, y_pred: np.ndarray, file_name: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    payload = {
        "true_negative": int(cm[0][0]),
        "false_positive": int(cm[0][1]),
        "false_negative": int(cm[1][0]),
        "true_positive": int(cm[1][1]),
    }
    _log_json_artifact(payload, file_name)


def save_feature_importance_artifact(
    model: Any, feature_names: list[str], file_name: str
) -> None:
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        importances = np.asarray(model.coef_).ravel()

    if importances is None or len(importances) != len(feature_names):
        return

    ranked = sorted(zip(feature_names, importances.tolist()), key=lambda item: -abs(item[1]))
    payload = {"feature_importance": [{"feature": f, "importance": v} for f, v in ranked]}
    _log_json_artifact(payload, file_name)


def evaluate_best_model() -> float:
    setup_mlflow()

    if not os.path.exists(config.BEST_MODEL_INFO_PATH):
        raise FileNotFoundError(
            f"{config.BEST_MODEL_INFO_PATH} not found. Run train_models() first."
        )

    best_info = read_json(config.BEST_MODEL_INFO_PATH)
    run_id = best_info["run_id"]
    model_name = best_info["model_name"]
    dataset_version = best_info["dataset_version"]
    decision_threshold = best_info["decision_threshold"]
    test_data_path = best_info["test_data_path"]

    LOGGER.info(
        "Evaluating best model: %s (version=%s, run_id=%s)",
        model_name,
        dataset_version,
        run_id,
    )

    test_df = pd.read_csv(test_data_path)
    y_test = test_df[config.TARGET_COLUMN]
    x_test = test_df.drop(columns=[config.TARGET_COLUMN])

    model_uri = f"runs:/{run_id}/model"
    model = mlflow.pyfunc.load_model(model_uri)

    raw_scores = np.asarray(model.predict(x_test))
    if raw_scores.ndim > 1:
        raw_scores = raw_scores[:, -1]
    y_pred = (raw_scores >= decision_threshold).astype(int)

    metrics = compute_classification_metrics(y_test, y_pred)
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_test, raw_scores))
    except ValueError:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_pred))

    LOGGER.info("Final test metrics: %s", json.dumps(metrics, indent=2))

    with mlflow.start_run(run_id=run_id):
        for metric_name, value in metrics.items():
            mlflow.log_metric(f"test_{metric_name}", value)
        mlflow.set_tag("final_selected_model", "true")
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("dataset_version", dataset_version)
        save_confusion_matrix_artifact(y_test, y_pred, "test_confusion_matrix.json")

        try:
            mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=config.REGISTERED_MODEL_NAME)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Model registry step skipped: %s", exc)

    return metrics["f1"]


if __name__ == "__main__":
    evaluate_best_model()
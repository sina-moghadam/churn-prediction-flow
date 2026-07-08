from __future__ import annotations

import logging
import os
import time
from typing import Any

import mlflow
import mlflow.catboost
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from src import config
from src.evaluate import (
    compute_classification_metrics,
    compute_overall_score,
    optimize_decision_threshold,
    predict_positive_class_scores,
    save_confusion_matrix_artifact,
    save_feature_importance_artifact,
)
from src.logger import configure_logging
from src.mlflow_utils import setup_mlflow
from src.utils import read_json, write_json

configure_logging()
LOGGER = logging.getLogger(__name__)


def _model_specs() -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    return {
        "LogisticRegression": (
            LogisticRegression(max_iter=5000, random_state=config.RANDOM_SEED),
            config.PARAM_GRIDS["LogisticRegression"],
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=config.RANDOM_SEED),
            config.PARAM_GRIDS["RandomForest"],
        ),
        "XGBoost": (
            XGBClassifier(eval_metric="logloss", random_state=config.RANDOM_SEED),
            config.PARAM_GRIDS["XGBoost"],
        ),
        "CatBoost": (
            CatBoostClassifier(verbose=0, random_state=config.RANDOM_SEED),
            config.PARAM_GRIDS["CatBoost"],
        ),
    }


def _split_dataset(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    y = dataframe[config.TARGET_COLUMN]
    x = dataframe.drop(columns=[config.TARGET_COLUMN])

    test_size = config.TRAIN_TEST_SPLIT["test_size"]
    val_size_of_remainder = config.TRAIN_TEST_SPLIT["validation_size"]

    x_temp, x_test, y_temp, y_test = train_test_split(
        x, y, test_size=test_size, random_state=config.RANDOM_SEED, stratify=y
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp,
        y_temp,
        test_size=val_size_of_remainder,
        random_state=config.RANDOM_SEED,
        stratify=y_temp,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def train_on_version(data_path: str, dataset_version: str, global_best: dict[str, Any]) -> None:
    LOGGER.info("===== Training on dataset version: %s (%s) =====", dataset_version, data_path)
    dataframe = pd.read_csv(data_path)
    x_train, x_val, x_test, y_train, y_val, y_test = _split_dataset(dataframe)

    test_data_path = f"data/test_data_{dataset_version}.csv"
    pd.concat([x_test, y_test], axis=1).to_csv(test_data_path, index=False)

    cv = StratifiedKFold(n_splits=config.CV_SPLITS, shuffle=True, random_state=config.RANDOM_SEED)

    for model_name, (estimator, param_grid) in _model_specs().items():
        LOGGER.info("--- Tuning %s on %s ---", model_name, dataset_version)
        start_time = time.time()

        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(x_train, y_train)
        best_estimator = search.best_estimator_
        elapsed_seconds = time.time() - start_time

        val_scores = predict_positive_class_scores(best_estimator, x_val)
        decision_threshold, _ = optimize_decision_threshold(y_val, val_scores)
        y_val_pred = (val_scores >= decision_threshold).astype(int)

        val_metrics = compute_classification_metrics(y_val, y_val_pred)
        try:
            val_metrics["roc_auc"] = float(roc_auc_score(y_val, val_scores))
        except ValueError:
            val_metrics["roc_auc"] = float(roc_auc_score(y_val, y_val_pred))

        overall_score = compute_overall_score(val_metrics)

        with mlflow.start_run(run_name=f"{model_name}_{dataset_version}") as run:
            mlflow.set_tag("dataset_version", dataset_version)
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("dataset_version", dataset_version)
            mlflow.log_param("random_state", config.RANDOM_SEED)
            mlflow.log_param("cv_folds", config.CV_SPLITS)
            mlflow.log_params({f"best_{k}": v for k, v in search.best_params_.items()})

            mlflow.log_metric("cv_best_roc_auc", float(search.best_score_))
            mlflow.log_metric("training_time_seconds", elapsed_seconds)
            mlflow.log_metric("decision_threshold", decision_threshold)
            mlflow.log_metric("overall_score", overall_score)
            for metric_name, value in val_metrics.items():
                mlflow.log_metric(f"val_{metric_name}", value)

            save_confusion_matrix_artifact(
                y_val, y_val_pred, f"{model_name}_{dataset_version}_val_confusion_matrix.json"
            )
            save_feature_importance_artifact(
                best_estimator,
                list(x_train.columns),
                f"{model_name}_{dataset_version}_feature_importance.json",
            )

            if model_name == "CatBoost":
                mlflow.catboost.log_model(best_estimator, "model")
            elif model_name == "XGBoost":
                mlflow.xgboost.log_model(best_estimator, "model")
            else:
                mlflow.sklearn.log_model(best_estimator, "model")

            LOGGER.info(
                "%s [%s] -> overall_score=%.4f val_f1=%.4f val_roc_auc=%.4f threshold=%.3f",
                model_name,
                dataset_version,
                overall_score,
                val_metrics["f1"],
                val_metrics["roc_auc"],
                decision_threshold,
            )

            if overall_score > global_best.get("overall_score", -1.0):
                global_best.update(
                    {
                        "run_id": run.info.run_id,
                        "model_name": model_name,
                        "dataset_version": dataset_version,
                        "overall_score": overall_score,
                        "decision_threshold": decision_threshold,
                        "test_data_path": test_data_path,
                        "validation_metrics": val_metrics,
                    }
                )


def train_models() -> None:
    setup_mlflow()

    global_best: dict[str, Any] = {"overall_score": -1.0}
    if os.path.exists(config.BEST_MODEL_INFO_PATH):
        global_best.update(read_json(config.BEST_MODEL_INFO_PATH))
        global_best.setdefault("overall_score", -1.0)

    for dataset_version, data_path in config.DATASET_VERSIONS.items():
        if os.path.exists(data_path):
            train_on_version(data_path, dataset_version, global_best)
        else:
            LOGGER.warning("Skipping %s: %s not found.", dataset_version, data_path)

    if global_best.get("run_id"):
        write_json(config.BEST_MODEL_INFO_PATH, global_best)
        LOGGER.info(
            "Best overall model: %s on %s (overall_score=%.4f)",
            global_best["model_name"],
            global_best["dataset_version"],
            global_best["overall_score"],
        )
    else:
        LOGGER.error("No model was trained successfully.")


if __name__ == "__main__":
    train_models()
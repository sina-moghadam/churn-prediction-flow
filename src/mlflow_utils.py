from __future__ import annotations

import logging
import os

import mlflow

from src.logger import configure_logging

configure_logging()
LOGGER = logging.getLogger(__name__)

EXPERIMENT_NAME = "telco-customer-churn"


def setup_mlflow(tracking_dir: str = "mlruns") -> None:
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking_uri = f"file:{os.path.abspath(tracking_dir)}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    LOGGER.info("MLflow tracking URI set to %s", tracking_uri)
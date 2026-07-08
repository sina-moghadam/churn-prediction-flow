from __future__ import annotations

import logging

from src.data_loader import load_data
from src.evaluate import evaluate_best_model
from src.features import engineer_features
from src.logger import configure_logging
from src.preprocessing import preprocess_data
from src.train import train_models

configure_logging()
LOGGER = logging.getLogger(__name__)


def main() -> None:
    LOGGER.info("========== STAGE 1/5: Load raw data (v1) ==========")
    load_data()

    LOGGER.info("========== STAGE 2/5: Preprocess -> v2 ==========")
    preprocess_data()

    LOGGER.info("========== STAGE 3/5: Feature engineering -> v3 ==========")
    engineer_features()

    LOGGER.info("========== STAGE 4/5: Train & compare models (MLflow) ==========")
    train_models()

    LOGGER.info("========== STAGE 5/5: Evaluate best model on test set ==========")
    evaluate_best_model()

    LOGGER.info("Pipeline finished successfully.")


if __name__ == "__main__":
    main()
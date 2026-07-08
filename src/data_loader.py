from __future__ import annotations

import logging
import os

import pandas as pd

from src import config
from src.logger import configure_logging

configure_logging()
LOGGER = logging.getLogger(__name__)


def load_data(file_path: str = config.DATA_PATHS["raw"]) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    LOGGER.info("Loading raw dataset from %s", file_path)
    dataframe = pd.read_excel(file_path)
    LOGGER.info("Loaded raw dataset with shape %s", dataframe.shape)

    return dataframe


if __name__ == "__main__":
    load_data()

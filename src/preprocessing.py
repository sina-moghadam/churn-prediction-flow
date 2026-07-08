from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src import config
from src.logger import configure_logging
from src.utils import load_dataframe, save_dataframe, write_json

configure_logging()
LOGGER = logging.getLogger(__name__)


def _coerce_numeric_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()
    for column in config.NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column].replace(" ", np.nan), errors="coerce")
    return df


def clean_dataframe(
    dataframe: pd.DataFrame,
    fit_metadata: bool,
    metadata: dict[str, Any] | None = None,
    include_target: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = dataframe.copy()
    df = _coerce_numeric_columns(df)

    columns_to_drop = [column for column in config.DROP_COLUMNS if column in df.columns]
    df = df.drop(columns=columns_to_drop, errors="ignore")

    target_present = config.TARGET_COLUMN in df.columns
    if include_target and target_present:
        if df[config.TARGET_COLUMN].dtype == object:
            df[config.TARGET_COLUMN] = df[config.TARGET_COLUMN].map({"Yes": 1, "No": 0})
        df = df.dropna(subset=[config.TARGET_COLUMN])
        df[config.TARGET_COLUMN] = df[config.TARGET_COLUMN].astype(int)
    elif config.TARGET_COLUMN in df.columns:
        df = df.drop(columns=[config.TARGET_COLUMN])

    feature_columns = [column for column in df.columns if column != config.TARGET_COLUMN]
    numeric_columns = [c for c in feature_columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_columns = [c for c in feature_columns if c not in numeric_columns]

    if fit_metadata:
        numeric_impute_values = {
            column: float(df[column].median()) if pd.notna(df[column].median()) else 0.0
            for column in numeric_columns
        }
        categorical_impute_values = {}
        categorical_categories: dict[str, list[str]] = {}
        for column in categorical_columns:
            non_null = df[column].dropna().astype(str)
            mode = non_null.mode()
            categorical_impute_values[column] = str(mode.iloc[0]) if not mode.empty else "Unknown"
            categorical_categories[column] = sorted(non_null.unique().tolist())

        metadata = {
            "target_column": config.TARGET_COLUMN,
            "dropped_columns": sorted(columns_to_drop),
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "numeric_impute_values": numeric_impute_values,
            "categorical_impute_values": categorical_impute_values,
            "categorical_categories": categorical_categories,
        }
    else:
        if metadata is None:
            raise ValueError("metadata must be provided when fit_metadata=False")
        numeric_columns = metadata["numeric_columns"]
        categorical_columns = metadata["categorical_columns"]

    numeric_values = metadata["numeric_impute_values"]
    categorical_values = metadata["categorical_impute_values"]

    for column in numeric_columns:
        if column not in df.columns:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(numeric_values[column])

    for column in categorical_columns:
        if column not in df.columns:
            df[column] = np.nan
        df[column] = (
            df[column]
            .astype("object")
            .where(df[column].notna(), categorical_values[column])
            .astype(str)
            .str.strip()
        )

    ordered_columns = list(numeric_columns) + list(categorical_columns)
    if include_target and target_present:
        ordered_columns.append(config.TARGET_COLUMN)
    df = df[ordered_columns]

    LOGGER.info("Cleaned dataframe shape: %s", df.shape)
    return df, metadata


def encode_categorical_features(
    dataframe: pd.DataFrame,
    metadata: dict[str, Any],
) -> pd.DataFrame:
    df = dataframe.copy()
    target_column = metadata["target_column"]
    target = df[target_column] if target_column in df.columns else None
    if target is not None:
        df = df.drop(columns=[target_column])

    encoded_parts = [df[metadata["numeric_columns"]]]
    for column in metadata["categorical_columns"]:
        categories = metadata["categorical_categories"][column]
        dummies = pd.get_dummies(df[column], prefix=column, dtype=int)
        expected_columns = [f"{column}_{cat}" for cat in categories]
        for expected_column in expected_columns:
            if expected_column not in dummies.columns:
                dummies[expected_column] = 0
        dummies = dummies[expected_columns]
        encoded_parts.append(dummies)

    encoded = pd.concat(encoded_parts, axis=1)
    if target is not None:
        encoded[target_column] = target.values
    LOGGER.info("Encoded dataframe shape: %s", encoded.shape)
    return encoded


def preprocess_data(
    input_path: str = config.DATA_PATHS["raw"],
    output_path: str = config.DATA_PATHS["v2"],
    metadata_path: str = config.DATA_PATHS["v2_metadata"],
) -> pd.DataFrame:
    LOGGER.info("Starting preprocessing: %s -> %s", input_path, output_path)
    raw_dataframe = pd.read_excel(input_path)

    cleaned_dataframe, metadata = clean_dataframe(
        raw_dataframe, fit_metadata=True, include_target=True
    )
    encoded_dataframe = encode_categorical_features(cleaned_dataframe, metadata)

    leaked = [c for c in encoded_dataframe.columns if "Churn Label" in c or "Churn Score" in c]
    if leaked:
        raise RuntimeError(f"Leakage columns still present after cleaning: {leaked}")

    metadata["feature_columns_v2"] = [
        c for c in encoded_dataframe.columns if c != config.TARGET_COLUMN
    ]

    save_dataframe(encoded_dataframe, output_path)
    write_json(metadata_path, metadata)
    LOGGER.info("Preprocessing completed. Final shape: %s", encoded_dataframe.shape)
    return encoded_dataframe


if __name__ == "__main__":
    preprocess_data()
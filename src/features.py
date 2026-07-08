from __future__ import annotations

import logging
from typing import Any

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import config
from src.logger import configure_logging
from src.utils import load_dataframe, save_dataframe, write_json

configure_logging()
LOGGER = logging.getLogger(__name__)


def _find_dummy_column(dataframe: pd.DataFrame, prefix: str, value: str) -> str | None:
    column_name = f"{prefix}_{value}"
    return column_name if column_name in dataframe.columns else None


def add_domain_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()

    def _count_active(columns: list[str]) -> pd.Series:
        total = pd.Series(0, index=df.index, dtype=int)
        for column in columns:
            dummy_column = _find_dummy_column(df, column, "Yes")
            if dummy_column is not None:
                total = total + df[dummy_column]
        return total

    df["total_active_services"] = _count_active(config.SERVICE_COLUMNS)
    df["protection_services_count"] = _count_active(config.PROTECTION_COLUMNS)
    df["streaming_services_count"] = _count_active(config.STREAMING_COLUMNS)

    contract_length = pd.Series(1, index=df.index, dtype=int)
    for label, months in config.CONTRACT_LENGTH_MONTHS.items():
        if label == "Month-to-month":
            continue
        dummy_column = _find_dummy_column(df, "Contract", label)
        if dummy_column is not None:
            contract_length = contract_length.where(df[dummy_column] == 0, months)
    df["contract_length_months"] = contract_length

    if "Total Charges" in df.columns and "Tenure Months" in df.columns:
        df["avg_charge_per_tenure"] = df["Total Charges"] / (df["Tenure Months"] + 1e-5)

    return df


def _continuous_columns_to_scale(dataframe: pd.DataFrame, target_column: str) -> list[str]:
    candidates = dataframe.select_dtypes(include=["float64", "int64"]).columns
    return [
        column
        for column in candidates
        if column != target_column and dataframe[column].nunique() > 2
    ]


def engineer_features(
    input_path: str = config.DATA_PATHS["v2"],
    output_path: str = config.DATA_PATHS["v3"],
    metadata_path: str = config.DATA_PATHS["v3_metadata"],
    scaler_path: str = config.DATA_PATHS["v3_scaler"],
) -> pd.DataFrame:
    LOGGER.info("Starting feature engineering: %s -> %s", input_path, output_path)
    df = load_dataframe(input_path)

    df = add_domain_features(df)

    columns_to_scale = _continuous_columns_to_scale(df, config.TARGET_COLUMN)
    scaler = StandardScaler()
    if columns_to_scale:
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
        joblib.dump(scaler, scaler_path)
        LOGGER.info("Scaled columns: %s", columns_to_scale)

    feature_columns = [c for c in df.columns if c != config.TARGET_COLUMN]
    write_json(
        metadata_path,
        {
            "feature_columns_v3": feature_columns,
            "scaled_columns": columns_to_scale,
            "scaler_path": scaler_path,
        },
    )

    save_dataframe(df, output_path)
    LOGGER.info("Feature engineering completed. Final shape: %s", df.shape)
    return df


def transform_encoded_record_for_inference(
    encoded_dataframe: pd.DataFrame,
    feature_metadata: dict[str, Any],
    scaler: StandardScaler | None,
) -> pd.DataFrame:
    df = add_domain_features(encoded_dataframe)

    feature_columns = feature_metadata["feature_columns_v3"]
    for column in feature_columns:
        if column not in df.columns:
            df[column] = 0
    df = df[feature_columns]

    scaled_columns = feature_metadata.get("scaled_columns", [])
    if scaler is not None and scaled_columns:
        df[scaled_columns] = scaler.transform(df[scaled_columns])

    return df


if __name__ == "__main__":
    engineer_features()
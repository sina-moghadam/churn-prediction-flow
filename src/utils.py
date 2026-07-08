from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def save_dataframe(dataframe: pd.DataFrame, path: str) -> None:
    ensure_parent_dir(path)
    dataframe.to_csv(path, index=False)


def load_dataframe(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def write_json(path: str, payload: dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
"""Cleaning utilities for batch and streaming telecom telemetry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

TIMESTAMP_COLUMN = "timestamp"
REQUIRED_COLUMNS = {
    "tower_id",
    "cell_id",
    "connected_users",
    "download_traffic_mb",
    "upload_traffic_mb",
    "bandwidth_utilization",
    "latency_ms",
    "packet_loss_percent",
}
NUMERIC_COLUMNS = [
    "connected_users",
    "download_traffic_mb",
    "upload_traffic_mb",
    "bandwidth_utilization",
    "latency_ms",
    "packet_loss_percent",
    "download_speed_mbps",
    "upload_speed_mbps",
    "signal_strength_dbm",
]


def _validate_columns(frame: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Telemetry is missing required columns: {', '.join(missing)}")


def clean_telemetry(frame: pd.DataFrame, tower_info: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return a deterministic, typed, deduplicated telemetry frame.

    Missing numeric values are interpolated within each cell and then filled
    with the column median. Values outside physical KPI bounds are clipped.
    """
    _validate_columns(frame)
    cleaned = frame.copy()
    cleaned[TIMESTAMP_COLUMN] = pd.to_datetime(cleaned[TIMESTAMP_COLUMN], errors="coerce")
    cleaned = cleaned.dropna(subset=[TIMESTAMP_COLUMN, "tower_id", "cell_id"])
    cleaned = cleaned.drop_duplicates(subset=[TIMESTAMP_COLUMN, "cell_id"], keep="last")

    for column in NUMERIC_COLUMNS:
        if column in cleaned:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
            cleaned[column] = cleaned.groupby("cell_id", group_keys=False)[column].transform(
                lambda values: values.interpolate(limit_direction="both")
            )
            cleaned[column] = cleaned[column].fillna(cleaned[column].median()).fillna(0.0)

    bounds = {
        "connected_users": (0, None),
        "download_traffic_mb": (0, None),
        "upload_traffic_mb": (0, None),
        "bandwidth_utilization": (0, 100),
        "latency_ms": (0, None),
        "packet_loss_percent": (0, 100),
        "download_speed_mbps": (0, None),
        "upload_speed_mbps": (0, None),
        "signal_strength_dbm": (-150, 0),
    }
    for column, (lower, upper) in bounds.items():
        if column in cleaned:
            cleaned[column] = cleaned[column].clip(lower=lower, upper=upper)

    if tower_info is not None:
        metadata = tower_info.drop_duplicates("cell_id")
        metadata_columns = [
            column for column in metadata.columns if column not in cleaned.columns or column in {"latitude", "longitude"}
        ]
        cleaned = cleaned.merge(metadata[["cell_id", *metadata_columns]], on="cell_id", how="left")

    return cleaned.sort_values([TIMESTAMP_COLUMN, "cell_id"]).reset_index(drop=True)


@dataclass
class TelemetryCleaner:
    """Reusable cleaner for incoming micro-batches."""

    tower_info: pd.DataFrame | None = None

    def transform(self, records: pd.DataFrame | Iterable[dict]) -> pd.DataFrame:
        frame = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
        return clean_telemetry(frame, self.tower_info)


def load_and_clean(metrics_path, tower_info_path=None) -> pd.DataFrame:
    metrics = pd.read_csv(metrics_path)
    tower_info = pd.read_csv(tower_info_path) if tower_info_path else None
    return clean_telemetry(metrics, tower_info)

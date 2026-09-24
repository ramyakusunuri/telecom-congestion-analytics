"""Feature engineering and rule-based congestion scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.config_loader import get_config


def _normalise(series: pd.Series, reference: float) -> pd.Series:
    return (series / max(reference, 1e-9)).clip(0, 1)


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add temporal, traffic, load, rolling and congestion features."""
    if "timestamp" not in frame or "cell_id" not in frame:
        raise ValueError("Features require timestamp and cell_id columns")
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    result = result.sort_values(["cell_id", "timestamp"]).reset_index(drop=True)

    result["total_traffic_mb"] = result["download_traffic_mb"] + result["upload_traffic_mb"]
    if "capacity_users" in result:
        result["user_load_ratio"] = (
            result["connected_users"] / result["capacity_users"].replace(0, np.nan)
        ).fillna(0).clip(0, 2)
    else:
        result["user_load_ratio"] = result["connected_users"] / result["connected_users"].quantile(0.95).clip(lower=1)
        result["user_load_ratio"] = result["user_load_ratio"].clip(0, 2)
    result["traffic_norm"] = _normalise(result["total_traffic_mb"], result["total_traffic_mb"].quantile(0.95))
    config = get_config()
    congestion_config = config["congestion"]
    weights = congestion_config["weights"]
    result["latency_norm"] = _normalise(result["latency_ms"], congestion_config["latency_reference_ms"])
    result["packet_loss_norm"] = _normalise(
        result["packet_loss_percent"], congestion_config["packet_loss_reference_pct"]
    )
    result["user_load_norm"] = (result["user_load_ratio"] / 1.0).clip(0, 1)
    result["congestion_score"] = (
        result["bandwidth_utilization"] / 100 * weights["bandwidth_utilization"]
        + result["user_load_norm"] * weights["user_load_ratio"]
        + result["traffic_norm"] * weights["traffic_norm"]
        + result["latency_norm"] * weights["latency_norm"]
        + result["packet_loss_norm"] * weights["packet_loss_norm"]
    ) * 100
    thresholds = congestion_config["thresholds"]
    result["congestion_level"] = pd.cut(
        result["congestion_score"],
        bins=[-np.inf, thresholds["low_max"], thresholds["moderate_max"], thresholds["high_max"], np.inf],
        labels=["LOW", "MODERATE", "HIGH", "CRITICAL"],
    ).astype(str)
    result["hour"] = result["timestamp"].dt.hour
    result["day_of_week"] = result["timestamp"].dt.dayofweek
    result["is_weekend"] = result["day_of_week"] >= 5
    grouped = result.groupby("cell_id", group_keys=False)
    for column in ["congestion_score", "bandwidth_utilization", "latency_ms", "packet_loss_percent"]:
        result[f"{column}_rolling_mean_3"] = grouped[column].transform(
            lambda values: values.rolling(3, min_periods=1).mean()
        )
    return result


# Friendly alias used by pipeline callers.
engineer_features = add_features

"""End-to-end local preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.config_loader import get_path
from .clean_data import clean_telemetry
from .feature_engineering import add_features


def run_pipeline(metrics_path=None, tower_info_path=None, output_path=None) -> pd.DataFrame:
    metrics_file = Path(metrics_path or get_path("network_metrics_file"))
    tower_file = Path(tower_info_path or get_path("tower_info_file"))
    output_file = Path(output_path or get_path("processed_features_csv"))
    frame = clean_telemetry(pd.read_csv(metrics_file), pd.read_csv(tower_file))
    enriched = add_features(frame)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_file, index=False)
    try:
        enriched.to_parquet(get_path("processed_features_file"), index=False)
    except (ImportError, ValueError):
        pass
    return enriched


if __name__ == "__main__":
    result = run_pipeline()
    print(f"Wrote {len(result):,} processed telemetry rows")

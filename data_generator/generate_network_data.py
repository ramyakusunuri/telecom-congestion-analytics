"""
generate_network_data.py
-------------------------
Generates SYNTHETIC/SIMULATED telecom network performance data
(time series, per cell) with realistic causal relationships:

    more connected users
        -> more traffic
            -> higher bandwidth utilization
                -> higher latency
                    -> higher packet loss
                        -> higher congestion

Also injects:
    - diurnal patterns (morning / lunch / evening peaks, night lows)
    - weekly patterns (weekday vs weekend, business vs residential areas)
    - occasional random congestion "events" (short bursts of overload)
    - geographic clustering effects (already encoded in tower_info via
      is_high_density_area -> larger capacity + denser clustering)

Outputs:
    data/synthetic/network_metrics.csv   (raw per-cell time series)
    data/synthetic/traffic_history.csv   (per-tower hourly aggregation)

The row count is controlled by config.yaml -> data_generation.num_records.
This is a TARGET (actual rows = timestamps_per_cell * total_cells), so the
real count will be close to, but not always exactly, num_records.

Everything uses a FIXED random seed for full reproducibility.
NOTE: This is 100% synthetic data generated for academic/demo purposes.
It does not represent any real telecom operator or real subscribers.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.config_loader import get_config, get_path, ensure_dirs
from data_generator.generate_towers import generate_tower_info


def _hour_multiplier(hour: int, peak_cfg: dict) -> float:
    """Diurnal traffic multiplier based on hour-of-day peak windows from config."""
    if hour in peak_cfg["morning"]:
        return 1.55
    if hour in peak_cfg["lunch"]:
        return 1.35
    if hour in peak_cfg["evening"]:
        return 1.75
    if hour in peak_cfg["night_low"]:
        return 0.20
    return 0.85  # off-peak baseline (e.g. mid-morning, afternoon, late evening)


def _weekend_multiplier(is_weekend: bool, is_high_density: bool) -> float:
    """
    Business/high-density (office district) areas quieten down on weekends.
    Other (more residential/mixed) areas stay roughly the same or slightly busier.
    """
    if not is_weekend:
        return 1.0
    return 0.55 if is_high_density else 1.05


def _build_timestamps(simulation_days: int, timestamps_per_cell: int) -> pd.DatetimeIndex:
    end = datetime.now().replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=simulation_days)
    total_seconds = simulation_days * 86400
    freq_seconds = max(60, total_seconds // timestamps_per_cell)  # never below 1 minute
    return pd.date_range(start=start, periods=timestamps_per_cell,
                          freq=pd.Timedelta(seconds=freq_seconds))


def generate_network_metrics(tower_df: pd.DataFrame = None, save: bool = True) -> pd.DataFrame:
    cfg = get_config()
    gen_cfg = cfg["data_generation"]
    seed = cfg["project"]["random_seed"]
    rng = np.random.default_rng(seed)

    if tower_df is None:
        tower_df = generate_tower_info(save=True)

    total_cells = len(tower_df)
    target_records = gen_cfg["num_records"]
    simulation_days = gen_cfg["simulation_days"]
    event_prob = gen_cfg["congestion_event_probability"]
    peak_cfg = gen_cfg["peak_hours"]

    timestamps_per_cell = max(20, target_records // total_cells)
    timestamps = _build_timestamps(simulation_days, timestamps_per_cell)

    print(f"[generate_network_data] {total_cells} cells x {timestamps_per_cell} "
          f"timestamps = ~{total_cells * timestamps_per_cell} rows "
          f"(target was {target_records})")

    all_frames = []

    # Technology-based per-user average throughput (Mbps) - 5G much higher capacity per user
    tech_download_mbps = {"4G": 3.0, "5G": 9.0}
    tech_upload_mbps = {"4G": 1.0, "5G": 3.0}

    for _, tower in tower_df.iterrows():
        n = timestamps_per_cell
        hours = timestamps.hour.values
        dows = timestamps.dayofweek.values  # 0=Mon .. 6=Sun
        is_weekend = dows >= 5

        hour_mult = np.array([_hour_multiplier(h, peak_cfg) for h in hours])
        weekend_mult = np.array([
            _weekend_multiplier(bool(w), bool(tower["is_high_density_area"]))
            for w in is_weekend
        ])

        # small day-to-day random walk noise so the pattern isn't perfectly periodic
        daily_noise = rng.normal(1.0, 0.06, size=n)
        daily_noise = np.clip(daily_noise, 0.7, 1.3)

        capacity = tower["capacity_users"]
        base_load_ratio = rng.uniform(0.35, 0.55)  # baseline average occupancy of this cell

        # ---- Connected users (root cause driver) ----
        connected_users = capacity * base_load_ratio * hour_mult * weekend_mult * daily_noise
        connected_users = np.clip(connected_users, 1, capacity * 1.15)  # can slightly overload

        # ---- Occasional random congestion EVENTS (short overload bursts) ----
        event_mask = rng.random(n) < event_prob
        event_multiplier = np.where(event_mask, rng.uniform(1.4, 2.0, size=n), 1.0)
        connected_users = np.clip(connected_users * event_multiplier, 1, capacity * 1.6)
        connected_users = connected_users.round().astype(int)

        user_load_ratio = connected_users / capacity  # can exceed 1.0 during events

        # ---- Traffic (driven by users) ----
        dl_rate = tech_download_mbps[tower["technology"]]
        ul_rate = tech_upload_mbps[tower["technology"]]
        # per-user usage also fluctuates a bit
        per_user_noise = rng.normal(1.0, 0.15, size=n)
        per_user_noise = np.clip(per_user_noise, 0.6, 1.6)

        download_traffic_mb = connected_users * dl_rate * per_user_noise * rng.uniform(8, 15)
        upload_traffic_mb = connected_users * ul_rate * per_user_noise * rng.uniform(8, 15)

        # ---- Bandwidth utilization (driven by traffic vs capacity) ----
        total_traffic_mbps_equiv = (download_traffic_mb + upload_traffic_mb) / 60.0  # rough Mbps
        theoretical_max_mbps = tower["bandwidth_mhz"] * (6 if tower["technology"] == "5G" else 3)
        utilization_from_traffic = (total_traffic_mbps_equiv / theoretical_max_mbps) * 100
        bandwidth_utilization = 0.6 * utilization_from_traffic + 0.4 * (user_load_ratio * 100)
        bandwidth_utilization = np.clip(bandwidth_utilization + rng.normal(0, 3, size=n), 0, 100)

        # ---- Latency (driven by utilization, nonlinear) ----
        base_latency = rng.uniform(8, 15)  # ms, best-case latency for this cell
        latency_ms = base_latency + 90 * (bandwidth_utilization / 100) ** 2.2
        latency_ms += rng.normal(0, 3, size=n)
        latency_ms = np.clip(latency_ms, 2, 400)

        # ---- Packet loss (driven by utilization + latency) ----
        packet_loss_percent = 0.02 + 12 * (bandwidth_utilization / 100) ** 3
        packet_loss_percent += rng.normal(0, 0.15, size=n)
        packet_loss_percent = np.clip(packet_loss_percent, 0, 25)

        # ---- User-experienced speeds (degrade as utilization rises) ----
        congestion_penalty = 1 - 0.75 * (bandwidth_utilization / 100) ** 1.5
        congestion_penalty = np.clip(congestion_penalty, 0.1, 1.0)
        max_dl_speed = 150 if tower["technology"] == "5G" else 50
        max_ul_speed = 50 if tower["technology"] == "5G" else 15
        download_speed_mbps = max_dl_speed * congestion_penalty * rng.uniform(0.85, 1.0, size=n)
        upload_speed_mbps = max_ul_speed * congestion_penalty * rng.uniform(0.85, 1.0, size=n)

        # ---- Signal strength (mostly independent tower characteristic) ----
        base_signal = rng.uniform(-95, -65)
        signal_strength_dbm = base_signal + rng.normal(0, 4, size=n)
        signal_strength_dbm = np.clip(signal_strength_dbm, -120, -50)

        frame = pd.DataFrame({
            "timestamp": timestamps,
            "tower_id": tower["tower_id"],
            "cell_id": tower["cell_id"],
            "connected_users": connected_users,
            "download_traffic_mb": download_traffic_mb.round(2),
            "upload_traffic_mb": upload_traffic_mb.round(2),
            "bandwidth_utilization": bandwidth_utilization.round(2),
            "latency_ms": latency_ms.round(2),
            "packet_loss_percent": packet_loss_percent.round(3),
            "download_speed_mbps": download_speed_mbps.round(2),
            "upload_speed_mbps": upload_speed_mbps.round(2),
            "signal_strength_dbm": signal_strength_dbm.round(1),
        })
        all_frames.append(frame)

    df = pd.concat(all_frames, ignore_index=True)
    df = df.sort_values(["timestamp", "tower_id", "cell_id"]).reset_index(drop=True)

    # Inject a small amount of realistic "messiness" for the preprocessing stage to clean:
    # a few missing values and a few exact duplicate rows.
    n_missing = max(1, int(len(df) * 0.002))
    missing_idx = rng.choice(df.index, size=n_missing, replace=False)
    missing_col = rng.choice(
        ["latency_ms", "packet_loss_percent", "signal_strength_dbm", "download_traffic_mb"],
        size=n_missing
    )
    for idx, col in zip(missing_idx, missing_col):
        df.loc[idx, col] = np.nan

    n_dupes = max(1, int(len(df) * 0.001))
    dupe_rows = df.sample(n=n_dupes, random_state=seed)
    df = pd.concat([df, dupe_rows], ignore_index=True)

    if save:
        ensure_dirs()
        out_path = get_path("network_metrics_file")
        df.to_csv(out_path, index=False)
        print(f"[generate_network_data] Saved {len(df)} rows -> {out_path}")

    return df


def generate_traffic_history(metrics_df: pd.DataFrame = None, save: bool = True) -> pd.DataFrame:
    """
    Builds traffic_history.csv: hourly aggregation per tower.
    Columns: timestamp, tower_id, total_traffic_gb, peak_users, average_users,
             peak_bandwidth, average_latency
    """
    if metrics_df is None:
        metrics_df = pd.read_csv(get_path("network_metrics_file"), parse_dates=["timestamp"])

    df = metrics_df.copy()
    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour_bucket"] = df["timestamp"].dt.floor("h")
    df["total_traffic_mb"] = df["download_traffic_mb"].fillna(0) + df["upload_traffic_mb"].fillna(0)

    agg = df.groupby(["hour_bucket", "tower_id"]).agg(
        total_traffic_gb=("total_traffic_mb", lambda x: round(x.sum() / 1024, 3)),
        peak_users=("connected_users", "max"),
        average_users=("connected_users", "mean"),
        peak_bandwidth=("bandwidth_utilization", "max"),
        average_latency=("latency_ms", "mean"),
    ).reset_index()

    agg = agg.rename(columns={"hour_bucket": "timestamp"})
    agg["average_users"] = agg["average_users"].round(1)
    agg["peak_bandwidth"] = agg["peak_bandwidth"].round(2)
    agg["average_latency"] = agg["average_latency"].round(2)
    agg = agg.sort_values(["timestamp", "tower_id"]).reset_index(drop=True)

    if save:
        out_path = get_path("traffic_history_file")
        agg.to_csv(out_path, index=False)
        print(f"[generate_network_data] Saved {len(agg)} rows -> {out_path}")

    return agg


def generate_all(save: bool = True):
    """Runs the full synthetic data generation pipeline: towers -> metrics -> traffic history."""
    print("=" * 70)
    print("SYNTHETIC/SIMULATED TELECOM DATA GENERATION")
    print("(This is fabricated demo data - not real network data.)")
    print("=" * 70)
    tower_df = generate_tower_info(save=save)
    metrics_df = generate_network_metrics(tower_df=tower_df, save=save)
    traffic_df = generate_traffic_history(metrics_df=metrics_df, save=save)
    print("=" * 70)
    print("DONE.")
    print(f"  tower_info.csv       : {len(tower_df)} rows")
    print(f"  network_metrics.csv  : {len(metrics_df)} rows")
    print(f"  traffic_history.csv  : {len(traffic_df)} rows")
    print("=" * 70)
    return tower_df, metrics_df, traffic_df


if __name__ == "__main__":
    generate_all(save=True)

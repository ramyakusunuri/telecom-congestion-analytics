"""
generate_towers.py
-------------------
Generates SYNTHETIC/SIMULATED telecom tower infrastructure data.

Output: data/synthetic/tower_info.csv
Columns:
    tower_id, tower_name, latitude, longitude, city, area, zone,
    tower_type, technology, capacity_users, bandwidth_mhz, installation_date

Design notes:
- Towers are grouped into geographic "areas" per city, and each area is
  built around a random anchor lat/lon with towers scattered tightly
  around it (small gaussian jitter). This creates real geographic
  CLUSTERS so DBSCAN hotspot detection later has genuine structure to find,
  instead of uniformly random points.
- A subset of areas are deliberately marked as "high traffic" zones
  (dense business districts / transit hubs) by giving them smaller
  jitter (denser clustering) and higher capacity towers. This is what
  produces natural congestion hotspots downstream.
- Uses a FIXED random seed (from config.yaml) for full reproducibility.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.config_loader import get_config, get_path, ensure_dirs

# Approximate city center coordinates (India) used as anchors for area clusters
CITY_CENTERS = {
    "Hyderabad": (17.3850, 78.4867),
    "Bengaluru": (12.9716, 77.5946),
    "Mumbai": (19.0760, 72.8777),
    "Pune": (18.5204, 73.8567),
    "Chennai": (13.0827, 80.2707),
}


def _pick_weighted(rng, options: dict):
    """options: {value: probability}. Returns a single sampled value."""
    keys = list(options.keys())
    probs = np.array(list(options.values()), dtype=float)
    probs = probs / probs.sum()
    return rng.choice(keys, p=probs)


def generate_tower_info(save: bool = True) -> pd.DataFrame:
    cfg = get_config()
    gen_cfg = cfg["data_generation"]
    seed = cfg["project"]["random_seed"]
    rng = np.random.default_rng(seed)

    num_towers = gen_cfg["num_towers"]
    cells_per_tower = gen_cfg["cells_per_tower"]
    cities = gen_cfg["cities"]
    zones = gen_cfg["zones"]
    tower_types = gen_cfg["tower_types"]
    tech_mix = gen_cfg["technology_mix"]

    # Build a fixed set of "areas" per city (business district style clusters).
    # Roughly 3-5 areas per city, each with its own anchor point.
    areas = []
    for city in cities:
        base_lat, base_lon = CITY_CENTERS.get(city, (20.5937, 78.9629))
        n_areas = rng.integers(3, 6)
        for i in range(n_areas):
            # scatter area anchors within ~15km of the city center
            anchor_lat = base_lat + rng.normal(0, 0.07)
            anchor_lon = base_lon + rng.normal(0, 0.07)
            # ~25% of areas are "high density" (business/transit hubs -> future hotspots)
            is_high_density = rng.random() < 0.25
            areas.append({
                "city": city,
                "area": f"{city[:3].upper()}-Area{i+1}",
                "zone": rng.choice(zones),
                "anchor_lat": anchor_lat,
                "anchor_lon": anchor_lon,
                "is_high_density": is_high_density,
            })
    areas_df = pd.DataFrame(areas)

    records = []
    installation_start = datetime(2019, 1, 1)
    installation_end = datetime(2023, 12, 31)
    install_range_days = (installation_end - installation_start).days

    for t in range(1, num_towers + 1):
        tower_id = f"TWR{t:04d}"
        area_row = areas_df.iloc[rng.integers(0, len(areas_df))]

        # Denser jitter (tighter cluster / more towers packed together) in
        # high-density areas -> realistic congestion hotspots.
        jitter_scale = 0.004 if area_row["is_high_density"] else 0.015
        lat = area_row["anchor_lat"] + rng.normal(0, jitter_scale)
        lon = area_row["anchor_lon"] + rng.normal(0, jitter_scale)

        technology = _pick_weighted(rng, tech_mix)
        tower_type = rng.choice(tower_types)

        # High density areas get bigger macro towers with more capacity/bandwidth
        if area_row["is_high_density"]:
            base_capacity = rng.integers(800, 1500)
            bandwidth_mhz = rng.choice([40, 60, 80, 100])
        else:
            base_capacity = rng.integers(200, 800)
            bandwidth_mhz = rng.choice([10, 20, 40])

        # 5G towers generally get more bandwidth allocation
        if technology == "5G":
            bandwidth_mhz = int(bandwidth_mhz * rng.uniform(1.2, 1.8))

        install_offset = rng.integers(0, install_range_days)
        install_date = installation_start + timedelta(days=int(install_offset))

        tower_name = f"{area_row['area']}-Site{t:04d}"

        for c in range(1, cells_per_tower + 1):
            cell_id = f"{tower_id}-C{c}"
            records.append({
                "tower_id": tower_id,
                "cell_id": cell_id,
                "tower_name": tower_name,
                "latitude": round(float(lat), 6),
                "longitude": round(float(lon), 6),
                "city": area_row["city"],
                "area": area_row["area"],
                "zone": area_row["zone"],
                "tower_type": tower_type,
                "technology": technology,
                "capacity_users": int(base_capacity // cells_per_tower),
                "bandwidth_mhz": int(bandwidth_mhz),
                "installation_date": install_date.strftime("%Y-%m-%d"),
                "is_high_density_area": bool(area_row["is_high_density"]),
            })

    df = pd.DataFrame(records)

    if save:
        ensure_dirs()
        out_path = get_path("tower_info_file")
        df.to_csv(out_path, index=False)
        print(f"[generate_towers] Saved {len(df)} cell/tower records "
              f"({num_towers} towers x {cells_per_tower} cells) -> {out_path}")

    return df


if __name__ == "__main__":
    print("Generating SYNTHETIC tower & cell infrastructure data...")
    df = generate_tower_info()
    print(df.head(10).to_string(index=False))
    print(f"\nTotal towers: {df['tower_id'].nunique()}")
    print(f"Total cells:  {len(df)}")
    print(f"Cities: {df['city'].unique().tolist()}")
    print(f"Technology split:\n{df['technology'].value_counts()}")

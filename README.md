# Telecom Network Congestion Prediction & Hotspot Analytics

> **Status: Batch 1 of 6 implemented.** This README is a working stub and will
> be replaced with the full project README (installation, all pipeline
> stages, troubleshooting, etc.) in the final batch. See `PROGRESS.md` for
> what's done so far.

## What's implemented in this batch

- Full project folder structure (see tree below)
- `config/config.yaml` + `config/config_loader.py` — single source of truth
  for every parameter used across the whole project (paths, seeds, model
  hyperparameters, thresholds, DB settings)
- `requirements.txt`
- `data_generator/generate_towers.py` — generates 100 towers (300 cells,
  3 sectors/tower) across 5 cities with realistic geographic clustering
  (some areas deliberately dense -> future hotspots)
- `data_generator/generate_network_data.py` — generates the full
  synthetic time-series (`network_metrics.csv`, `traffic_history.csv`)
  with a real causal chain: users -> traffic -> utilization -> latency ->
  packet loss, diurnal + weekly patterns, random congestion events, and
  intentionally injected missing values/duplicates for the preprocessing
  stage (Batch 2) to clean.

All data is **synthetic / simulated** — generated for this academic
project, not real subscriber or operator data.

## How to generate the data right now

```bash
# from the project root
pip install -r requirements.txt

# generate tower_info.csv, network_metrics.csv, traffic_history.csv
python data_generator/generate_network_data.py
```

Output lands in `data/synthetic/`:

| File | Rows (default config) | Description |
|---|---|---|
| `tower_info.csv` | 300 (100 towers x 3 cells) | Tower/cell metadata + geo coordinates |
| `network_metrics.csv` | ~150,000 | Per-cell time series network KPIs |
| `traffic_history.csv` | ~50,000 | Hourly per-tower traffic aggregation |

To scale up toward the 500k-1M record target for a bigger Big-Data demo,
edit `config/config.yaml`:

```yaml
data_generation:
  num_records: 1000000     # was 150000
  simulation_days: 90
```

then re-run the same command. Everything downstream (Spark, ML, dashboard)
reads its row counts from this file, so no other code changes are needed.

## Project structure

```text
telecom-congestion-analytics/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/            <- generator output lands here
│
├── data_generator/
│   ├── generate_towers.py          [DONE - Batch 1]
│   └── generate_network_data.py    [DONE - Batch 1]
│
├── preprocessing/
│   ├── clean_data.py                [Batch 2]
│   └── feature_engineering.py       [Batch 2]
│
├── spark/
│   ├── batch_processing.py          [Batch 2]
│   ├── spark_sql.py                 [Batch 2]
│   └── streaming.py                 [Batch 6]
│
├── ml/
│   ├── train.py                     [Batch 3]
│   ├── evaluate.py                  [Batch 3]
│   ├── predict.py                   [Batch 3]
│   └── models/                      [Batch 3 - saved artifacts]
│
├── hotspot/
│   ├── clustering.py                [Batch 4]
│   └── hotspot_score.py             [Batch 4]
│
├── alerts/
│   └── alert_engine.py              [Batch 4]
│
├── database/
│   ├── schema.sql                   [Batch 5]
│   └── database.py                  [Batch 5]
│
├── kafka/
│   ├── producer.py                  [Batch 6]
│   └── consumer.py                  [Batch 6]
│
├── hadoop/
│   └── hdfs_commands.md             [Batch 6]
│
├── dashboard/
│   ├── app.py                       [Batch 5]
│   └── pages/                       [Batch 5]
│
├── notebooks/
│   └── eda.ipynb                    [Batch 6]
│
├── tests/                           [Batch 6]
│
├── config/
│   ├── config.yaml           [DONE - Batch 1]
│   └── config_loader.py      [DONE - Batch 1]
│
├── .env.example               [DONE - Batch 1]
├── requirements.txt           [DONE - Batch 1]
├── README.md                  [this file]
└── run_project.py                   [Batch 6 - final orchestrator]
```

## Next batch (Batch 2)

PySpark preprocessing, feature engineering (rolling windows, time
features), congestion scoring engine, Spark SQL analysis queries.

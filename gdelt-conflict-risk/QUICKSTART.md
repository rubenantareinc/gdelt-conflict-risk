# Quickstart

This project runs fully offline using the included toy dataset.

## 1) Set up a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Run the ablation pipeline (time-aware split)

```bash
python src/train_eval.py --train_end_month 2024-02
```

**Expected outputs** (created under `outputs/`):
- `ablation_time_split_metrics.csv`
- `ablation_time_split_metrics.json`

## 3) Run the leakage demonstration

```bash
python src/leakage_demo.py --train_end_month 2024-02
```

**Expected outputs** (created under `outputs/`):
- `leakage_demo_metrics.csv`

## 4) (Optional) Download a GDELT Doc API sample

```bash
python src/download_gdelt.py --query "Tripoli militia" --start 20240101 --end 20240331 --out data/gdelt_articles.csv
```

This helper produces a CSV with a coarse `city` field and a `location_raw` column. For real city-level experiments, replace `city` using a proper geocoding step or GDELT GKG/location fields.

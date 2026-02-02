# GDELT Conflict Risk — Toy, Reproducible Pipeline with Leakage Demo

This repository is a compact, **offline-runnable** research scaffold for city-month conflict risk modeling using GDELT-style news text. It emphasizes **reproducibility**, **time-aware evaluation**, and a concrete **data leakage discovery**: the failure mode where improper aggregation leaks future-month text into earlier samples.

The project ships a **toy dataset** so every command runs without external downloads. You can swap in real GDELT-derived exports when ready (see [Data](#data)).

## Why this repo (narrative alignment)

- **Data leakage discovery:** `src/leakage_demo.py` demonstrates how city-level aggregation can leak future text into past labels when copied across months, and shows the correct city-month aggregation instead.
- **Time-aware splits:** both the main pipeline and leakage demo use month-based splits to avoid look-ahead bias.
- **Feature ablations:** the training script runs text-only, numeric-only, and combined feature sets to compare contribution.
- **Reproducible and offline:** a small toy dataset and deterministic outputs keep experiments runnable and reviewable.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Ablations with a time-aware split
python src/train_eval.py --train_end_month 2024-02

# Demonstrate the leakage failure mode vs the correct aggregation
python src/leakage_demo.py --train_end_month 2024-02
```

Outputs are written to `outputs/`:
- `outputs/ablation_time_split_metrics.csv`
- `outputs/ablation_time_split_metrics.json`
- `outputs/leakage_demo_metrics.csv`

## Data

Toy data (already included):
- `data/toy_articles.csv` — article-level text with dates and locations
- `data/toy_city_month_labels.csv` — city-month labels

To use real GDELT-derived data, replace `data/toy_articles.csv` with a CSV that matches the expected schema (columns: `date, city, text, numeric_refugees_m, numeric_gdp_growth`). The helper `src/download_gdelt.py` can bootstrap a CSV from the GDELT Doc API, but it **does not resolve true city names**; it records a coarse `location_raw` and uses source country as a placeholder `city` field. For city-level modeling, insert a proper geocoding step.

## Entry points

- `src/train_eval.py` — end-to-end pipeline: city-month documents → TF-IDF + numeric cues → ablations → time-aware evaluation
- `src/leakage_demo.py` — minimal demonstration of the leakage pattern and why time-aware splits matter
- `src/download_gdelt.py` — optional helper for downloading GDELT Doc API results (requires geocoding for real city-level work)

## Documentation

- `docs/LEAKAGE_ANALYSIS.md` — explains the leakage mechanism and the correct aggregation approach
- `docs/ABLATION_STUDY.md` — documents implemented ablations and how to read metrics
- `QUICKSTART.md` — step-by-step setup and expected outputs
- `CONTRIBUTING.md` — lightweight contribution guide

## Scope and non-claims

- ✅ Implements city-month document construction, TF-IDF features, ablations, and time-aware evaluation.
- ✅ Demonstrates a real leakage failure mode caused by incorrect aggregation logic.
- ❌ Does **not** ship large-scale GDELT datasets.
- ❌ Does **not** claim production forecasting performance or external validity.

## License

MIT. See [LICENSE](LICENSE).

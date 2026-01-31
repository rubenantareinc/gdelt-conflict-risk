# City-Level Conflict Risk from News Text Signals (GDELT-style)

This repository implements an end-to-end **city-month text** pipeline for conflict-risk modeling:
1) collect articles for a place and time window (e.g., via **GDELT**),
2) build **city-month documents**,
3) extract **TF-IDF text features** plus simple **numeric cues**,
4) run **ablations** (text-only vs numeric-only vs combined),
5) evaluate with **time-aware splits**.

Because application review and reproducibility matter, this repo also includes a **leakage demonstration**:
a small script that shows how an *improper aggregation* (city-level docs copied into each month) can leak future text into earlier examples.

> Note: This repo includes a **toy dataset** so everything runs offline. Swap in real GDELT-derived data for real experiments.

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

Outputs are written to `outputs/`.

## Data

- `data/toy_articles.csv` — article-level text with dates and locations (toy)
- `data/toy_city_month_labels.csv` — city-month target labels (toy)

For real data, replace `toy_articles.csv` with a file exported from your own GDELT query (same schema: `date, city, text, ...`).

## What to look at

- `src/train_eval.py` — full pipeline + ablation + time-aware evaluation
- `src/leakage_demo.py` — demonstrates the leakage bug pattern and why time splits matter

## What this repo claims (and what it doesn't)

- ✅ Implements city-month document construction + TF-IDF features + ablations + time-aware split evaluation.
- ✅ Demonstrates a common leakage pattern caused by improper aggregation logic.
- ❌ Does not ship large-scale GDELT datasets inside the repo (download/export them separately).
- ❌ Does not claim production forecasting performance or universal causal conclusions.

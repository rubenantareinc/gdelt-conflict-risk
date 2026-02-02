# Leakage Analysis

This repo includes a minimal, reproducible leakage demonstration in `src/leakage_demo.py`. The goal is to show how a **wrong aggregation** can inject future-month text into earlier examples, making evaluation look better than it should.

## The leakage pattern

### Wrong aggregation (leaky)

The leaky path aggregates all text **at the city level**, ignoring time, and then copies that same city-level document into every month for that city. This is implemented in:

- `leaky_docs()` in `src/leakage_demo.py`

This creates a feature matrix where documents for early months contain **future text**, which is information the model should not have at training time.

### Correct aggregation (time-aware)

The correct path aggregates articles **by city and month** so each month only contains text available in that month. This is implemented in:

- `correct_docs()` in `src/leakage_demo.py`

With this approach, each city-month example is based on contemporaneous text only.

## Why time-aware splits matter

Even with correct aggregation, using random splits can still leak information across time. The leakage demo uses a **time-based split** so that training data only includes months up to `--train_end_month`, and testing uses later months:

- `time_split()` in `src/leakage_demo.py`

This ensures that both **features and labels** respect temporal ordering.

## How to run

```bash
python src/leakage_demo.py --train_end_month 2024-02
```

Outputs:
- `outputs/leakage_demo_metrics.csv`

The file compares the evaluation metrics under the correct aggregation vs. the leaky aggregation so you can see the inflation effect.

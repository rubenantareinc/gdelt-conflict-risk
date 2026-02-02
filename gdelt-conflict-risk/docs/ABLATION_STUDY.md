# Ablation Study

The ablation study is implemented in `src/train_eval.py`. It compares three feature configurations under a **time-aware split**:

1. **Text-only**: TF-IDF on city-month documents.
2. **Numeric-only**: simple numeric cues (toy example includes `numeric_refugees_m`, `numeric_gdp_growth`, and `n_articles`).
3. **Combined**: concatenated text + numeric features.

## Feature sets (high level)

- **TF-IDF text**: built from city-month documents via `build_city_month_docs()` and `TfidfVectorizer`.
- **Numeric cues**: mean values per city-month from the toy dataset plus `n_articles` (article count).

## Reproducing results

Run the pipeline:

```bash
python src/train_eval.py --train_end_month 2024-02
```

Outputs are written to `outputs/`:
- `ablation_time_split_metrics.csv` — tabular metrics for each ablation.
- `ablation_time_split_metrics.json` — JSON summary of the same metrics.

These files are designed to be small and deterministic so reviewers can inspect them without external data.

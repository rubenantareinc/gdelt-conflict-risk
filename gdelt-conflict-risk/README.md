# SpaceX Starship Incident Dataset & Pipeline (Offline, Reproducible)

This repository provides a **fully offline**, reproducible pipeline for labeling and evaluating Starship incident narratives. It ships a **small, checked-in dataset** of incident narratives (25 incidents) so reviewers can run everything without network access, then scale using their own additional sources.

## Project goals

- Build a labeled incident dataset with transparent provenance.
- Make labeling consistent and reproducible.
- Provide evaluation that scales as the dataset grows.

## Dataset snapshot (checked in)

- **Incidents:** 25 narratives in `data/processed/incidents.jsonl`.
- **Labels:** 25 labeled incidents in `data/labels.csv`.
- **Raw narratives:** `data/raw_text/{source_id}.txt`.
- **Sources registry:** `data/sources.csv` (with explicit URL placeholders when missing).

> ⚠️ The current sources include **explicit missing URLs** and short provenance notes. Replace them with verified URLs and summaries when you expand the dataset.

## Data statement

- **Sources:** Public narratives (official updates, webcasts, or press coverage). The repo ships **cleaned incident narratives**, not full webpages.
- **Scope:** A small subset is included for offline reproducibility; the pipeline supports scaling by adding new sources and raw text.
- **Provenance:** Each incident retains a `source_id`, `url` (or `MISSING`), `retrieved_date` (optional), and a short `source_summary`.

## Reproduce in 3 commands

```bash
python src/ingest/build_dataset.py
python src/baselines/empty_baseline.py
python src/evaluation/evaluate_labels.py
```

Outputs:
- `outputs/metrics.json`
- `outputs/metrics.md`
- `outputs/predictions.csv`

## Labeling workflow

- Run the interactive label tool:

```bash
python src/labeling/label_tool.py --resume
```

- Label definitions and consistency rules: `docs/LABEL_GUIDE.md`.

## Scaling the dataset (no scraping in this environment)

1. Add source metadata to `data/sources.csv`.
2. Add narrative text to `data/raw_text/{source_id}.txt` or provide a CSV with columns `source_id,incident_name,date,text`.
3. Rebuild the dataset:

```bash
python src/ingest/build_dataset.py --raw-dir data/raw_text --sources-csv data/sources.csv
```

## Incident cards

The smoke test generates a sample incident card at:

- `outputs/incident_cards/incident_spx-001.md`

## Repository layout

```
data/
  raw_text/                # Narrative text (one file per source_id)
  processed/incidents.jsonl
  labels.csv
  sources.csv
  schema.yaml
  schema.json
src/
  ingest/build_dataset.py
  labeling/label_tool.py
  baselines/empty_baseline.py
  evaluation/evaluate_labels.py
scripts/
  smoke_end_to_end.py
```

## Notes on honesty and reproducibility

- No web scraping is performed in this environment.
- URLs are left blank or explicitly marked as `MISSING` when unknown.
- Labels are assigned only from text evidence and can be refined when sources are expanded.

## License

MIT. See [LICENSE](LICENSE).

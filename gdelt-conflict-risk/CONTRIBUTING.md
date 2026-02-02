# Contributing

Thanks for helping improve this research scaffold. Please keep changes small, reproducible, and aligned with the leakage-focused narrative.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running scripts

```bash
python src/train_eval.py --train_end_month 2024-02
python src/leakage_demo.py --train_end_month 2024-02
```

## Experiments

- Keep new experiments in `src/` and document them in `docs/`.
- Prefer deterministic outputs written to `outputs/` and avoid large files.

## Issues and PRs

- File an issue with a clear repro or question.
- For PRs, include a concise summary and how to reproduce results locally.

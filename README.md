# GDELT Conflict Risk

This repository centers on city-month conflict risk modeling using GDELT-style news and event signals. It provides an end-to-end pipeline to build city-month documents, run time-aware splits, and demonstrate leakage pitfalls for reproducible evaluation. See the main project README for full details.

## Repository structure

- `/gdelt-conflict-risk` — main project (pipeline + time-aware splits + leakage demo + outputs)
- `/legacy/middle-east-conflict-risk` — older/secondary analysis kept for reference

## Quick start

```bash
cd gdelt-conflict-risk
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Ablations with a time-aware split
python src/train_eval.py --train_end_month 2024-02

# Demonstrate the leakage failure mode vs the correct aggregation
python src/leakage_demo.py --train_end_month 2024-02
```

For more background and context, see [`gdelt-conflict-risk/README.md`](gdelt-conflict-risk/README.md).

#!/usr/bin/env python3
"""Baseline that predicts empty labels for all fields."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

FIELDS = ["subsystem", "failure_mode", "impact", "cause"]


def load_incidents(path: Path) -> Dict[str, Dict[str, str]]:
    incidents = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            incidents[row["incident_id"]] = {
                "date": row.get("date", ""),
            }
    return incidents


def time_split(incidents: Dict[str, Dict[str, str]], train_end_date: str) -> List[str]:
    test_ids = []
    for incident_id, row in incidents.items():
        date = row.get("date", "")
        if not date:
            continue
        if date > train_end_date:
            test_ids.append(incident_id)
    return test_ids


def random_split(incident_ids: List[str], test_size: float, seed: int) -> List[str]:
    rng = seed
    ids = sorted(incident_ids)
    shuffled = []
    for incident_id in ids:
        rng = (1103515245 * rng + 12345) & 0x7FFFFFFF
        shuffled.append((rng, incident_id))
    shuffled.sort()
    cut = int(len(shuffled) * (1 - test_size))
    return [incident_id for _, incident_id in shuffled[cut:]]


def main() -> None:
    ap = argparse.ArgumentParser(description="Empty baseline predictions")
    ap.add_argument("--incidents", default="data/processed/incidents.jsonl", help="Incidents JSONL")
    ap.add_argument("--split", choices=["random", "time"], default="random", help="Split strategy")
    ap.add_argument("--train-end-date", default="", help="Train end date for time split")
    ap.add_argument("--test-size", type=float, default=0.2, help="Test size fraction for random split")
    ap.add_argument("--seed", type=int, default=13, help="Random seed")
    ap.add_argument("--output", default="outputs/predictions.csv", help="Output predictions CSV")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    incidents = load_incidents(root / args.incidents)

    if args.split == "time":
        if not args.train_end_date:
            raise SystemExit("--train-end-date is required for time split")
        test_ids = time_split(incidents, args.train_end_date)
    else:
        test_ids = random_split(list(incidents.keys()), args.test_size, args.seed)

    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["incident_id", *FIELDS])
        for incident_id in test_ids:
            writer.writerow([incident_id, "", "", "", ""])

    print(f"Wrote {len(test_ids)} predictions to {output_path}")


if __name__ == "__main__":
    main()

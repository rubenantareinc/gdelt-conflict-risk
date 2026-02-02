#!/usr/bin/env python3
"""Evaluate multi-label predictions for Starship incidents."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


FIELDS = ["subsystem", "failure_mode", "impact", "cause"]


@dataclass
class SplitResult:
    train_ids: List[str]
    test_ids: List[str]


def load_incidents(path: Path) -> Dict[str, Dict[str, str]]:
    incidents = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            incidents[row["incident_id"]] = {
                "date": row.get("date", ""),
            }
    return incidents


def load_labels_csv(path: Path) -> Dict[str, Dict[str, Set[str]]]:
    labels: Dict[str, Dict[str, Set[str]]] = {}
    if not path.exists():
        return labels
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            incident_id = row.get("incident_id", "").strip()
            if not incident_id:
                continue
            labels[incident_id] = {
                field: parse_set(row.get(field, "")) for field in FIELDS
            }
    return labels


def parse_set(value: str) -> Set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(";") if item.strip()}


def time_split(incidents: Dict[str, Dict[str, str]], train_end_date: str) -> SplitResult:
    train_ids, test_ids = [], []
    for incident_id, row in incidents.items():
        date = row.get("date", "")
        if not date:
            continue
        if date <= train_end_date:
            train_ids.append(incident_id)
        else:
            test_ids.append(incident_id)
    return SplitResult(train_ids=train_ids, test_ids=test_ids)


def random_split(incident_ids: List[str], test_size: float, seed: int) -> SplitResult:
    rng = seed
    ids = sorted(incident_ids)
    shuffled = []
    for incident_id in ids:
        rng = (1103515245 * rng + 12345) & 0x7FFFFFFF
        shuffled.append((rng, incident_id))
    shuffled.sort()
    cut = int(len(shuffled) * (1 - test_size))
    train_ids = [incident_id for _, incident_id in shuffled[:cut]]
    test_ids = [incident_id for _, incident_id in shuffled[cut:]]
    return SplitResult(train_ids=train_ids, test_ids=test_ids)


def compute_metrics(
    gold: Dict[str, Dict[str, Set[str]]],
    pred: Dict[str, Dict[str, Set[str]]],
    test_ids: Iterable[str],
) -> Dict[str, Dict[str, object]]:
    metrics: Dict[str, Dict[str, object]] = {}
    for field in FIELDS:
        label_set = set()
        for incident_id in test_ids:
            label_set.update(gold.get(incident_id, {}).get(field, set()))
            label_set.update(pred.get(incident_id, {}).get(field, set()))

        per_label = {}
        total_tp = total_fp = total_fn = 0
        support_counts = Counter()

        for label in sorted(label_set):
            tp = fp = fn = 0
            for incident_id in test_ids:
                gold_labels = gold.get(incident_id, {}).get(field, set())
                pred_labels = pred.get(incident_id, {}).get(field, set())
                if label in gold_labels:
                    support_counts[label] += 1
                if label in gold_labels and label in pred_labels:
                    tp += 1
                elif label not in gold_labels and label in pred_labels:
                    fp += 1
                elif label in gold_labels and label not in pred_labels:
                    fn += 1
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            per_label[label] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": support_counts[label],
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
            total_tp += tp
            total_fp += fp
            total_fn += fn

        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
        micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)) if (micro_precision + micro_recall) else 0.0

        macro_f1 = 0.0
        if per_label:
            macro_f1 = sum(label_metrics["f1"] for label_metrics in per_label.values()) / len(per_label)

        top_labels = [label for label, _ in support_counts.most_common(5)]
        confusion_summary = {
            label: {
                "tp": per_label[label]["tp"],
                "fp": per_label[label]["fp"],
                "fn": per_label[label]["fn"],
            }
            for label in top_labels
        }

        metrics[field] = {
            "micro_f1": round(micro_f1, 4),
            "macro_f1": round(macro_f1, 4),
            "per_label": per_label,
            "support_counts": dict(support_counts),
            "confusion_summary": confusion_summary,
        }
    return metrics


def metrics_to_markdown(metrics: Dict[str, Dict[str, object]]) -> str:
    lines = ["# Evaluation Metrics", ""]
    for field, field_metrics in metrics.items():
        lines.append(f"## {field}")
        lines.append("")
        lines.append(f"- Micro F1: {field_metrics['micro_f1']}")
        lines.append(f"- Macro F1: {field_metrics['macro_f1']}")
        lines.append("")
        lines.append("### Support Counts")
        for label, count in sorted(field_metrics["support_counts"].items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"- {label}: {count}")
        lines.append("")
        lines.append("### Confusion Summary (Top Labels)")
        for label, summary in field_metrics["confusion_summary"].items():
            lines.append(f"- {label}: tp={summary['tp']}, fp={summary['fp']}, fn={summary['fn']}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate multi-label predictions")
    ap.add_argument("--incidents", default="data/processed/incidents.jsonl", help="Incidents JSONL")
    ap.add_argument("--gold", default="data/labels.csv", help="Gold labels CSV")
    ap.add_argument("--pred", default="outputs/predictions.csv", help="Predicted labels CSV")
    ap.add_argument("--split", choices=["random", "time"], default="random", help="Split strategy")
    ap.add_argument("--train-end-date", default="", help="Train end date for time split (YYYY-MM-DD)")
    ap.add_argument("--test-size", type=float, default=0.2, help="Test size fraction for random split")
    ap.add_argument("--seed", type=int, default=13, help="Random seed")
    ap.add_argument("--outdir", default="outputs", help="Output directory")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    incidents = load_incidents(root / args.incidents)
    gold = load_labels_csv(root / args.gold)
    pred = load_labels_csv(root / args.pred)

    if args.split == "time":
        if not args.train_end_date:
            raise SystemExit("--train-end-date is required for time split")
        split = time_split(incidents, args.train_end_date)
        test_ids = split.test_ids
    else:
        split = random_split(list(incidents.keys()), args.test_size, args.seed)
        test_ids = split.test_ids

    metrics = compute_metrics(gold, pred, test_ids)

    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (outdir / "metrics.md").write_text(metrics_to_markdown(metrics), encoding="utf-8")

    print(f"Wrote metrics to {outdir / 'metrics.json'} and {outdir / 'metrics.md'}")


if __name__ == "__main__":
    main()

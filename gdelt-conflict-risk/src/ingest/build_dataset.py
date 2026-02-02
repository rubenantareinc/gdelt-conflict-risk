#!/usr/bin/env python3
"""Build incidents.jsonl from raw text files or a CSV of narratives."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class SourceRecord:
    source_id: str
    url: str
    title: str
    incident_name: str
    notes: str


def load_sources(path: Path) -> Dict[str, SourceRecord]:
    sources: Dict[str, SourceRecord] = {}
    if not path.exists():
        return sources
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source_id = row.get("source_id", "").strip()
            if not source_id:
                continue
            sources[source_id] = SourceRecord(
                source_id=source_id,
                url=row.get("url", "").strip() or "MISSING",
                title=row.get("title", "").strip(),
                incident_name=row.get("incident_name", "").strip(),
                notes=row.get("notes", "").strip(),
            )
    return sources


def load_labels(path: Path) -> Dict[str, Dict[str, List[str]]]:
    labels: Dict[str, Dict[str, List[str]]] = {}
    if not path.exists():
        return labels
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            incident_id = row.get("incident_id", "").strip()
            if not incident_id:
                continue
            labels[incident_id] = {
                "subsystem": split_labels(row.get("subsystem", "")),
                "failure_mode": split_labels(row.get("failure_mode", "")),
                "impact": split_labels(row.get("impact", "")),
                "cause": split_labels(row.get("cause", "")),
            }
    return labels


def split_labels(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(";") if item.strip()]


def iter_raw_text(raw_dir: Path) -> Iterable[Dict[str, str]]:
    if not raw_dir.exists():
        return []
    for path in sorted(raw_dir.glob("*.txt")):
        source_id = path.stem
        text = path.read_text(encoding="utf-8").strip()
        yield {
            "source_id": source_id,
            "text": text,
        }


def iter_csv_narratives(path: Path) -> Iterable[Dict[str, str]]:
    if not path or not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source_id = row.get("source_id", "").strip()
            if not source_id:
                continue
            yield {
                "source_id": source_id,
                "text": row.get("text", "").strip(),
                "incident_name": row.get("incident_name", "").strip(),
                "date": row.get("date", "").strip(),
            }


def build_incidents(
    sources: Dict[str, SourceRecord],
    raw_rows: Iterable[Dict[str, str]],
    csv_rows: Iterable[Dict[str, str]],
    labels: Dict[str, Dict[str, List[str]]],
) -> List[Dict[str, object]]:
    incidents = []
    seen = set()
    combined = list(raw_rows) + list(csv_rows)
    for idx, row in enumerate(combined, start=1):
        source_id = row.get("source_id")
        if not source_id or source_id in seen:
            continue
        seen.add(source_id)
        source = sources.get(source_id)
        incident_name = row.get("incident_name") or (source.incident_name if source else f"Incident {idx:03d}")
        incident_id = f"spx-{idx:03d}"
        incident_labels = labels.get(incident_id, {"subsystem": [], "failure_mode": [], "impact": [], "cause": []})
        incidents.append(
            {
                "incident_id": incident_id,
                "incident_name": incident_name,
                "date": row.get("date") or "",
                "text": row.get("text", ""),
                "sources": [
                    {
                        "source_id": source_id,
                        "url": source.url if source else "MISSING",
                        "title": source.title if source else "",
                        "retrieved_date": "",
                        "source_summary": source.notes if source else "",
                    }
                ],
                "labels": incident_labels,
                "notes": "",
            }
        )
    return incidents


def write_jsonl(path: Path, incidents: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for incident in incidents:
            f.write(json.dumps(incident, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build incidents.jsonl from raw text and/or narrative CSV")
    ap.add_argument("--raw-dir", default="data/raw_text", help="Directory of raw text files named {source_id}.txt")
    ap.add_argument("--narratives-csv", default="", help="Optional CSV with columns: source_id,incident_name,date,text")
    ap.add_argument("--sources-csv", default="data/sources.csv", help="CSV of sources with provenance metadata")
    ap.add_argument("--labels-csv", default="data/labels.csv", help="CSV of labels keyed by incident_id")
    ap.add_argument("--output", default="data/processed/incidents.jsonl", help="Output JSONL path")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    sources = load_sources(root / args.sources_csv)
    labels = load_labels(root / args.labels_csv)
    raw_rows = list(iter_raw_text(root / args.raw_dir))
    csv_rows = list(iter_csv_narratives(root / args.narratives_csv)) if args.narratives_csv else []

    if not raw_rows and not csv_rows:
        raise SystemExit("No raw text or narrative CSV rows found.")

    incidents = build_incidents(sources, raw_rows, csv_rows, labels)
    write_jsonl(root / args.output, incidents)
    print(f"Wrote {len(incidents)} incidents to {args.output}")


if __name__ == "__main__":
    main()

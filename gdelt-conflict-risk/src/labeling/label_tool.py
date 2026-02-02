#!/usr/bin/env python3
"""Interactive labeling tool for Starship incidents."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

LABEL_OPTIONS = {
    "subsystem": [
        "propulsion",
        "structures",
        "avionics",
        "guidance_navigation_control",
        "thermal_protection",
        "ground_support",
        "flight_termination",
        "software",
        "communications",
        "power",
        "recovery",
        "operations",
        "unspecified",
    ],
    "failure_mode": [
        "engine_failure",
        "pressurization_failure",
        "structural_failure",
        "fire",
        "explosion",
        "loss_of_control",
        "separation_failure",
        "landing_failure",
        "ignition_abort",
        "telemetry_loss",
        "pad_damage",
        "debris_impact",
        "thermal_damage",
        "propellant_leak",
        "valve_failure",
        "sensor_failure",
        "abort",
    ],
    "impact": [
        "vehicle_loss",
        "booster_loss",
        "ship_loss",
        "pad_damage",
        "test_abort",
        "mission_loss",
        "partial_damage",
        "safe_abort",
        "successful_recovery",
    ],
    "cause": [
        "unknown",
        "design_issue",
        "manufacturing_defect",
        "operational_error",
        "environmental",
        "hypothesis_only",
    ],
}


def load_incidents(path: Path) -> List[Dict[str, object]]:
    incidents = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            incidents.append(json.loads(line))
    return incidents


def load_labels(path: Path) -> Dict[str, Dict[str, str]]:
    labels: Dict[str, Dict[str, str]] = {}
    if not path.exists():
        return labels
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            incident_id = row.get("incident_id", "").strip()
            if not incident_id:
                continue
            labels[incident_id] = row
    return labels


def prompt_multiselect(field: str) -> List[str]:
    options = LABEL_OPTIONS[field]
    print(f"\nSelect {field} labels (comma-separated numbers, or leave blank for none):")
    for idx, option in enumerate(options, start=1):
        print(f"  {idx:2d}. {option}")
    raw = input("Selection: ").strip()
    if not raw:
        return []
    selected = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit() or not (1 <= int(part) <= len(options)):
            print(f"Skipping invalid selection: {part}")
            continue
        selected.append(options[int(part) - 1])
    return sorted(set(selected))


def write_labels(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["incident_id", "subsystem", "failure_mode", "impact", "cause", "notes"]
    rows_sorted = sorted(rows, key=lambda r: r["incident_id"])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_sorted:
            writer.writerow(row)


def label_incident(incident: Dict[str, object]) -> Dict[str, str]:
    print("\n" + "=" * 80)
    print(f"Incident: {incident['incident_name']} ({incident['incident_id']})")
    print("Sources:")
    for source in incident.get("sources", []):
        print(f"  - {source.get('source_id')}: {source.get('title')} | URL: {source.get('url')}")
        print(f"    Summary: {source.get('source_summary')}")
    print("\nNarrative:\n")
    print(incident.get("text", ""))

    labels = {}
    for field in ["subsystem", "failure_mode", "impact", "cause"]:
        labels[field] = ";".join(prompt_multiselect(field))

    notes = input("Notes (optional): ").strip()
    return {
        "incident_id": incident["incident_id"],
        "subsystem": labels["subsystem"],
        "failure_mode": labels["failure_mode"],
        "impact": labels["impact"],
        "cause": labels["cause"],
        "notes": notes,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive labeling tool")
    ap.add_argument("--incidents", default="data/processed/incidents.jsonl", help="Incidents JSONL")
    ap.add_argument("--labels", default="data/labels.csv", help="Output labels CSV")
    ap.add_argument("--resume", action="store_true", help="Skip incidents already labeled")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    incidents = load_incidents(root / args.incidents)
    existing = load_labels(root / args.labels)
    rows = list(existing.values())

    for incident in incidents:
        incident_id = incident["incident_id"]
        if args.resume and incident_id in existing:
            continue
        rows.append(label_incident(incident))
        write_labels(root / args.labels, rows)
        print(f"Saved labels to {args.labels}")

    print("Labeling complete.")


if __name__ == "__main__":
    main()

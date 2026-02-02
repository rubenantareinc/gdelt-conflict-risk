#!/usr/bin/env python3
"""Smoke test: build dataset, run baseline, evaluate, generate incident card."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def generate_incident_card(incidents_path: Path, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    with incidents_path.open("r", encoding="utf-8") as f:
        first = json.loads(next(f))

    card_path = outdir / f"incident_{first['incident_id']}.md"
    sources_lines = []
    for source in first.get("sources", []):
        sources_lines.append(
            f"- {source.get('source_id')}: {source.get('title')} (URL: {source.get('url')})"
        )
    card = "\n".join(
        [
            f"# Incident Card: {first['incident_name']}",
            "",
            f"**Incident ID:** {first['incident_id']}",
            f"**Date:** {first.get('date', '') or 'N/A'}",
            "",
            "## Narrative",
            first.get("text", ""),
            "",
            "## Sources",
            "\n".join(sources_lines) if sources_lines else "- None",
        ]
    )
    card_path.write_text(card + "\n", encoding="utf-8")
    return card_path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    run(["python", str(root / "src/ingest/build_dataset.py")])
    run(["python", str(root / "src/baselines/empty_baseline.py")])
    run(["python", str(root / "src/evaluation/evaluate_labels.py")])

    card_path = generate_incident_card(
        root / "data/processed/incidents.jsonl", root / "outputs" / "incident_cards"
    )
    print(f"Wrote incident card to {card_path}")


if __name__ == "__main__":
    main()

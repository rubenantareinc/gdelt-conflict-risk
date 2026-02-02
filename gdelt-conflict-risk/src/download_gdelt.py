#!/usr/bin/env python3
"""
Helper: download article text from GDELT Doc 2.1 API.

This is OPTIONAL and not used by default (the repo runs with toy data).
Use it to create a CSV matching the schema expected by src/train_eval.py.

Usage example:
  python src/download_gdelt.py --query "Tripoli militia" --start 20240101 --end 20240331 --out data/gdelt_articles.csv

Notes:
- GDELT returns JSON; results may be rate-limited.
- You may need to paginate (this script includes simple paging).
- The Doc API does not provide reliable city-level geocoding. We therefore store
  `location_raw` and set `city` to a coarse proxy (source country). For true
  city-level modeling, replace `city` after running a geocoder or using GDELT
  GKG/location fields.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="Full-text query")
    ap.add_argument("--start", required=True, help="YYYYMMDD")
    ap.add_argument("--end", required=True, help="YYYYMMDD")
    ap.add_argument("--mode", default="ArtList", help="GDELT Doc API mode")
    ap.add_argument("--max_records", type=int, default=250)
    ap.add_argument("--out", default="data/gdelt_articles.csv")
    args = ap.parse_args()

    base = "https://api.gdeltproject.org/api/v2/doc/doc"
    all_rows = []
    start = 0

    while True:
        params = {
            "query": args.query,
            "mode": args.mode,
            "format": "json",
            "startdatetime": args.start + "000000",
            "enddatetime": args.end + "235959",
            "maxrecords": args.max_records,
            "startrecord": start + 1,
        }
        r = requests.get(base, params=params, timeout=30)
        r.raise_for_status()
        j = r.json()
        articles = j.get("articles", [])
        if not articles:
            break

        for a in articles:
            source_country = a.get("sourceCountry") or ""
            all_rows.append(
                {
                    "date": a.get("seendate"),
                    "city": source_country,
                    "country": source_country,
                    "location_raw": source_country,
                    "text": (a.get("title","") + " " + a.get("seendate","") + " " + a.get("url","")).strip(),
                }
            )

        start += len(articles)
        if len(articles) < args.max_records:
            break
        time.sleep(0.8)

    df = pd.DataFrame(all_rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")


if __name__ == "__main__":
    main()

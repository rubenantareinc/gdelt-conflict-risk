#!/usr/bin/env python3
"""
GDELT-style city-month text pipeline (toy example included).

What this provides:
- Converts article-level text into city-month "documents" (concatenated text)
- Extracts TF-IDF features from those documents
- Adds numeric cues (example columns provided in toy_articles.csv)
- Runs ablations: text-only, numeric-only, combined
- Evaluates with a TIME-AWARE split by month

This repo includes a small toy dataset so the pipeline runs end-to-end without external downloads.
For real experiments, replace the toy data with GDELT-derived articles/events.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score


@dataclass
class SplitResult:
    name: str
    mae: float
    r2: float


def month_str(dt: pd.Series) -> pd.Series:
    return pd.to_datetime(dt).dt.to_period("M").astype(str)


def build_city_month_docs(articles: pd.DataFrame) -> pd.DataFrame:
    """Aggregate article-level rows into city-month documents."""
    df = articles.copy()
    df["month"] = month_str(df["date"])
    grouped = df.groupby(["city", "month"], as_index=False).agg(
        doc_text=("text", lambda x: " ".join(map(str, x))),
        numeric_refugees_m=("numeric_refugees_m", "mean"),
        numeric_gdp_growth=("numeric_gdp_growth", "mean"),
        n_articles=("text", "count"),
    )
    return grouped


def time_aware_split(df: pd.DataFrame, train_end_month: str):
    """Train on months <= train_end_month, test on later months."""
    train = df[df["month"] <= train_end_month].copy()
    test = df[df["month"] > train_end_month].copy()
    return train, test


def fit_eval(train: pd.DataFrame, test: pd.DataFrame, mode: str) -> SplitResult:
    y_train = train["future_conflict_events"].to_numpy()
    y_test = test["future_conflict_events"].to_numpy()

    X_train_parts = []
    X_test_parts = []

    if mode in ("text", "combined"):
        vec = TfidfVectorizer(min_df=1, max_features=2000, ngram_range=(1,2))
        Xtr_text = vec.fit_transform(train["doc_text"])
        Xte_text = vec.transform(test["doc_text"])
        X_train_parts.append(Xtr_text)
        X_test_parts.append(Xte_text)

    if mode in ("numeric", "combined"):
        num_cols = ["numeric_refugees_m", "numeric_gdp_growth", "n_articles"]
        Xtr_num = train[num_cols].to_numpy(dtype=float)
        Xte_num = test[num_cols].to_numpy(dtype=float)
        # convert to sparse-like by keeping as numpy; Ridge handles both
        X_train_parts.append(Xtr_num)
        X_test_parts.append(Xte_num)

    # concatenate
    # If we have mixed sparse + dense, convert sparse to array (toy-sized). For real data, use scipy.sparse.hstack.
    def to_dense(x):
        return x.toarray() if hasattr(x, "toarray") else x

    X_train = np.hstack([to_dense(x) for x in X_train_parts])
    X_test = np.hstack([to_dense(x) for x in X_test_parts])

    model = Ridge(alpha=1.0, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return SplitResult(
        name=mode,
        mae=float(mean_absolute_error(y_test, preds)),
        r2=float(r2_score(y_test, preds)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--articles", default="data/toy_articles.csv", help="Article-level CSV (GDELT-derived or toy)")
    ap.add_argument("--labels", default="data/toy_city_month_labels.csv", help="City-month labels CSV")
    ap.add_argument("--train_end_month", default="2024-02", help="Last month included in training (YYYY-MM)")
    ap.add_argument("--outdir", default="outputs", help="Directory for metrics CSV")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    articles = pd.read_csv(root / args.articles)
    labels = pd.read_csv(root / args.labels)

    docs = build_city_month_docs(articles)
    merged = docs.merge(labels, on=["city", "month"], how="inner")

    train, test = time_aware_split(merged, args.train_end_month)
    if len(test) == 0 or len(train) == 0:
        raise SystemExit("Time split produced empty train or test. Adjust --train_end_month or provide more months.")

    results = []
    for mode in ["text", "numeric", "combined"]:
        results.append(fit_eval(train, test, mode))

    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    res_df = pd.DataFrame([r.__dict__ for r in results])
    res_df.to_csv(outdir / "ablation_time_split_metrics.csv", index=False)
    with (outdir / "ablation_time_split_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(res_df.to_dict(orient="records"), f, indent=2)

    print(res_df.to_string(index=False))
    print(f"Saved: {outdir / 'ablation_time_split_metrics.csv'}")
    print(f"Saved: {outdir / 'ablation_time_split_metrics.json'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Leakage demo: improper aggregation can leak future information into past features.

This script demonstrates *the failure mode* described in the SOP:
- If you accidentally aggregate text at the city level (ignoring month),
  then copy that same city-level document into every month, you are
  effectively injecting future-month text into earlier examples.

We show:
1) Correct city-month docs + time-aware split
2) Leaky city-level docs copied into each month + same split

This runs on the included toy dataset. Replace with real GDELT-derived data for real experiments.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score


def month_str(dt: pd.Series) -> pd.Series:
    return pd.to_datetime(dt).dt.to_period("M").astype(str)


def correct_docs(articles: pd.DataFrame) -> pd.DataFrame:
    df = articles.copy()
    df["month"] = month_str(df["date"])
    return df.groupby(["city", "month"], as_index=False).agg(doc_text=("text", lambda x: " ".join(map(str, x))),
                                                            n_articles=("text","count"))


def leaky_docs(articles: pd.DataFrame) -> pd.DataFrame:
    df = articles.copy()
    df["month"] = month_str(df["date"])
    # BUG: aggregate by city only -> document contains text from ALL months
    city_docs = df.groupby(["city"], as_index=False).agg(doc_text=("text", lambda x: " ".join(map(str, x))))
    # BUG: copy same doc to all months observed for that city
    months = df[["city","month"]].drop_duplicates()
    leak = months.merge(city_docs, on="city", how="left")
    leak["n_articles"] = np.nan
    return leak


def time_split(df: pd.DataFrame, train_end_month: str):
    train = df[df["month"] <= train_end_month].copy()
    test = df[df["month"] > train_end_month].copy()
    return train, test


def eval_text_only(df: pd.DataFrame, train_end_month: str):
    train, test = time_split(df, train_end_month)
    vec = TfidfVectorizer(min_df=1, max_features=2000, ngram_range=(1,2))
    Xtr = vec.fit_transform(train["doc_text"])
    Xte = vec.transform(test["doc_text"])
    ytr = train["future_conflict_events"].to_numpy()
    yte = test["future_conflict_events"].to_numpy()

    model = Ridge(alpha=1.0, random_state=0)
    model.fit(Xtr, ytr)
    preds = model.predict(Xte)
    return float(mean_absolute_error(yte, preds)), float(r2_score(yte, preds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--articles", default="data/toy_articles.csv")
    ap.add_argument("--labels", default="data/toy_city_month_labels.csv")
    ap.add_argument("--train_end_month", default="2024-02")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    articles = pd.read_csv(root / args.articles)
    labels = pd.read_csv(root / args.labels)

    corr = correct_docs(articles).merge(labels, on=["city","month"], how="inner")
    leak = leaky_docs(articles).merge(labels, on=["city","month"], how="inner")

    corr_mae, corr_r2 = eval_text_only(corr, args.train_end_month)
    leak_mae, leak_r2 = eval_text_only(leak, args.train_end_month)

    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([
        {"setting":"correct_city_month", "mae":corr_mae, "r2":corr_r2},
        {"setting":"LEAKY_city_level_copied", "mae":leak_mae, "r2":leak_r2},
    ])
    summary.to_csv(outdir / "leakage_demo_metrics.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Saved: {outdir / 'leakage_demo_metrics.csv'}")


if __name__ == "__main__":
    main()

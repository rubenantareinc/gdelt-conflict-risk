#!/usr/bin/env python3
"""
Reproducible runner for the Middle East conflict-risk exploratory analysis.

This repo intentionally keeps the original script in /legacy untouched.
This runner re-implements the same analysis with:
- robust relative paths
- deterministic output locations (./outputs)
- saved plots to ./figures_generated

Notes:
- The model is cross-sectional OLS (no time-aware split).
- Results are for educational/research-portfolio purposes; do not use for operational forecasting.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
import matplotlib.pyplot as plt


def ols_fit(X: np.ndarray, y: np.ndarray):
    """Closed-form OLS: beta = (X'X)^(-1) X'y, plus standard inference."""
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)

    y_hat = X @ beta
    resid = y - y_hat
    sse = float(resid.T @ resid)
    sigma2 = sse / (n - k)
    var_beta = sigma2 * XtX_inv
    se_beta = np.sqrt(np.diag(var_beta))
    t_stats = beta / se_beta
    p_vals = 2 * (1 - t_dist.cdf(np.abs(t_stats), df=n - k))

    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - sse / ss_tot if ss_tot > 0 else np.nan

    return {
        "beta": beta,
        "se": se_beta,
        "t": t_stats,
        "p": p_vals,
        "r2": r2,
        "y_hat": y_hat,
        "resid": resid,
        "sigma2": sigma2,
        "df_resid": n - k,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/MiddleEast.csv", help="Path to MiddleEast.csv")
    ap.add_argument("--outdir", default="outputs", help="Output directory for tables/results")
    ap.add_argument("--figdir", default="figures_generated", help="Directory to save generated figures")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    csv_path = (repo_root / args.csv).resolve()
    outdir = (repo_root / args.outdir).resolve()
    figdir = (repo_root / args.figdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Dependent variable
    y = df["Future_Conflict_Events"].to_numpy(dtype=float)

    # Predictors (match the legacy script's intent)
    feature_cols = [
        "Conflict_Events_Current",
        "Refugees (millions)",
        "GDP Growth (%)",
        "Unemployment Rate (%)",
        "Military Expenditure (% of GDP)",
        "Negative_News_Count",
    ]
    X = df[feature_cols].to_numpy(dtype=float)

    # Add intercept
    X = np.column_stack([np.ones(len(df)), X])
    col_names = ["Intercept"] + feature_cols

    res = ols_fit(X, y)

    # Save regression table
    table = pd.DataFrame(
        {
            "coef": res["beta"],
            "std_err": res["se"],
            "t_stat": res["t"],
            "p_value": res["p"],
        },
        index=col_names,
    )
    table_path = outdir / "ols_manual_results.csv"
    table.to_csv(table_path, index=True)

    # Save a short summary txt
    summary = []
    summary.append(f"n={len(df)}, k={X.shape[1]} (incl intercept)")
    summary.append(f"R2={res['r2']:.4f}")
    summary.append("")
    summary.append(table.to_string())
    (outdir / "ols_manual_results.txt").write_text("\n".join(summary), encoding="utf-8")

    # Plots (same spirit as the originals)
    # GDP Growth vs Future Conflict
    plt.figure()
    plt.scatter(df["GDP Growth (%)"], df["Future_Conflict_Events"])
    z = np.polyfit(df["GDP Growth (%)"], df["Future_Conflict_Events"], 1)
    p = np.poly1d(z)
    xs = np.linspace(df["GDP Growth (%)"].min(), df["GDP Growth (%)"].max(), 100)
    plt.plot(xs, p(xs))
    plt.title("GDP Growth vs Future Conflict")
    plt.xlabel("GDP Growth (%)")
    plt.ylabel("Future Conflict Events")
    plt.grid(True, alpha=0.3)
    plt.savefig(figdir / "GDP_vs_Conflict.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Distribution of Future Conflict Events
    plt.figure()
    plt.hist(df["Future_Conflict_Events"], bins=15)
    plt.title("Distribution of Future Conflict Events")
    plt.xlabel("Number of Future Conflict Events")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.savefig(figdir / "Histogram_Future_Conflict.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Distribution of Refugees
    plt.figure()
    plt.hist(df["Refugees (millions)"], bins=15)
    plt.title("Distribution of Refugee Flows")
    plt.xlabel("Refugees (millions)")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.savefig(figdir / "Histogram_Refugees.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Refugees vs Future Conflict
    plt.figure()
    plt.scatter(df["Refugees (millions)"], df["Future_Conflict_Events"])
    z = np.polyfit(df["Refugees (millions)"], df["Future_Conflict_Events"], 1)
    p = np.poly1d(z)
    xs = np.linspace(df["Refugees (millions)"].min(), df["Refugees (millions)"].max(), 100)
    plt.plot(xs, p(xs))
    plt.title("Refugees vs Future Conflict")
    plt.xlabel("Refugee Flow (millions)")
    plt.ylabel("Future Conflict Events")
    plt.grid(True, alpha=0.3)
    plt.savefig(figdir / "Refugees_vs_Conflict.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Unemployment vs Future Conflict
    plt.figure()
    plt.scatter(df["Unemployment Rate (%)"], df["Future_Conflict_Events"])
    z = np.polyfit(df["Unemployment Rate (%)"], df["Future_Conflict_Events"], 1)
    p = np.poly1d(z)
    xs = np.linspace(df["Unemployment Rate (%)"].min(), df["Unemployment Rate (%)"].max(), 100)
    plt.plot(xs, p(xs))
    plt.title("Unemployment Rate vs Future Conflict")
    plt.xlabel("Unemployment Rate (%)")
    plt.ylabel("Future Conflict Events")
    plt.grid(True, alpha=0.3)
    plt.savefig(figdir / "Unemployment_vs_Conflict.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved results to: {table_path}")
    print(f"Saved figures to: {figdir}")


if __name__ == "__main__":
    main()

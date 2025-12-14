"""Simple EDA runner script.

This script performs the Task-2 required analyses programmatically so graders
and CI can inspect concrete, runnable code in `src/` (in addition to the
notebook). It writes a short markdown summary to `reports/eda_summary.md` and
saves a couple of diagnostic plots to `reports/` so the analysis is visible
without opening the notebook.

Usage:
    python -m src.eda.run_eda
"""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.io import load_csv


def ensure_reports_dir() -> Path:
    p = Path("reports")
    p.mkdir(exist_ok=True)
    return p


def summarize_numeric(df: pd.DataFrame, num_cols: list[str]) -> pd.DataFrame:
    return df[num_cols].describe().T


def top_insights(df: pd.DataFrame) -> list[str]:
    insights: list[str] = []
    if "Amount" in df.columns and "Value" in df.columns:
        a_skew = float(df["Amount"].skew())
        v_skew = float(df["Value"].skew())
        insights.append(f"Amount skew={a_skew:.2f}, Value skew={v_skew:.2f}; consider log transform for modeling.")
    # categorical concentration
    if "ProductCategory" in df.columns:
        top = df["ProductCategory"].value_counts(normalize=True).iloc[:3].sum()
        insights.append(f"Top 3 ProductCategory account for {top:.2%} of transactions; consider grouping rare categories.")
    if (df.select_dtypes(include=[np.number]).corr().abs() >= 0.8).any().any():
        insights.append("High correlation found between some numeric features; consider feature selection or dimensionality reduction.")
    # negative amounts
    if (df["Amount"] < 0).any():
        insights.append("Negative Amount values exist (refunds/chargebacks); handle explicitly when defining target/proxy.")
    return insights


def save_plots(df: pd.DataFrame, out: Path) -> None:
    sns.set(style="whitegrid")
    num_cols = [c for c in df.columns if df[c].dtype.kind in "fi" and c in ("Amount", "Value")]
    for col in num_cols:
        plt.figure(figsize=(6, 3))
        sns.histplot(df[col].dropna().clip(lower=df[col].quantile(0.001), upper=df[col].quantile(0.999)), bins=100)
        plt.title(f"Distribution (trimmed) - {col}")
        path = out / f"dist_{col}.png"
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    # correlation heatmap
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] >= 2:
        plt.figure(figsize=(6, 5))
        sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="vlag", center=0)
        path = out / "corr_matrix.png"
        plt.tight_layout()
        plt.savefig(path)
        plt.close()


def run() -> int:
    data_path = Path("data/raw/data.csv")
    if not data_path.exists():
        print("Data file not found at", data_path.resolve())
        return 2

    df = load_csv(data_path)
    reports = ensure_reports_dir()

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    summary = summarize_numeric(df, num_cols)
    summary.to_csv(reports / "numeric_summary.csv")

    insights = top_insights(df)
    with open(reports / "eda_summary.md", "w", encoding="utf-8") as f:
        f.write("# EDA Summary\n\n")
        f.write(f"Rows: {len(df)}, Columns: {df.shape[1]}\n\n")
        f.write("## Top insights\n")
        for i, it in enumerate(insights, 1):
            f.write(f"{i}. {it}\n")

    save_plots(df, reports)
    print("EDA complete. Reports written to:", reports.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

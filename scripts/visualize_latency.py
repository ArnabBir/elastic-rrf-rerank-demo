from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd

from src.utils import get_data_dir, get_report_dir

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        plt.style.use("ggplot")


def main():
    data_dir = get_data_dir()
    report_dir = get_report_dir(data_dir)
    latency_path = report_dir / "latency.csv"
    if not latency_path.exists():
        print(f"Run 04_run_queries.py first. Expected: {latency_path}")
        sys.exit(1)

    df = pd.read_csv(latency_path)
    df = df.head(20)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(df))
    width = 0.7

    ax.bar(x, df["bm25_ms"], width, label="BM25", color="#3498db")
    ax.bar(x, df["knn_ms"], width, bottom=df["bm25_ms"], label="kNN", color="#2ecc71")
    ax.bar(x, df["fusion_ms"], width, bottom=df["bm25_ms"] + df["knn_ms"], label="Fusion", color="#f39c12")
    ax.bar(x, df["rerank_ms"], width, bottom=df["bm25_ms"] + df["knn_ms"] + df["fusion_ms"], label="Rerank", color="#e74c3c")

    ax.set_xlabel("Query", labelpad=8)
    ax.set_ylabel("Latency (ms)", labelpad=8)
    ax.set_title("Latency Breakdown per Query (first 20)", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(df["query_id"], rotation=45, ha="right")
    ax.tick_params(axis="x", pad=6)
    ax.tick_params(axis="y", pad=6)
    ax.legend()

    out = report_dir / "latency_breakdown.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

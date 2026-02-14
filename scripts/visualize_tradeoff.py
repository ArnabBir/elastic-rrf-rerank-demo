from __future__ import annotations

import json
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
    metrics_path = report_dir / "metrics.json"
    latency_path = report_dir / "latency.csv"
    if not metrics_path.exists() or not latency_path.exists():
        print("Run 04_run_queries.py and 05_compute_metrics.py first.")
        sys.exit(1)

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    df = pd.read_csv(latency_path)

    avg_bm25 = df["bm25_ms"].mean()
    avg_knn = df["knn_ms"].mean()
    avg_hybrid = (df["bm25_ms"] + df["knn_ms"] + df["fusion_ms"]).mean()
    avg_rerank = df["total_ms"].mean()

    systems = ["bm25", "knn", "hybrid_rrf", "hybrid_rrf_rerank"]
    labels = ["BM25", "kNN", "Hybrid RRF", "Hybrid + Rerank"]
    ndcgs = [metrics[s]["ndcg@10"] for s in systems]
    latencies = [avg_bm25, avg_knn, avg_hybrid, avg_rerank]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c"]
    for i, (label, ndcg, lat) in enumerate(zip(labels, ndcgs, latencies)):
        ax.scatter(lat, ndcg, s=200, c=colors[i], label=label, zorder=5, edgecolors="black", linewidths=1.5)
        ax.annotate(label, (lat, ndcg), xytext=(10, 10), textcoords="offset points", fontsize=10)

    ax.set_xlabel("Avg Latency (ms)", labelpad=8)
    ax.set_ylabel("NDCG@10", labelpad=8)
    ax.set_title("Quality vs Latency Tradeoff", pad=12)
    ax.tick_params(axis="x", pad=6)
    ax.tick_params(axis="y", pad=6)
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = report_dir / "tradeoff_quality_latency.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

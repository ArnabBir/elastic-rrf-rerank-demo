from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        plt.style.use("ggplot")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11
import numpy as np
import pandas as pd

_PROJECT = Path(__file__).resolve().parents[1]
REPORTS = _PROJECT / "reports"
ASSETS = _PROJECT / "assets"

EXPERIMENTS = [
    ("data_lite_4bd36c3b", "data_lite", "Lite (15 queries)"),
    ("data_large_run1_c10fcdee", "data_large_run1", "Large baseline (750q)"),
    ("data_large_c10fcdee", "data_large", "Large improved (750q)"),
]

SYSTEMS = ["bm25", "knn", "hybrid_rrf", "hybrid_rrf_rerank"]
LABELS = ["BM25", "kNN", "Hybrid RRF", "Hybrid + Rerank"]
COLORS = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c"]


def load_experiment_data():
    data = []
    for dir_name, exp_id, exp_label in EXPERIMENTS:
        report_dir = REPORTS / dir_name
        metrics_path = report_dir / "metrics.json"
        latency_path = report_dir / "latency.csv"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        row = {"exp_id": exp_id, "exp_label": exp_label}
        for s in SYSTEMS:
            row[f"{s}_ndcg"] = metrics.get(s, {}).get("ndcg@10", 0)
            row[f"{s}_mrr"] = metrics.get(s, {}).get("mrr@10", 0)
            row[f"{s}_recall"] = metrics.get(s, {}).get("recall@50", 0)
        if latency_path.exists():
            df = pd.read_csv(latency_path)
            row["avg_latency_ms"] = df["total_ms"].mean()
        else:
            row["avg_latency_ms"] = 0
        data.append(row)
    return data


def chart_cross_experiment_ndcg(data):
    """Grouped bar: NDCG@10 by system, across experiments."""
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(EXPERIMENTS))
    width = 0.2

    for i, (sys, label) in enumerate(zip(SYSTEMS, LABELS)):
        vals = [d[f"{sys}_ndcg"] for d in data]
        offset = (i - 1.5) * width
        ax.bar(x + offset, vals, width, label=label, color=COLORS[i])

    ax.set_ylabel("NDCG@10", fontsize=12)
    ax.set_title("NDCG@10 by System Across Experiments", fontsize=14, pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels([d["exp_label"] for d in data], fontsize=10)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="both", pad=6)
    fig.tight_layout()
    fig.savefig(ASSETS / "blog_ndcg_cross_experiment.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {ASSETS}/blog_ndcg_cross_experiment.png")


def chart_radar_per_experiment(data):
    """One radar chart per experiment: compare the 4 systems (BM25, kNN, Hybrid RRF, Hybrid+Rerank)."""
    categories = ["NDCG@10", "MRR@10", "Recall@50"]
    num_vars = len(categories)
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]

    for d in data:
        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection="polar"))

        for i, (sys, label) in enumerate(zip(SYSTEMS, LABELS)):
            ndcg = d[f"{sys}_ndcg"]
            mrr = d[f"{sys}_mrr"]
            recall = d[f"{sys}_recall"]
            values = [ndcg, mrr, recall] + [ndcg]
            ax.plot(angles, values, "o-", linewidth=2, label=label, color=COLORS[i])
            ax.fill(angles, values, alpha=0.12, color=COLORS[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=10)
        ax.set_title(f"Model Comparison — {d['exp_label']}", pad=20, fontsize=14)
        fig.tight_layout()
        slug = d["exp_id"].replace(" ", "_")
        out = ASSETS / f"blog_radar_{slug}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {out}")


def chart_hybrid_vs_rerank(data):
    """Hybrid RRF vs Hybrid+Rerank: does reranking help?"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    metrics = ["ndcg", "mrr", "recall"]
    titles = ["NDCG@10", "MRR@10", "Recall@50"]

    for ax, m, title in zip(axes, metrics, titles):
        hybrid = [d["hybrid_rrf_" + m] for d in data]
        rerank = [d["hybrid_rrf_rerank_" + m] for d in data]
        x = np.arange(len(data))
        width = 0.35
        ax.bar(x - width / 2, hybrid, width, label="Hybrid RRF", color="#f39c12")
        ax.bar(x + width / 2, rerank, width, label="Hybrid + Rerank", color="#e74c3c")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([d["exp_label"] for d in data], rotation=15, ha="right")
        ax.legend()
        ax.set_ylim(0, 1.0)

    fig.suptitle("Hybrid RRF vs Hybrid + Rerank Across Experiments", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(ASSETS / "blog_rerank_impact.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {ASSETS}/blog_rerank_impact.png")


def chart_latency_tradeoff(data):
    """Quality vs latency: hybrid_rrf_rerank across experiments."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, d in enumerate(data):
        ndcg = d["hybrid_rrf_rerank_ndcg"]
        lat = d.get("avg_latency_ms", 0) / 1000  # seconds
        ax.scatter(lat, ndcg, s=300, c=COLORS[i], label=d["exp_label"], zorder=5, edgecolors="black", linewidths=2)
        ax.annotate(d["exp_label"], (lat, ndcg), xytext=(12, 8), textcoords="offset points", fontsize=10)

    ax.set_xlabel("Avg Latency (seconds)", fontsize=12)
    ax.set_ylabel("NDCG@10 (Hybrid + Rerank)", fontsize=12)
    ax.set_title("Quality vs Latency Tradeoff", fontsize=14, pad=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ASSETS / "blog_latency_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {ASSETS}/blog_latency_tradeoff.png")


def chart_stacked_system_comparison(data):
    """Horizontal grouped bars: all 4 systems, 3 metrics, per experiment."""
    fig, ax = plt.subplots(figsize=(14, 6))

    exp_labels = [d["exp_label"] for d in data]
    x = np.arange(len(exp_labels))
    width = 0.2

    for i, (sys, label) in enumerate(zip(SYSTEMS, LABELS)):
        ndcg_vals = [d[f"{sys}_ndcg"] for d in data]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, ndcg_vals, width, label=label, color=COLORS[i])
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8, fontweight="bold")

    ax.set_ylabel("NDCG@10")
    ax.set_title("NDCG@10 by System and Experiment")
    ax.set_xticks(x)
    ax.set_xticklabels(exp_labels, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(ASSETS / "blog_system_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {ASSETS}/blog_system_comparison.png")


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    data = load_experiment_data()
    if not data:
        print("No experiment data found. Run 04, 05, 06 on data_lite and data_large first.")
        return

    chart_cross_experiment_ndcg(data)
    chart_radar_per_experiment(data)
    chart_hybrid_vs_rerank(data)
    chart_stacked_system_comparison(data)
    if any(d.get("avg_latency_ms", 0) > 0 for d in data):
        chart_latency_tradeoff(data)

    print(f"\nDone. Assets in {ASSETS}/")


if __name__ == "__main__":
    main()

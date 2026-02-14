from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from src.utils import get_data_dir, get_report_dir

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")
except OSError:
    plt.style.use("ggplot")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11


def main():
    data_dir = get_data_dir()
    report_dir = get_report_dir(data_dir)
    metrics_path = report_dir / "metrics.json"
    if not metrics_path.exists():
        print(f"Run 05_compute_metrics.py first. Expected: {metrics_path}")
        sys.exit(1)

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    systems = list(metrics.keys())
    labels = [s.replace("_", " ").title() for s in systems]

    ndcg = [metrics[s]["ndcg@10"] for s in systems]
    mrr = [metrics[s]["mrr@10"] for s in systems]
    recall = [metrics[s]["recall@50"] for s in systems]

    x = np.arange(len(systems))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, ndcg, width, label="NDCG@10", color="#2ecc71")
    bars2 = ax.bar(x, mrr, width, label="MRR@10", color="#3498db")
    bars3 = ax.bar(x + width, recall, width, label="Recall@50", color="#9b59b6")

    ax.set_ylabel("Score", labelpad=8)
    ax.set_xlabel("System", labelpad=8)
    ax.set_title("Relevance Metrics by System", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", pad=6)
    ax.tick_params(axis="y", pad=6)
    ax.legend()
    ax.set_ylim(0, 1.05)

    out = report_dir / "metrics_bar.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

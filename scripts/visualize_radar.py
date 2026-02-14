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
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        plt.style.use("ggplot")


def main():
    data_dir = get_data_dir()
    report_dir = get_report_dir(data_dir)
    metrics_path = report_dir / "metrics.json"
    if not metrics_path.exists():
        print(f"Run 05_compute_metrics.py first. Expected: {metrics_path}")
        sys.exit(1)

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    categories = ["NDCG@10", "MRR@10", "Recall@50"]
    num_vars = len(categories)
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
    for i, (name, m) in enumerate(metrics.items()):
        values = [m["ndcg@10"], m["mrr@10"], m["recall@50"]]
        values += values[:1]
        label = name.replace("_", " ").title()
        ax.plot(angles, values, "o-", linewidth=2, label=label, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.15, color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.tick_params(pad=6)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.set_title("System Comparison (Radar)", pad=20)

    out = report_dir / "metrics_radar.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

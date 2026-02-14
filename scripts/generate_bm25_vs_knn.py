"""Generate BM25 vs kNN comparison visualization for blog."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_PROJECT = Path(__file__).resolve().parents[1]
ASSETS = _PROJECT / "assets"

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        plt.style.use("ggplot")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11


def main():
    # Conceptual retrieval strength by query type (illustrative)
    query_types = [
        "Paraphrase / synonyms\n(e.g. \"webhooks not delivering\")",
        "Exact identifiers\n(e.g. \"HTTP 429\", \"VEC-109\")",
    ]
    bm25 = [0.25, 0.95]  # BM25 weak on paraphrase, strong on exact
    knn = [0.90, 0.35]   # kNN strong on paraphrase, weak on exact

    x = np.arange(len(query_types))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6.5))
    bars1 = ax.bar(x - width / 2, bm25, width, label="BM25 (lexical)", color="#3498db", edgecolor="white", linewidth=1.2)
    bars2 = ax.bar(x + width / 2, knn, width, label="kNN (semantic)", color="#2ecc71", edgecolor="white", linewidth=1.2)

    ax.set_ylabel("Retrieval strength", fontsize=12, labelpad=8)
    ax.set_title("BM25 vs kNN: Different query types favor different methods", fontsize=14, pad=16)
    ax.set_xticks(x)
    ax.set_xticklabels(query_types, fontsize=10)
    ax.legend(loc="upper right", fontsize=11, framealpha=0.95)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["Low", "0.25", "0.5", "0.75", "High"])
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4)
    ax.tick_params(axis="both", pad=6)

    # Value annotations on bars
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(
            f"{h:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(
            f"{h:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    # Callout annotations (axes coords, below chart)
    ax.text(0.25, -0.15, "BM25 misses meaning", ha="center", fontsize=9, color="#3498db", style="italic", transform=ax.transAxes)
    ax.text(0.25, -0.22, "kNN captures intent", ha="center", fontsize=9, color="#2ecc71", style="italic", transform=ax.transAxes)
    ax.text(0.75, -0.15, "kNN misses rare tokens", ha="center", fontsize=9, color="#2ecc71", style="italic", transform=ax.transAxes)
    ax.text(0.75, -0.22, "BM25 exact match", ha="center", fontsize=9, color="#3498db", style="italic", transform=ax.transAxes)

    # Takeaway box
    props = dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#dee2e6", alpha=0.95)
    ax.text(0.5, -0.32, "Hybrid (BM25 + kNN + RRF) combines both → robust across query types", ha="center", fontsize=10, transform=ax.transAxes, bbox=props)

    fig.tight_layout()
    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / "bm25_vs_knn_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_embed_dims, get_embedding_model, get_reranker_model, get_retrieval_config
from src.utils import get_data_dir, get_report_dir, write_text


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def main():
    data_dir = get_data_dir()
    report_dir = get_report_dir(data_dir)
    metrics_path = report_dir / "metrics.json"

    if not metrics_path.exists():
        print(f"Run 05_compute_metrics.py first. Expected: {metrics_path}")
        sys.exit(1)

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rcfg = get_retrieval_config()

    n_docs = count_lines(data_dir / "documents.jsonl")
    n_queries = count_lines(data_dir / "queries.jsonl")
    n_qrels = count_lines(data_dir / "qrels.tsv") - 1  # header

    data_name = data_dir.name or "data"
    try:
        data_path = str(data_dir.relative_to(Path.cwd()))
    except ValueError:
        data_path = str(data_dir)
    embed_model = get_embedding_model()
    reranker_model = get_reranker_model()
    embed_dims = get_embed_dims()

    lines = []
    lines.append(f"# {data_name} — Run Specs\n")
    lines.append(f"Experiment run on `{data_path}`.\n")
    lines.append("## Dataset\n")
    lines.append("| Item | Count |\n")
    lines.append("|------|------:|\n")
    lines.append(f"| Documents | {n_docs:,} |\n")
    lines.append(f"| Queries | {n_queries:,} |\n")
    lines.append(f"| Qrels | {n_qrels:,} |\n")
    lines.append("\n## Models\n")
    lines.append("| Component | Model |\n")
    lines.append("|-----------|-------|\n")
    lines.append(f"| Embedding | {embed_model} ({embed_dims}d) |\n")
    lines.append(f"| Reranker | {reranker_model} |\n")
    lines.append("\n## Retrieval Config\n")
    lines.append("| Param | Value |\n")
    lines.append("|-------|------:|\n")
    lines.append(f"| BM25/kNN top-n | {rcfg.topn} |\n")
    lines.append(f"| kNN num_candidates | {rcfg.knn_candidates} |\n")
    lines.append(f"| RRF k | {rcfg.rrf_k} |\n")
    lines.append(f"| Rerank top-n | {rcfg.rerank_topn} |\n")
    lines.append("\nBM25 fields: `title^2`, `body`. Vector similarity: cosine.\n")
    lines.append("\n## Results\n")
    lines.append("| System | NDCG@10 | MRR@10 | Recall@50 |\n")
    lines.append("|--------|--------:|-------:|----------:|\n")
    for name, m in metrics.items():
        lines.append(f"| {name} | {m['ndcg@10']:.4f} | {m['mrr@10']:.4f} | {m['recall@50']:.4f} |\n")
    lines.append("\n## Outputs\n")
    lines.append("- `runs.json` — ranked doc IDs per query per system\n")
    lines.append("- `latency.csv` — per-query latency breakdown (bm25_ms, knn_ms, fusion_ms, rerank_ms)\n")
    lines.append("- `metrics.json` / `metrics.md` — aggregated metrics\n")
    lines.append("- `*.png` — bar chart, radar, latency breakdown, tradeoff scatter\n")

    out = report_dir / "README.md"
    write_text(out, "".join(lines))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

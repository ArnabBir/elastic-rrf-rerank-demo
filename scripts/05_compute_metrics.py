from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import get_data_dir, get_report_dir, write_json, write_text
from src.metrics import Qrels, evaluate_run


def load_qrels(data_dir: Path) -> Qrels:
    qrels = defaultdict(dict)
    with (data_dir / "qrels.tsv").open("r", encoding="utf-8") as f:
        next(f)
        for line in f:
            qid, doc_id, rel = line.strip().split("\t")
            qrels[qid][doc_id] = int(rel)
    return Qrels(qrels=dict(qrels))


def main():
    data_dir = get_data_dir()
    report_dir = get_report_dir(data_dir)
    runs_path = report_dir / "runs.json"

    if not runs_path.exists():
        print(f"Run 04_run_queries.py first. Expected: {runs_path}")
        sys.exit(1)

    runs = json.loads(runs_path.read_text(encoding="utf-8"))
    qrels = load_qrels(data_dir)

    metrics = {}
    for name, run in runs.items():
        metrics[name] = evaluate_run(run, qrels)

    write_json(report_dir / "metrics.json", metrics)

    lines = []
    lines.append("# Experiment Results\n")
    lines.append("| System | NDCG@10 | MRR@10 | Recall@50 | #Queries |\n|---|---:|---:|---:|---:|")
    for name, m in metrics.items():
        lines.append(f"| {name} | {m['ndcg@10']:.4f} | {m['mrr@10']:.4f} | {m['recall@50']:.4f} | {m['num_queries']} |")
    write_text(report_dir / "metrics.md", "\n".join(lines) + "\n")

    print(f"Wrote {report_dir}/metrics.json, metrics.md")


if __name__ == "__main__":
    main()

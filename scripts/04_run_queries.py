from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional
from collections import defaultdict
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

warnings.filterwarnings("ignore")

from src.es_client import get_client, get_config
from src.config import get_retrieval_config
from src.utils import read_jsonl, write_json, get_data_dir, get_report_dir
from src.embed import embed_texts
from src.fusion import rrf_fuse
from src.metrics import Qrels
from src.rerank import rerank

def load_qrels(data_dir: Path) -> Qrels:
    qrels = defaultdict(dict)
    with (data_dir / "qrels.tsv").open("r", encoding="utf-8") as f:
        next(f)
        for line in f:
            qid, doc_id, rel = line.strip().split("\t")
            qrels[qid][doc_id] = int(rel)
    return Qrels(qrels=dict(qrels))


def bm25_search(es, index: str, query: str, topn: int = 50) -> list:
    resp = es.search(
        index=index,
        size=topn,
        query={"multi_match": {"query": query, "fields": ["title^2", "body"]}},
    )
    return [hit["_id"] for hit in resp["hits"]["hits"]]


def knn_search(es, index: str, query_vec: list, topn: int = 50, candidates: int = 100) -> list:
    resp = es.search(
        index=index,
        size=topn,
        knn={
            "field": "embedding",
            "query_vector": query_vec,
            "k": topn,
            "num_candidates": candidates,
        },
    )
    return [hit["_id"] for hit in resp["hits"]["hits"]]


def fetch_doc_texts(es, index: str, doc_ids: list) -> dict:
    mget = es.mget(index=index, ids=doc_ids)
    out = {}
    for doc in mget["docs"]:
        if not doc.get("found"):
            continue
        src = doc["_source"]
        out[doc["_id"]] = f"{src.get('title','')}\n\n{src.get('body','')}"
    return out


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, (time.perf_counter() - t0) * 1000


def load_existing_runs(report_dir: Path) -> tuple[dict, list]:
    runs_path = report_dir / "runs.json"
    latency_path = report_dir / "latency.csv"
    runs = {"bm25": {}, "knn": {}, "hybrid_rrf": {}, "hybrid_rrf_rerank": {}}
    latency_rows = []
    if runs_path.exists():
        data = json.loads(runs_path.read_text(encoding="utf-8"))
        runs = data
    if latency_path.exists():
        df = pd.read_csv(latency_path)
        latency_rows = df.to_dict("records")
    return runs, latency_rows


def save_runs(report_dir: Path, runs: dict, latency_rows: list) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    runs_path = report_dir / "runs.json"
    latency_path = report_dir / "latency.csv"
    tmp_runs = report_dir / "runs.json.tmp"
    tmp_latency = report_dir / "latency.csv.tmp"
    write_json(tmp_runs, runs)
    pd.DataFrame(latency_rows).to_csv(tmp_latency, index=False)
    tmp_runs.replace(runs_path)
    tmp_latency.replace(latency_path)


def log(msg: str, log_file: Optional[Path], detached: bool) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    if log_file:
        with log_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    if not detached:
        print(line, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run queries incrementally with resume support")
    parser.add_argument(
        "--detached",
        action="store_true",
        help="Run in background; logs go to report_dir/progress.log",
    )
    args = parser.parse_args()

    if args.detached:
        script = Path(__file__).resolve()
        cwd = script.parents[1]
        env = os.environ.copy()
        proc = subprocess.Popen(
            [sys.executable, str(script)],
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        report_dir = get_report_dir(get_data_dir())
        log_path = report_dir / "progress.log"
        print(f"Running in background. PID={proc.pid}. Log: {log_path}")
        sys.exit(0)

    data_dir = get_data_dir()
    report_dir = get_report_dir(data_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    log_file = report_dir / "progress.log"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_file.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] Started\n")
    if not args.detached:
        print(f"Log: {log_file}")

    es = get_client()
    cfg = get_config()
    queries = read_jsonl(data_dir / "queries.jsonl")
    total = len(queries)

    runs, latency_rows = load_existing_runs(report_dir)
    n_done = min(
        len(latency_rows),
        len(runs.get("bm25", {})),
        len(runs.get("knn", {})),
        len(runs.get("hybrid_rrf", {})),
        len(runs.get("hybrid_rrf_rerank", {})),
    )
    remaining = queries[n_done:]

    if not remaining:
        log(f"All {total} queries already completed. Run 05_compute_metrics.py for final metrics.", log_file, args.detached)
        return

    if n_done > 0:
        log(f"Resuming: {n_done} done, {len(remaining)} remaining", log_file, args.detached)

    rcfg = get_retrieval_config()

    last_pct = -1
    for i, q in enumerate(remaining):
        qid = q["query_id"]
        qtext = q["query"]

        (qvec,) = embed_texts([qtext])

        bm25_ids, bm25_ms = timed(bm25_search, es, cfg.index_name, qtext, rcfg.topn)
        knn_ids, knn_ms = timed(knn_search, es, cfg.index_name, qvec, rcfg.topn, rcfg.knn_candidates)
        fused_pairs, fusion_ms = timed(rrf_fuse, [bm25_ids, knn_ids], rcfg.rrf_k, rcfg.rerank_topn)
        fused_ids = [doc_id for doc_id, _ in fused_pairs]

        doc_texts = fetch_doc_texts(es, cfg.index_name, fused_ids[:rcfg.rerank_topn])
        rerank_input = [(doc_id, doc_texts.get(doc_id, "")) for doc_id in fused_ids[:rcfg.rerank_topn]]
        reranked_pairs, rerank_ms = timed(rerank, qtext, rerank_input)
        reranked_ids = [doc_id for doc_id, _ in reranked_pairs] + fused_ids[rcfg.rerank_topn:]

        runs["bm25"][qid] = bm25_ids
        runs["knn"][qid] = knn_ids
        runs["hybrid_rrf"][qid] = fused_ids
        runs["hybrid_rrf_rerank"][qid] = reranked_ids

        latency_rows.append({
            "query_id": qid,
            "bm25_ms": bm25_ms,
            "knn_ms": knn_ms,
            "fusion_ms": fusion_ms,
            "rerank_ms": rerank_ms,
            "total_ms": bm25_ms + knn_ms + fusion_ms + rerank_ms,
        })

        save_runs(report_dir, runs, latency_rows)

        done = n_done + i + 1
        pct = int(100 * done / total)
        if pct >= last_pct + 10 or done == total:
            last_pct = pct
            log(f"Progress: {done}/{total} ({pct}%) — last: {qid}", log_file, args.detached)

    log(f"Done. {total} queries in {report_dir}. Run 05_compute_metrics.py for metrics.", log_file, args.detached)


if __name__ == "__main__":
    main()

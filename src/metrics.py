from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import math

@dataclass
class Qrels:
    qrels: Dict[str, Dict[str, int]]

def dcg(rels: List[int]) -> float:
    out = 0.0
    for i, r in enumerate(rels, start=1):
        out += (2**r - 1) / math.log2(i + 1)
    return out

def ndcg_at_k(ranked_doc_ids: List[str], qrels: Dict[str, int], k: int = 10) -> float:
    rels = [qrels.get(d, 0) for d in ranked_doc_ids[:k]]
    ideal_rels = sorted(qrels.values(), reverse=True)[:k]
    denom = dcg(ideal_rels)
    return 0.0 if denom == 0 else dcg(rels) / denom

def mrr_at_k(ranked_doc_ids: List[str], qrels: Dict[str, int], k: int = 10) -> float:
    for i, d in enumerate(ranked_doc_ids[:k], start=1):
        if qrels.get(d, 0) > 0:
            return 1.0 / i
    return 0.0

def recall_at_k(ranked_doc_ids: List[str], qrels: Dict[str, int], k: int = 50) -> float:
    relevant = {d for d, r in qrels.items() if r > 0}
    if not relevant:
        return 0.0
    retrieved = set(ranked_doc_ids[:k])
    return len(relevant & retrieved) / len(relevant)

def evaluate_run(
    run: Dict[str, List[str]],
    qrels_all: Qrels,
    k_ndcg: int = 10,
    k_mrr: int = 10,
    k_recall: int = 50
) -> Dict[str, float]:
    ndcgs, mrrs, recalls = [], [], []
    for qid, ranked in run.items():
        qrels = qrels_all.qrels.get(qid, {})
        ndcgs.append(ndcg_at_k(ranked, qrels, k=k_ndcg))
        mrrs.append(mrr_at_k(ranked, qrels, k=k_mrr))
        recalls.append(recall_at_k(ranked, qrels, k=k_recall))
    def avg(xs): return sum(xs)/len(xs) if xs else 0.0
    return {
        f"ndcg@{k_ndcg}": avg(ndcgs),
        f"mrr@{k_mrr}": avg(mrrs),
        f"recall@{k_recall}": avg(recalls),
        "num_queries": len(run),
    }

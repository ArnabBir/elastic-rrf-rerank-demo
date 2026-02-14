from __future__ import annotations
from typing import Dict, List, Tuple

def rrf_fuse(
    ranked_lists: List[List[str]],
    k: int = 60,
    max_out: int = 100
) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = {}
    for lst in ranked_lists:
        for i, doc_id in enumerate(lst, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + i)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused[:max_out]

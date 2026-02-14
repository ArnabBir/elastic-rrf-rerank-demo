from __future__ import annotations
import os
from typing import List, Tuple, Dict, Any
from dotenv import load_dotenv

load_dotenv()

def rerank_local_cross_encoder(query: str, docs: List[Tuple[str, str]]) -> List[Tuple[str, float]]:
    from sentence_transformers import CrossEncoder
    from src.config import get_reranker_model
    model = CrossEncoder(get_reranker_model())
    pairs = [[query, text] for _, text in docs]
    scores = model.predict(pairs).tolist()
    ranked = sorted([(doc_id, float(s)) for (doc_id, _), s in zip(docs, scores)], key=lambda x: x[1], reverse=True)
    return ranked

def rerank_http(query: str, docs: List[Tuple[str, str]]) -> List[Tuple[str, float]]:
    import requests
    url = os.getenv("RERANK_HTTP_URL")
    if not url:
        raise ValueError("RERANK_HTTP_URL not set but RERANK_MODE=http")
    header = os.getenv("RERANK_HTTP_AUTH_HEADER", "")
    headers = {}
    if header:
        # Put full header string like: "Authorization: Bearer XXX"
        if ":" in header:
            k, v = header.split(":", 1)
            headers[k.strip()] = v.strip()
    payload = {
        "query": query,
        "documents": [{"id": doc_id, "text": text} for doc_id, text in docs],
    }
    r = requests.post(url, json=payload, headers=headers, timeout=60)
    r.raise_for_status()
    data: Dict[str, Any] = r.json()
    results = data.get("results", [])
    ranked = [(item["id"], float(item.get("score", 0.0))) for item in results]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked

def rerank(query: str, docs: List[Tuple[str, str]]) -> List[Tuple[str, float]]:
    mode = os.getenv("RERANK_MODE", "local").lower().strip()
    if mode == "http":
        return rerank_http(query, docs)
    return rerank_local_cross_encoder(query, docs)

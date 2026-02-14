"""Central config from env. All scripts use these values."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_env_path = find_dotenv(usecwd=True) or str(_PROJECT_ROOT / ".env")
load_dotenv(_env_path)


@dataclass
class RetrievalConfig:
    topn: int
    knn_candidates: int
    rrf_k: int
    rerank_topn: int


def get_embed_dims() -> int:
    return int(os.getenv("EMBED_DIMS", "384"))


def get_embedding_model() -> str:
    return os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def get_reranker_model() -> str:
    return os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


def get_retrieval_config() -> RetrievalConfig:
    return RetrievalConfig(
        topn=int(os.getenv("RETRIEVAL_TOPN", "50")),
        knn_candidates=int(os.getenv("KNN_NUM_CANDIDATES", "100")),
        rrf_k=int(os.getenv("RRF_K", "60")),
        rerank_topn=int(os.getenv("RERANK_TOPN", "50")),
    )

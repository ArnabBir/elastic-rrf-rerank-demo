# data_large — Run Specs

Experiment run on `dataset/data_large`. Baseline configuration for comparison with future iterations.

## Dataset

| Item | Count |
|------|------:|
| Documents | 4,000 |
| Queries | 750 |
| Qrels | 2,000 |

## Models

| Component | Model |
|-----------|-------|
| Embedding | sentence-transformers/all-MiniLM-L6-v2 (384d) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |

## Retrieval Config

| Param | Value |
|-------|------:|
| BM25/kNN top-n | 50 |
| kNN num_candidates | 100 |
| RRF k | 60 |
| Rerank top-n | 50 |

BM25 fields: `title^2`, `body`. Vector similarity: cosine.

## Results

| System | NDCG@10 | MRR@10 | Recall@50 |
|--------|--------:|-------:|----------:|
| bm25 | 0.3421 | 0.3530 | 0.7542 |
| knn | 0.2375 | 0.2460 | 0.6591 |
| hybrid_rrf | 0.3605 | 0.3600 | 0.7776 |
| hybrid_rrf_rerank | 0.3315 | 0.3608 | 0.7776 |

## Outputs

- `runs.json` — ranked doc IDs per query per system
- `latency.csv` — per-query latency breakdown (bm25_ms, knn_ms, fusion_ms, rerank_ms)
- `metrics.json` / `metrics.md` — aggregated metrics
- `*.png` — bar chart, radar, latency breakdown, tradeoff scatter

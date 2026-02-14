# data_large — Run Specs
Experiment run on `/Users/arnab.bir/Documents/GitHub/elastic-rrf-rerank-demo/dataset/data_large`.
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
| Reranker | cross-encoder/ms-marco-MiniLM-L-12-v2 |

## Retrieval Config
| Param | Value |
|-------|------:|
| BM25/kNN top-n | 100 |
| kNN num_candidates | 200 |
| RRF k | 60 |
| Rerank top-n | 100 |

BM25 fields: `title^2`, `body`. Vector similarity: cosine.

## Results
| System | NDCG@10 | MRR@10 | Recall@50 |
|--------|--------:|-------:|----------:|
| bm25 | 0.3421 | 0.3530 | 0.7542 |
| knn | 0.2375 | 0.2460 | 0.6602 |
| hybrid_rrf | 0.3659 | 0.3647 | 0.8002 |
| hybrid_rrf_rerank | 0.3055 | 0.3139 | 0.8216 |

## Outputs
- `runs.json` — ranked doc IDs per query per system
- `latency.csv` — per-query latency breakdown (bm25_ms, knn_ms, fusion_ms, rerank_ms)
- `metrics.json` / `metrics.md` — aggregated metrics
- `*.png` — bar chart, radar, latency breakdown, tradeoff scatter

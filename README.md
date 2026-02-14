# elastic-rrf-rerank-demo

Hybrid search on Elasticsearch: BM25 + kNN vectors, fused with RRF, then reranked. Built for running retrieval experiments and comparing systems (BM25, kNN, hybrid, hybrid+rerank).

## Related Blog Post

For a deep dive into the architecture, technical tradeoffs, and experimental results of this project, check out the full article on Substack.
**[Hybrid Search Done Right on Elastic: BM25 + Vector kNN + RRF + Reranking](https://arnabbir.substack.com/p/hybrid-search-done-right-on-elastic)**

*This project was submitted as part of the [Elastic Blogathon](https://events.elastic.co/blogathon?utm_source=invite&utm_medium=email&utm_campaign=hackerearth-rm).*

## Setup

```bash
cp .env.example .env
# edit .env if needed — ES_URL, EMBEDDING_MODEL, RERANK_MODE
docker compose up -d
pip install -r requirements.txt
```

Data lives in `dataset/data_lite/` by default: `documents.jsonl`, `queries.jsonl`, `qrels.tsv`. Use `DATA_DIR=./dataset/data_large` for the larger dataset.

## Pipeline

| Step | Script | Description |
|------|--------|-------------|
| 1 | `01_prepare_dataset.py` | Validate data (documents.jsonl, queries.jsonl, qrels.tsv) |
| 2 | `02_create_index.py` | Create ES index (dense_vector + text) |
| 3 | `03_index_documents.py` | Embed docs, bulk index |
| 4 | `04_run_queries.py` | Run queries, save runs + latency |
| 5 | `05_compute_metrics.py` | Compute metrics from runs |
| 6 | `06_visualize_all.py` | Generate charts |
| 7 | `07_create_run_readme.py` | Create README with run specs |

```bash
python scripts/01_prepare_dataset.py
python scripts/02_create_index.py
python scripts/03_index_documents.py
python scripts/04_run_queries.py
python scripts/05_compute_metrics.py
python scripts/06_visualize_all.py
python scripts/07_create_run_readme.py
```

`04_run_queries.py` runs incrementally and saves after each query so you can stop and resume.

Charts end up in `reports/{data_dir}_{hash}/`.

## What it does

- **BM25** on `title^2` and `body`
- **kNN** on 384-d embeddings (all-MiniLM-L6-v2 by default)
- **RRF** fusion of the two ranked lists (k=60)
- **Rerank** top 50 with a cross-encoder (ms-marco-MiniLM-L-6-v2 locally, or HTTP endpoint via `RERANK_MODE=http`)

Metrics: NDCG@10, MRR@10, Recall@50.

## Config

| Env | Default |
|-----|---------|
| `DATA_DIR` | `./dataset/data_lite` |
| `ES_URL` | `http://localhost:9200` |
| `INDEX_NAME` | `blogathon-rrf-demo` |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` |
| `EMBED_DIMS` | `384` |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `RERANK_MODE` | `local` |
| `RETRIEVAL_TOPN` | `50` |
| `KNN_NUM_CANDIDATES` | `100` |
| `RRF_K` | `60` |
| `RERANK_TOPN` | `50` |

For `RERANK_MODE=http`, set `RERANK_HTTP_URL` and optionally `RERANK_HTTP_AUTH_HEADER`.

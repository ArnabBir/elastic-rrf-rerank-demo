from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm
from elasticsearch import helpers
from src.es_client import get_client, get_config
from src.utils import read_jsonl, get_data_dir
from src.embed import embed_texts

def main():
    es = get_client()
    cfg = get_config()
    data_dir = get_data_dir()
    docs = read_jsonl(data_dir / "documents.jsonl")
    texts = [f"{d['title']}\n\n{d['body']}" for d in docs]
    vectors = embed_texts(texts)
    actions = []
    for d, v in zip(docs, vectors):
        actions.append({
            "_index": cfg.index_name,
            "_id": d["doc_id"],
            "_source": {
                **d,
                "embedding": v,
            }
        })

    helpers.bulk(es, actions, refresh=True)
    print(f"Indexed {len(actions)} documents into {cfg.index_name}")

if __name__ == "__main__":
    main()

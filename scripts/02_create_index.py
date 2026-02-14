from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.es_client import get_client, get_config
from src.config import get_embed_dims
from elasticsearch import NotFoundError

def main():
    es = get_client()
    cfg = get_config()
    index_name = cfg.index_name

    try:
        es.indices.delete(index=index_name)
        print(f"Deleted existing index: {index_name}")
    except NotFoundError:
        pass

    mapping = {
        "mappings": {
            "properties": {
                "doc_id": {"type": "keyword"},
                "title": {"type": "text"},
                "body": {"type": "text"},
                "tags": {"type": "keyword"},
                "source": {"type": "keyword"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": get_embed_dims(),
                    "index": True,
                    "similarity": "cosine",
                },
            }
        }
    }

    es.indices.create(index=index_name, **mapping)
    print(f"Created index: {index_name}")

if __name__ == "__main__":
    main()

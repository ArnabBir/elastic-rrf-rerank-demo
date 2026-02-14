import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv()

@dataclass
class ESConfig:
    url: str
    api_key: Optional[str]
    index_name: str

def get_config() -> ESConfig:
    url = os.getenv("ES_URL", "http://localhost:9200")
    api_key = os.getenv("ES_API_KEY") or None
    index_name = os.getenv("INDEX_NAME", "blogathon-rrf-demo")
    return ESConfig(url=url, api_key=api_key, index_name=index_name)

def get_client() -> Elasticsearch:
    cfg = get_config()
    if cfg.api_key:
        return Elasticsearch(cfg.url, api_key=cfg.api_key)
    return Elasticsearch(cfg.url)

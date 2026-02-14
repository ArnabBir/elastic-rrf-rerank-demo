from functools import lru_cache
from typing import List

from src.config import get_embedding_model


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(get_embedding_model())

def embed_texts(texts: List[str]) -> List[List[float]]:
    model = _get_model()
    vectors = model.encode(texts, normalize_embeddings=True).tolist()
    return vectors

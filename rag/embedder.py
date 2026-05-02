import logging
from typing import List
from sentence_transformers import SentenceTransformer

from config import config

logger = logging.getLogger(__name__)


class Embedder:

    def __init__(self) -> None:
        logger.info(f"Загрузка модели эмбеддингов: {config.rag.embedding_model}")
        self._model = SentenceTransformer(config.rag.embedding_model)
        logger.info("Модель эмбеддингов загружена")

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return vectors.tolist()

    def embed_single(self, text: str) -> List[float]:
        return self.embed([text])[0]

embedder = Embedder()

import logging
from typing import List, Optional
from dataclasses import dataclass

from config import config
from rag.embedder import embedder
from rag.indexer import indexer

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDoc:
    title: str
    category: str
    hint: str
    example_error: str
    example_fix: str
    similarity: float

    def to_context_string(self) -> str:
        lines = [
            f"[{self.category}] {self.title}",
            f"Типичная ошибка: {self.hint}",
        ]
        if self.example_error:
            lines.append(f"Пример ошибки:\n{self.example_error}")
        if self.example_fix:
            lines.append(f"Исправление:\n{self.example_fix}")
        return "\n".join(lines)


class RAGRetriever:
    def __init__(self) -> None:
        self._collection = indexer._collection

    def retrieve(
        self,
        query: str,
        code: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[RetrievedDoc]:
        if top_k is None:
            top_k = config.rag.top_k

        search_text = query
        if code:
            code_snippet = code[:500].strip()
            search_text = f"{query}\n\nКод:\n{code_snippet}"

        query_embedding = embedder.embed_single(search_text)

        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, max(1, indexer.document_count)),
                include=["metadatas", "distances", "documents"],
            )
        except Exception as e:
            logger.error(f"Ошибка поиска в ChromaDB: {e}")
            return []

        retrieved = []
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for meta, distance in zip(metadatas, distances):
            similarity = 1.0 - distance

            if similarity < config.rag.similarity_threshold:
                logger.debug(f"Пропуск документа с similarity={similarity:.3f} (ниже порога)")
                continue

            doc = RetrievedDoc(
                title=meta.get("title", ""),
                category=meta.get("category", ""),
                hint=meta.get("hint", ""),
                example_error=meta.get("example_error", ""),
                example_fix=meta.get("example_fix", ""),
                similarity=similarity,
            )
            retrieved.append(doc)
            logger.debug(f"Найден документ: '{doc.title}' (similarity={similarity:.3f})")

        logger.info(f"RAG: найдено {len(retrieved)} релевантных документов из {top_k} запрошенных")
        return retrieved

    def format_context(self, docs: List[RetrievedDoc]) -> str:
        if not docs:
            return ""

        sections = []
        for i, doc in enumerate(docs, 1):
            sections.append(f"--- Подсказка {i} (релевантность: {doc.similarity:.0%}) ---")
            sections.append(doc.to_context_string())

        return "\n\n".join(sections)


retriever = RAGRetriever()

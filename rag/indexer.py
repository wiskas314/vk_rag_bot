

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings

from config import config
from rag.embedder import embedder

logger = logging.getLogger(__name__)


KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "knowledge_base" / "informatics_tasks.json"


def _doc_to_text(doc: Dict[str, Any]) -> str:
    parts = [
        doc.get("title", ""),
        doc.get("description", ""),
        doc.get("common_mistake", ""),
        " ".join(doc.get("keywords", [])),
        doc.get("hint", ""),
    ]
    return "\n".join(p for p in parts if p)


def _doc_to_metadata(doc: Dict[str, Any]) -> Dict[str, str]:
    return {
        "category": doc.get("category", ""),
        "title": doc.get("title", ""),
        "hint": doc.get("hint", ""),
        "example_error": doc.get("example_error", ""),
        "example_fix": doc.get("example_fix", ""),
    }


class KnowledgeIndexer:

    def __init__(self) -> None:
        self._chroma = chromadb.PersistentClient(
            path=config.rag.chroma_db_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._chroma.get_or_create_collection(
            name=config.rag.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def document_count(self) -> int:
        return self._collection.count()

    def is_populated(self) -> bool:
        return self._collection.count() > 0

    def index_knowledge_base(self, force: bool = False) -> int:
        if self.is_populated() and not force:
            logger.info(f"База знаний уже заполнена ({self.document_count} документов). Пропуск.")
            return self.document_count

        if not KNOWLEDGE_BASE_PATH.exists():
            raise FileNotFoundError(f"Файл базы знаний не найден: {KNOWLEDGE_BASE_PATH}")

        with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
            docs: List[Dict[str, Any]] = json.load(f)

        if force and self.is_populated():
            logger.info("Очистка старого индекса...")
            # Удаляем все документы
            all_ids = self._collection.get()["ids"]
            if all_ids:
                self._collection.delete(ids=all_ids)

        ids = []
        texts = []
        metadatas = []

        for doc in docs:
            doc_id = doc.get("id", f"doc_{len(ids)}")
            text = _doc_to_text(doc)
            meta = _doc_to_metadata(doc)

            ids.append(doc_id)
            texts.append(text)
            metadatas.append(meta)

        logger.info(f"Создание эмбеддингов для {len(texts)} документов...")
        embeddings = embedder.embed(texts)

        self._collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        count = self._collection.count()
        logger.info(f"Проиндексировано документов: {count}")
        return count

    def add_document(self, doc: Dict[str, Any]) -> None:
        doc_id = doc.get("id", f"dynamic_{self._collection.count()}")
        text = _doc_to_text(doc)
        embedding = embedder.embed_single(text)

        self._collection.upsert(
            ids=[doc_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[_doc_to_metadata(doc)],
        )
        logger.info(f"Добавлен документ: {doc_id}")


indexer = KnowledgeIndexer()

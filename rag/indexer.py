import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings

from config import config
from rag.embedder import embedder

logger = logging.getLogger(__name__)

KNOWLEDGE_BASE_DIR = Path(__file__).parent.parent / "knowledge_base"


# ─────────────────────────────────────────────────────────────────────────────
# Конвертеры: JSON-документ → (id, текст для эмбеддинга, метаданные)
# ─────────────────────────────────────────────────────────────────────────────

def _simple_doc_to_record(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Конвертирует простой документ в один запись для ChromaDB."""
    text = "\n".join(filter(None, [
        doc.get("title", ""),
        doc.get("description", ""),
        doc.get("common_mistake", ""),
        " ".join(doc.get("keywords", [])),
        doc.get("hint", ""),
        doc.get("example_error", ""),
        doc.get("example_fix", ""),
    ]))
    return [{
        "id": doc.get("id", "unknown"),
        "text": text,
        "metadata": {
            "category": doc.get("category", ""),
            "title": doc.get("title", ""),
            "hint": doc.get("hint", ""),
            "example_error": doc.get("example_error", ""),
            "example_fix": doc.get("example_fix", ""),
            "format": "simple",
        }
    }]


def _task_template_to_record(
        parent: Dict[str, Any],
        tmpl: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Конвертирует один шаблон из расширенного документа в запись ChromaDB.
    Шаблон — это конкретный вариант задачи с условием, кодом и подсказками.
    """
    # Собираем весь текст шаблона — он будет векторизован
    hints_text = "\n".join(tmpl.get("hints_by_error", {}).values())
    errors_text = "\n".join(tmpl.get("common_student_errors", []))

    text = "\n".join(filter(None, [
        parent.get("title", ""),
        parent.get("description", ""),
        " ".join(parent.get("keywords", [])),
        tmpl.get("condition", ""),
        tmpl.get("solution_code", ""),
        tmpl.get("solution_explanation", ""),
        errors_text,
        hints_text,
        parent.get("hint", ""),
    ]))

    # Общие подсказки родительского документа
    general_hints = parent.get("general_hints", {})
    general_hints_text = "\n".join(general_hints.values())

    # Все hints_by_error объединяем в строку для метаданных
    hints_by_error_flat = "\n---\n".join(
        f"[{key}]: {val}"
        for key, val in tmpl.get("hints_by_error", {}).items()
    )

    return {
        "id": f"{parent.get('id', 'doc')}_{tmpl.get('template_id', 'tmpl')}",
        "text": text,
        "metadata": {
            "category": parent.get("category", ""),
            "title": parent.get("title", ""),
            "template_id": tmpl.get("template_id", ""),
            "condition": tmpl.get("condition", "")[:500],  # ChromaDB лимит
            "answer": str(tmpl.get("answer", "")),
            "solution_code": tmpl.get("solution_code", "")[:800],
            "solution_explanation": tmpl.get("solution_explanation", "")[:500],
            "common_errors": "\n".join(tmpl.get("common_student_errors", []))[
                :500],
            "hints_by_error": hints_by_error_flat[:1000],
            "general_hints": general_hints_text[:500],
            # Родительские подсказка и примеры
            "hint": parent.get("hint", ""),
            "example_error": parent.get("example_error", ""),
            "example_fix": parent.get("example_fix", ""),
            "format": "task_template",
        }
    }


def _extended_doc_to_records(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Разворачивает расширенный документ в список записей (по одной на шаблон)."""
    templates = doc.get("task_templates", [])
    if not templates:
        # Нет шаблонов — конвертируем как простой документ
        return _simple_doc_to_record(doc)
    return [_task_template_to_record(doc, tmpl) for tmpl in templates]


def _doc_to_records(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Определяет формат документа и конвертирует в записи ChromaDB."""
    if "task_templates" in doc:
        return _extended_doc_to_records(doc)
    return _simple_doc_to_record(doc)


# ─────────────────────────────────────────────────────────────────────────────
# Индексатор
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeIndexer:
    """Управляет индексом базы знаний в ChromaDB."""

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
        """
        Загружает все JSON-файлы из папки knowledge_base/ в ChromaDB.
        Поддерживает оба формата (простой и расширенный с шаблонами).

        Args:
            force: Если True — удаляет старые данные и индексирует заново.

        Returns:
            Количество записей в коллекции.
        """
        if self.is_populated() and not force:
            logger.info(
                f"База знаний уже заполнена ({self.document_count} записей). Пропуск.")
            return self.document_count

        if not KNOWLEDGE_BASE_DIR.exists():
            raise FileNotFoundError(
                f"Папка базы знаний не найдена: {KNOWLEDGE_BASE_DIR}")

        json_files = list(KNOWLEDGE_BASE_DIR.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(
                f"JSON файлы не найдены в {KNOWLEDGE_BASE_DIR}")

        if force and self.is_populated():
            logger.info("Очистка старого индекса...")
            all_ids = self._collection.get()["ids"]
            if all_ids:
                self._collection.delete(ids=all_ids)

        all_records: List[Dict[str, Any]] = []

        for json_file in json_files:
            logger.info(f"Загрузка файла: {json_file.name}")
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    docs = json.load(f)
                for doc in docs:
                    records = _doc_to_records(doc)
                    all_records.extend(records)
                logger.info(f"  → {json_file.name}: {len(docs)} документов")
            except Exception as e:
                logger.error(f"Ошибка загрузки {json_file.name}: {e}")

        if not all_records:
            logger.warning("Нет записей для индексации")
            return 0

        # Батчевая векторизация
        logger.info(f"Создание эмбеддингов для {len(all_records)} записей...")
        texts = [r["text"] for r in all_records]
        embeddings = embedder.embed(texts)

        # Вставка в ChromaDB
        self._collection.upsert(
            ids=[r["id"] for r in all_records],
            documents=texts,
            embeddings=embeddings,
            metadatas=[r["metadata"] for r in all_records],
        )

        count = self._collection.count()
        logger.info(f"Проиндексировано записей: {count}")
        return count

    def add_document(self, doc: Dict[str, Any]) -> int:
        """Добавляет документ (любого формата) в индекс без перестройки."""
        records = _doc_to_records(doc)
        texts = [r["text"] for r in records]
        embeddings = embedder.embed(texts)
        self._collection.upsert(
            ids=[r["id"] for r in records],
            documents=texts,
            embeddings=embeddings,
            metadatas=[r["metadata"] for r in records],
        )
        logger.info(
            f"Добавлено {len(records)} записей из документа '{doc.get('id')}'")
        return len(records)


# Синглтон
indexer = KnowledgeIndexer()
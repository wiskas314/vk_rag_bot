"""
Конфигурация бота. Все чувствительные данные берутся из переменных окружения.
Создайте файл .env в корне проекта или задайте переменные напрямую.
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class VKConfig:
    # Токен сообщества VK (Управление -> Работа с API -> Ключи доступа)
    token: str = field(default_factory=lambda: os.getenv("VK_TOKEN", ""))
    # ID группы VK (без минуса)
    group_id: int = field(default_factory=lambda: int(os.getenv("VK_GROUP_ID", "0")))
    # Версия VK API
    api_version: str = "5.199"
    # Максимальный размер принимаемого файла (байты), 512 КБ
    max_file_size: int = 512 * 1024


@dataclass
class OllamaConfig:
    # Адрес Ollama сервера
    base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_URL", "http://localhost:11434"))
    # Лёгкая модель — хорошо понимает код, быстро работает
    # Альтернативы: deepseek-coder:1.3b, phi3:mini, codellama:7b
    model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen2.5-coder:3b"))
    # Таймаут запроса к Ollama (сек)
    timeout: int = 120
    # Параметры генерации
    temperature: float = 0.2   # низкая — более детерминированные ответы для учебных подсказок
    num_predict: int = 512  # максимум токенов в ответе


@dataclass
class RAGConfig:
    # Путь к базе векторов ChromaDB
    chroma_db_path: str = field(default_factory=lambda: os.getenv("CHROMA_PATH", "./chroma_db"))
    # Название коллекции
    collection_name: str = "informatics_knowledge"
    # Модель для эмбеддингов (multilingual, понимает русский)
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # Сколько релевантных документов брать из базы
    top_k: int = 3
    # Минимальный порог схожести (0-1), ниже — документ игнорируется
    similarity_threshold: float = 0.35


@dataclass
class BotConfig:
    vk: VKConfig = field(default_factory=VKConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)

    # Поддерживаемые расширения файлов с кодом
    supported_extensions: tuple = (".py", ".txt", ".pas", ".cpp", ".c", ".java", ".js", ".cs")
    # Максимальная длина кода, которую примем (символы)
    max_code_length: int = 10_000
    # Логирование
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_file: str = "bot.log"


# Глобальный конфиг — импортируется во всех модулях
config = BotConfig()


def validate_config() -> None:
    """Проверяет что обязательные поля заполнены."""
    errors = []
    if not config.vk.token:
        errors.append("VK_TOKEN не задан")
    if config.vk.group_id == 0:
        errors.append("VK_GROUP_ID не задан")
    if errors:
        raise EnvironmentError(
            "Ошибки конфигурации:\n" + "\n".join(f"  - {e}" for e in errors)
        )

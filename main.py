import asyncio
import logging
import sys
from logging.handlers import RotatingFileHandler

import vk_api
from vk_api.longpoll import VkLongPoll, VkEventType

from config import config, validate_config
from bot.handler import MessageHandler
from rag.indexer import indexer
from llm.ollama_client import ollama_client


def setup_logging() -> None:
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)

    file_handler = RotatingFileHandler(
        config.log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(getattr(logging, config.log_level, logging.INFO))
    root.addHandler(console)
    root.addHandler(file_handler)

    for noisy in ("urllib3", "httpcore", "httpx", "hpack", "chromadb.telemetry"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def startup_checks() -> None:
    logger = logging.getLogger("startup")


    logger.info("Проверка конфигурации...")
    validate_config()
    logger.info("✓ Конфигурация OK")


    logger.info("Проверка Ollama...")
    loop = asyncio.get_event_loop()

    if not loop.run_until_complete(ollama_client.is_available()):
        logger.error(
            f"✗ Ollama недоступна по адресу {config.ollama.base_url}\n"
            "  Запусти: ollama serve"
        )
        sys.exit(1)

    if not loop.run_until_complete(ollama_client.model_exists()):
        logger.warning(
            f"⚠ Модель '{config.ollama.model}' не найдена.\n"
            f"  Загрузи её: ollama pull {config.ollama.model}\n"
            "  Бот продолжит работу, но модель нужно загрузить."
        )
    else:
        logger.info(f"✓ Модель '{config.ollama.model}' готова")


    logger.info("Инициализация базы знаний...")
    count = indexer.index_knowledge_base(force=False)
    logger.info(f"✓ База знаний: {count} документов проиндексировано")


def run_longpoll() -> None:
    logger = logging.getLogger("main")

    logger.info("Подключение к VK API...")
    vk_session = vk_api.VkApi(token=config.vk.token)

    try:

        bot_info = vk_session.get_api().groups.getById(group_id=config.vk.group_id)
        group_name = bot_info[0].get("name", "неизвестная группа")
        logger.info(f"✓ Подключено к группе: {group_name} (id={config.vk.group_id})")
    except vk_api.exceptions.ApiError as e:
        logger.error(f"✗ Ошибка авторизации VK: {e}")
        sys.exit(1)

    handler = MessageHandler(vk_session)
    longpoll = VkLongPoll(vk_session, group_id=config.vk.group_id)

    logger.info("🚀 Бот запущен, ожидаю сообщений...")
    logger.info("   Нажми Ctrl+C для остановки")

    try:
        for event in longpoll.listen():
            if event.type == VkEventType.MESSAGE_NEW and event.to_me:
                try:
                    handler.handle_event(event)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    logger.exception(f"Ошибка обработки события: {e}")

    except KeyboardInterrupt:
        logger.info("Остановка бота...")
    finally:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(ollama_client.close())
        logger.info("Бот остановлен.")


def main() -> None:
    setup_logging()
    logger = logging.getLogger("main")

    logger.info("=" * 50)
    logger.info("  VK Бот — Помощник по информатике (RAG + Ollama)")
    logger.info("=" * 50)

    startup_checks()
    run_longpoll()


if __name__ == "__main__":
    main()

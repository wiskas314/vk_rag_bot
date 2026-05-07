"""
Обработчик сообщений VK бота.

Парсер умеет читать компоненты сообщения в ЛЮБОМ порядке:
  - вопрос → условие → код
  - код → вопрос → условие
  - условие → код → вопрос
  - только код, только условие+код, и т.д.

Алгоритм сегментный: сообщение разбивается на чередующиеся сегменты
(текст / блок кода), каждый сегмент классифицируется независимо,
результаты объединяются вне зависимости от порядка.
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional, Tuple

import httpx
import vk_api
from vk_api.longpoll import VkEventType

from config import config
from llm.ollama_client import ollama_client
from rag.retriever import retriever

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Сообщения бота
# ─────────────────────────────────────────────────────────────────────────────
MSG_WELCOME = (
    "👋 Привет! Я помогаю разобраться с ошибками в коде на Python.\n\n"
    "Пиши в любом порядке — я разберусь:\n\n"
    "Вариант 1 — только код:\n"
    "```python\ndef f(n): ...\n```\n\n"
    "Вариант 2 — вопрос + код:\n"
    "Почему выдаёт ошибку?\n"
    "```python\nкод\n```\n\n"
    "Вариант 3 — всё вместе (любой порядок):\n"
    "```python\nкод\n```\n"
    "F(n)=3 при n<10, F(n)=(n+4)*F(n-5) при n≥10\n"
    "Не понимаю почему падает программа\n\n"
    "Также можно прикрепить .py файл, а условие и вопрос написать рядом.\n\n"
    "Я дам подсказку — не готовый ответ, а направление 🎯"
)
MSG_TOO_LONG = "⚠️ Код слишком длинный (максимум {} символов). Пришли только проблемный фрагмент."
MSG_NOT_PY_FILE = "⚠️ Я принимаю только .py файлы. Убедись, что сохранил код как Python-файл."
MSG_NOT_PYTHON = (
    "🤔 Похоже, в сообщении нет Python-кода.\n\n"
    "Я работаю только с Python. Пришли код прямо в сообщении "
    "или прикрепи .py файл. Условие и вопрос можно написать в любом месте 🐍"
)
MSG_NO_CODE = (
    "🤔 Я не нашёл Python-код в твоём сообщении.\n\n"
    "Пришли код прямо в сообщении или прикрепи .py файл. "
    "Условие задачи и вопрос пиши рядом — я всё пойму в любом порядке."
)
MSG_PROCESSING = "🔍 Анализирую код и задачу, подожди немного..."
MSG_OLLAMA_UNAVAILABLE = "😔 Сервис анализа временно недоступен. Попробуй через несколько минут."

PYTHON_EXT = ".py"

# ─────────────────────────────────────────────────────────────────────────────
# Регулярные выражения — классификация
# ─────────────────────────────────────────────────────────────────────────────

# Блок кода: ```python ... ``` или ``` ... ```
_CODE_BLOCK_RE = re.compile(
    r"```(?:python|py)?\s*\n?(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

# Python-признаки в строке
_PYTHON_LINE_RE = re.compile(
    r"""
      \bdef\s+\w+\s*\(          # def func(
    | \bimport\s+\w+             # import os
    | \bfrom\s+\w+\s+import      # from x import
    | \bprint\s*\(               # print(
    | \binput\s*\(               # input(
    | \bif\b.+:\s*$              # if ...:
    | \bfor\s+\w+\s+in\b         # for x in
    | \bwhile\b.+:\s*$           # while ...:
    | \bclass\s+\w+[\s:(]        # class Foo
    | \belif\b.+:                # elif
    | \bTrue\b|\bFalse\b|\bNone\b
    | \blambda\b
    | \brange\s*\(
    | \blen\s*\(
    | @\w+                       # декоратор
    | \blru_cache\b
    | \breturn\b
    | \bsetrecursionlimit\b
    """,
    re.VERBOSE | re.MULTILINE,
)

# Маркеры чужих языков (Pascal, C++, Java, C#)
_NON_PYTHON_RE = re.compile(
    r"""
      \bbegin\b.*\bend\b
    | \bwriteln\b | \breadln\b
    | \bvar\b\s+\w+\s*:
    | \bprogram\s+\w+\s*;
    | \#include\s*<
    | \bint\s+main\s*\(
    | \bcout\s*<< | \bcin\s*>>
    | \bpublic\s+class\b
    | \bSystem\.out\b
    | \busing\s+namespace\b
    | \bConsole\.Write\b
    """,
    re.VERBOSE | re.IGNORECASE | re.DOTALL,
)

# Признаки вопроса
_QUESTION_RE = re.compile(
    r"""
      \?
    | \bпочему\b | \bзачем\b
    | \bошибк                    # ошибка/ошибку
    | \bне\s+работ               # не работает
    | \bне\s+понима              # не понимаю
    | \bне\s+знаю\b
    | \bчто\s+не\s+так\b
    | \bв\s+чём\s+(проблема|ошибка)
    | \bне\s+могу\s+понять\b
    | \bне\s+получается\b
    | \bвыдаёт\b
    | \bупал[оа]?\b
    | \bhelp\b | \bcrash\b | \bfail\b | \berror\b
    | \bпомогите\b | \bпомоги\b
    | \bпроверьте\b | \bпроверь\b
    | \bподскажи\b | \bподскажите\b
    | \bнеправильн
    | \bневерн
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Признаки условия задачи
_TASK_RE = re.compile(
    r"""
      \bдан[оа]?\b | \bнайти\b | \bнайдите\b
    | \bвычисли | \bвычислите\b
    | \bзадач | \bалгоритм\b
    | \bфункци                   # функция/функцию
    | \bзначени                  # значение/значению
    | \bусловие\b | \bтребуется\b | \bнеобходимо\b
    | \bопределить\b | \bопредели\b
    | \bрассмотрим\b
    | \bчему\s+равно\b
    | \bзапиши\b | \bзапишите\b
    | F\s*\(                     # F(n), F(257...)
    | ≥|≤|×|→|⩾|⩽
    | \bпри\s+n\b
    | \bесли\s+n\b
    | \bn\s*[<>≤≥=]\s*\d
    | \bсоотношени               # соотношения
    | \bрекуррент                # рекуррентное
    | \bвыражени                 # выражение
    | \bцелую?\s+часть\b
    | ЕГЭ|ОГЭ
    """,
    re.VERBOSE | re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Типы сегментов
# ─────────────────────────────────────────────────────────────────────────────

class SegKind(Enum):
    CODE_BLOCK = auto()  # ``` блок — однозначно код
    CODE_RAW = auto()  # строки-код без ```
    TASK = auto()  # условие задачи
    QUESTION = auto()  # вопрос ученика
    UNKNOWN = auto()  # неопределённые строки


@dataclass
class Segment:
    kind: SegKind
    text: str


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────────────────────────────────────

def _classify_line(line: str) -> SegKind:
    """Классифицирует одну строку текста."""
    s = line.strip()
    if not s:
        return SegKind.UNKNOWN

    # Сначала проверяем Python — приоритет
    if _PYTHON_LINE_RE.search(line):
        return SegKind.CODE_RAW

    # Вопрос перед условием — чтобы «почему не работает функция F(n)?»
    # не попал в TASK
    if _QUESTION_RE.search(s) and not _TASK_RE.search(s):
        return SegKind.QUESTION

    if _TASK_RE.search(s):
        return SegKind.TASK

    if _QUESTION_RE.search(s):
        return SegKind.QUESTION

    return SegKind.UNKNOWN


def _merge_consecutive(segments: List[Segment]) -> List[Segment]:
    """Объединяет соседние сегменты одного типа."""
    if not segments:
        return []
    merged = [segments[0]]
    for seg in segments[1:]:
        if seg.kind == merged[-1].kind:
            merged[-1].text = merged[-1].text + "\n" + seg.text
        else:
            merged.append(seg)
    return merged


def _expand_raw_code_context(segments: List[Segment]) -> List[Segment]:
    """
    Расширяет CODE_RAW блоки: строки с отступом рядом с кодом тоже считаются кодом.
    Это нужно для многострочных функций без явных Python-ключевых слов в каждой строке.
    """
    result = []
    for i, seg in enumerate(segments):
        if seg.kind != SegKind.UNKNOWN:
            result.append(seg)
            continue

        # UNKNOWN строка — проверяем соседей
        prev_is_code = i > 0 and segments[i - 1].kind in (SegKind.CODE_RAW,
                                                          SegKind.CODE_BLOCK)
        next_is_code = i < len(segments) - 1 and segments[i + 1].kind in (
            SegKind.CODE_RAW, SegKind.CODE_BLOCK)
        has_indent = seg.text != seg.text.lstrip()  # есть отступ

        if has_indent and (prev_is_code or next_is_code):
            result.append(Segment(SegKind.CODE_RAW, seg.text))
        else:
            result.append(seg)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Результат парсинга
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ParsedMessage:
    code: str = ""
    task_description: str = ""
    question: str = ""
    has_code_block: bool = False

    @property
    def is_valid(self) -> bool:
        return bool(self.code.strip())

    def debug_summary(self) -> str:
        parts = []
        if self.has_code_block:
            parts.append("```-блок")
        elif self.code:
            parts.append("сырой-код")
        if self.task_description:
            parts.append("условие")
        if self.question:
            parts.append("вопрос")
        return " | ".join(parts) if parts else "пусто"


# ─────────────────────────────────────────────────────────────────────────────
# ГЛАВНЫЙ ПАРСЕР
# ─────────────────────────────────────────────────────────────────────────────

def parse_message(text: str) -> ParsedMessage:
    """
    Разбирает сообщение в ЛЮБОМ порядке компонентов.

    Шаги:
    1. Извлекаем ``` блоки → они однозначно код (CODE_BLOCK).
    2. Оставшийся текст разбиваем на строки и классифицируем каждую.
    3. Объединяем соседние сегменты одного типа.
    4. Группируем: всё CODE* → код, всё TASK → условие, QUESTION → вопрос.
    5. UNKNOWN строки — относим к задаче или к вопросу по контексту.
    """
    result = ParsedMessage()
    segments: List[Segment] = []

    # ── Шаг 1: вычленяем ``` блоки ────────────────────────────────────────
    last_end = 0
    for m in _CODE_BLOCK_RE.finditer(text):
        before = text[last_end:m.start()]
        if before.strip():
            segments.append(Segment(SegKind.UNKNOWN, before))  # временный тип
        code_content = m.group(1).strip()
        if code_content:
            segments.append(Segment(SegKind.CODE_BLOCK, code_content))
        last_end = m.end()

    after = text[last_end:]
    if after.strip():
        segments.append(Segment(SegKind.UNKNOWN, after))

    # ── Шаг 2: классифицируем строки внутри UNKNOWN сегментов ────────────
    classified: List[Segment] = []
    for seg in segments:
        if seg.kind == SegKind.CODE_BLOCK:
            classified.append(seg)
            continue

        # Разбиваем UNKNOWN на строки и классифицируем каждую
        for line in seg.text.splitlines():
            if not line.strip():
                continue
            kind = _classify_line(line)
            classified.append(Segment(kind, line))

    # ── Шаг 3: расширяем контекст сырого кода ────────────────────────────
    classified = _expand_raw_code_context(classified)

    # ── Шаг 4: объединяем соседние одинаковые сегменты ───────────────────
    merged = _merge_consecutive(classified)

    # ── Шаг 5: собираем финальный результат ──────────────────────────────
    code_parts = []
    task_parts = []
    question_parts = []
    unknown_parts = []

    has_code_block = False

    for seg in merged:
        if seg.kind == SegKind.CODE_BLOCK:
            code_parts.append(seg.text)
            has_code_block = True
        elif seg.kind == SegKind.CODE_RAW:
            code_parts.append(seg.text)
        elif seg.kind == SegKind.TASK:
            task_parts.append(seg.text)
        elif seg.kind == SegKind.QUESTION:
            question_parts.append(seg.text)
        else:  # UNKNOWN
            unknown_parts.append(seg.text)

    # UNKNOWN: если есть хоть какой-то код — относим к условию,
    # иначе — к вопросу (скорее всего это «не работает» без явного знака вопроса)
    if unknown_parts:
        if code_parts or has_code_block:
            task_parts.extend(unknown_parts)
        else:
            question_parts.extend(unknown_parts)

    result.code = "\n\n".join(code_parts).strip()
    result.task_description = "\n".join(task_parts).strip()
    result.question = "\n".join(question_parts).strip()
    result.has_code_block = has_code_block

    # ── Шаг 6: фолбэк — если код не найден, пробуем весь текст как код ──
    if not result.code and _PYTHON_LINE_RE.search(text):
        result.code = text.strip()
        result.question = "В чём ошибка в этом коде?"
        logger.debug("Парсер [фолбэк]: весь текст воспринят как Python-код")
        return result

    if not result.question:
        result.question = "В чём ошибка в этом коде?"

    logger.debug(
        f"Парсер: {result.debug_summary()} | "
        f"код={len(result.code)}с | "
        f"задача={len(result.task_description)}с | "
        f"вопрос={len(result.question)}с"
    )
    return result


def is_non_python_code(code: str) -> bool:
    """True если код содержит явные маркеры другого языка."""
    return bool(_NON_PYTHON_RE.search(code))


# ─────────────────────────────────────────────────────────────────────────────
# Работа с файлами
# ─────────────────────────────────────────────────────────────────────────────

async def download_file(url: str) -> Optional[bytes]:
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            if len(resp.content) > config.vk.max_file_size:
                logger.warning(f"Файл слишком большой: {len(resp.content)} байт")
                return None
            return resp.content
    except Exception as e:
        logger.error(f"Ошибка скачивания {url}: {e}")
        return None


def parse_attachment_code(
        vk_session: vk_api.VkApi,
        message: dict,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Извлекает код из прикреплённого .py файла. Возвращает (код, имя, ошибка)."""
    for att in message.get("attachments", []):
        if att.get("type") != "doc":
            continue
        doc = att.get("doc", {})
        filename = doc.get("title", "file.txt")
        ext = Path(filename).suffix.lower()

        if ext != PYTHON_EXT:
            return None, None, MSG_NOT_PY_FILE

        url = doc.get("url", "")
        if not url:
            return None, None, "Не удалось получить ссылку на файл."

        content = asyncio.get_event_loop().run_until_complete(download_file(url))
        if content is None:
            return None, None, "Не удалось скачать файл (слишком большой или недоступен)."

        try:
            return content.decode("utf-8", errors="replace"), filename, None
        except Exception:
            return None, None, "Не удалось прочитать файл."

    return None, None, None


# ─────────────────────────────────────────────────────────────────────────────
# Обработчик VK событий
# ─────────────────────────────────────────────────────────────────────────────

class MessageHandler:

    def __init__(self, vk_session: vk_api.VkApi) -> None:
        self.vk = vk_session
        self.api = vk_session.get_api()

    def send_message(self, peer_id: int, text: str) -> None:
        import random
        try:
            self.api.messages.send(
                peer_id=peer_id,
                message=text,
                random_id=random.randint(0, 2**31),
            )
        except Exception as e:
            logger.error(f"Ошибка отправки {peer_id}: {e}")

    def handle_event(self, event) -> None:
        if event.type != VkEventType.MESSAGE_NEW or not event.to_me:
            return

        peer_id = event.peer_id
        raw_text = (event.text or "").strip()

        if raw_text.lower() in ("/start", "начало", "привет", "help", "помощь",
                                ""):
            self.send_message(peer_id, MSG_WELCOME)
            return

        # Получаем полное сообщение с вложениями
        try:
            msg_data = self.api.messages.getById(message_ids=event.message_id)
            message = msg_data["items"][0] if msg_data["items"] else {}
        except Exception as e:
            logger.error(f"Ошибка getById: {e}")
            message = {}

        # ── Файл-вложение ──────────────────────────────────────────────────
        file_code, filename, file_error = parse_attachment_code(self.vk, message)

        if file_error:
            self.send_message(peer_id, file_error)
            return

        if file_code:
            if is_non_python_code(file_code):
                self.send_message(peer_id, MSG_NOT_PYTHON)
                return

            # Текст сообщения разбираем как обычно — там может быть
            # условие, вопрос или и то и другое в любом порядке
            text_parsed = parse_message(
                raw_text) if raw_text else ParsedMessage()
            parsed = ParsedMessage(
                code=file_code,
                task_description=text_parsed.task_description,
                question=text_parsed.question or "В чём ошибка в этом коде?",
            )
            logger.info(f"Файл {filename}: {parsed.debug_summary()}")

        else:
            # ── Текстовое сообщение ────────────────────────────────────────
            if not raw_text:
                self.send_message(peer_id, MSG_NO_CODE)
                return

            parsed = parse_message(raw_text)

            if not parsed.is_valid:
                self.send_message(peer_id, MSG_NO_CODE)
                return

            if is_non_python_code(parsed.code):
                self.send_message(peer_id, MSG_NOT_PYTHON)
                return

            logger.info(f"Сообщение: {parsed.debug_summary()}")

        if len(parsed.code) > config.max_code_length:
            self.send_message(peer_id, MSG_TOO_LONG.format(config.max_code_length))
            return

        self.send_message(peer_id, MSG_PROCESSING)

        loop = asyncio.get_event_loop()
        hint = loop.run_until_complete(self._analyze(parsed))
        self.send_message(peer_id, f"💡 Подсказка:\n\n{hint}")
        logger.info(f"Ответ → {peer_id}")

    async def _analyze(self, parsed: ParsedMessage) -> str:
        if not await ollama_client.is_available():
            return MSG_OLLAMA_UNAVAILABLE

        rag_query = " ".join(filter(None, [
            parsed.task_description,
            parsed.question,
        ])) or parsed.code[:300]

        docs = retriever.retrieve(query=rag_query, code=parsed.code)
        rag_context = retriever.format_context(docs)

        if docs:
            logger.info("RAG: " + ", ".join(f"'{d.title}'" for d in docs))
        else:
            logger.info("RAG: совпадений не найдено")

        return await ollama_client.generate_hint(
            user_question=parsed.question,
            code=parsed.code,
            task_description=parsed.task_description,
            rag_context=rag_context,
        )
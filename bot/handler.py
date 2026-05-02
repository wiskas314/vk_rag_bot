import asyncio
import logging
import re
from pathlib import Path
from typing import Optional, Tuple
import httpx

import vk_api
from vk_api.longpoll import VkEventType

from config import config
from llm.ollama_client import ollama_client
from rag.retriever import retriever

logger = logging.getLogger(__name__)

MSG_WELCOME = (
    "👋 Привет! Я помогаю разобраться с ошибками в коде на Python.\n\n"
    "Как отправить задание:\n"
    "• Напиши код прямо в сообщении (можно обернуть в ```python ... ```)\n"
    "• Или прикрепи файл с расширением .py\n"
    "• Опиши, что должна делать программа и что идёт не так\n\n"
    "Я дам подсказку — не готовый ответ, а направление 🎯\n\n"
    "⚠️ Принимаю только код на Python."
)
MSG_TOO_LONG = (
    "⚠️ Код слишком длинный (максимум {} символов). "
    "Пришли только проблемный фрагмент."
)
MSG_NOT_PYTHON_FILE = (
    "⚠️ Я принимаю только файлы с расширением .py\n"
    "Убедись, что сохранил код как Python-файл."
)
MSG_NOT_PYTHON_CODE = (
    "🤔 Похоже, это не Python-код.\n\n"
    "Я работаю только с Python. Признаки других языков:\n"
    "• Pascal (begin/end, writeln) — не поддерживается\n"
    "• C/C++ (#include, int main) — не поддерживается\n"
    "• Java (public class) — не поддерживается\n\n"
    "Пришли код на Python 🐍"
)
MSG_NO_CODE = (
    "🤔 Я не нашёл Python-код в твоём сообщении.\n\n"
    "Пришли код прямо в сообщении или прикрепи .py файл."
)
MSG_PROCESSING = "🔍 Анализирую твой Python-код, подожди немного..."
MSG_OLLAMA_UNAVAILABLE = (
    "😔 Сервис анализа временно недоступен. Попробуй через несколько минут."
)

PYTHON_EXTENSION = ".py"

_PYTHON_POSITIVE = re.compile(
    r"""
    \bdef\s+\w+\s*\(        # def func(
    | \bimport\s+\w+         # import os
    | \bfrom\s+\w+\s+import  # from os import
    | \bprint\s*\(           # print(
    | \binput\s*\(           # input(
    | \bif\s+.+:             # if ...:
    | \bfor\s+\w+\s+in\s+   # for x in
    | \bwhile\s+.+:          # while ...:
    | \bclass\s+\w+          # class Foo
    | \belif\s+              # elif
    | \bTrue\b|\bFalse\b|\bNone\b  # Python literals
    | \blambda\s+            # lambda
    | \brange\s*\(           # range(
    | \blen\s*\(             # len(
    """,
    re.VERBOSE | re.IGNORECASE,
)

_NON_PYTHON_MARKERS = re.compile(
    r"""
    \bbegin\b.*\bend\b       # Pascal
    | \bwriteln\b            # Pascal
    | \breadln\b             # Pascal
    | \bvar\b\s+\w+\s*:     # Pascal var declaration
    | \bprogram\s+\w+\s*;   # Pascal program
    | \#include\s*<          # C/C++
    | \bint\s+main\s*\(      # C/C++
    | \bcout\s*<<            # C++
    | \bcin\s*>>             # C++
    | \bpublic\s+class\b     # Java
    | \bSystem\.out\.print   # Java
    | \bpublic\s+static\s+void\s+main  # Java
    | \busing\s+namespace    # C++
    | \bConsole\.Write       # C#
    """,
    re.VERBOSE | re.IGNORECASE | re.DOTALL,
)


def is_python_code(code: str, filename: Optional[str] = None) -> Tuple[bool, str]:

    if filename:
        ext = Path(filename).suffix.lower()
        if ext and ext != PYTHON_EXTENSION:
            return False, f"расширение файла {ext!r} не является .py"

    if _NON_PYTHON_MARKERS.search(code):
        return False, "обнаружены конструкции не-Python языка"

    if not _PYTHON_POSITIVE.search(code):
        return False, "не найдено ни одного Python-признака"

    return True, ""


def extract_code_from_message(text: str) -> Tuple[str, str]:

    code_block_pattern = r"```(?:python|py)?\n?(.*?)```"
    matches = re.findall(code_block_pattern, text, re.DOTALL | re.IGNORECASE)

    if matches:
        code = "\n".join(matches)
        question = re.sub(code_block_pattern, "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        return code.strip(), question or "В чём ошибка?"

    return text.strip(), text.strip()


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
        logger.error(f"Ошибка скачивания файла {url}: {e}")
        return None


def parse_attachment_code(
    vk: vk_api.VkApi, message: dict
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    attachments = message.get("attachments", [])

    for att in attachments:
        if att.get("type") != "doc":
            continue

        doc = att.get("doc", {})
        filename: str = doc.get("title", "file.txt")
        ext = Path(filename).suffix.lower()

        if ext != PYTHON_EXTENSION:
            return None, None, MSG_NOT_PYTHON_FILE

        url = doc.get("url", "")
        if not url:
            return None, None, "Не удалось получить ссылку на файл."

        content = asyncio.get_event_loop().run_until_complete(download_file(url))
        if content is None:
            return None, None, "Не удалось скачать файл (слишком большой или недоступен)."

        try:
            code = content.decode("utf-8", errors="replace")
            return code, filename, None
        except Exception:
            return None, None, "Не удалось прочитать файл."

    return None, None, None


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
            logger.error(f"Ошибка отправки сообщения {peer_id}: {e}")

    def handle_event(self, event) -> None:
        if event.type != VkEventType.MESSAGE_NEW or not event.to_me:
            return

        peer_id = event.peer_id
        text: str = event.text.strip() if event.text else ""


        if text.lower() in ("/start", "начало", "привет", "help", "помощь", ""):
            self.send_message(peer_id, MSG_WELCOME)
            return


        try:
            msg_data = self.api.messages.getById(message_ids=event.message_id)
            message = msg_data["items"][0] if msg_data["items"] else {}
        except Exception as e:
            logger.error(f"Ошибка получения сообщения: {e}")
            message = {}


        file_code, filename, file_error = parse_attachment_code(self.vk, message)

        if file_error:
            self.send_message(peer_id, file_error)
            return

        if file_code:
            code = file_code
            question = text or "В чём ошибка в этом коде?"


            ok, reason = is_python_code(code, filename)
            if not ok:
                logger.info(f"Отклонён файл {filename}: {reason}")
                self.send_message(peer_id, MSG_NOT_PYTHON_CODE)
                return

            logger.info(f"Принят файл: {filename}")

        else:

            code, question = extract_code_from_message(text)


            if len(code.strip()) < 10:
                self.send_message(peer_id, MSG_NO_CODE)
                return

            ok, reason = is_python_code(code)
            if not ok:
                logger.info(f"Отклонён код из сообщения: {reason}")
                if "не-Python" in reason:
                    self.send_message(peer_id, MSG_NOT_PYTHON_CODE)
                else:
                    self.send_message(peer_id, MSG_NO_CODE)
                return

        if len(code) > config.max_code_length:
            self.send_message(peer_id, MSG_TOO_LONG.format(config.max_code_length))
            return

        self.send_message(peer_id, MSG_PROCESSING)

        loop = asyncio.get_event_loop()
        hint = loop.run_until_complete(self._analyze_code(code, question))

        self.send_message(peer_id, f"💡 Подсказка:\n\n{hint}")
        logger.info(f"Ответ отправлен пользователю {peer_id}")

    async def _analyze_code(self, code: str, question: str) -> str:

        if not await ollama_client.is_available():
            logger.error("Ollama недоступна")
            return MSG_OLLAMA_UNAVAILABLE

        relevant_docs = retriever.retrieve(query=question, code=code)
        rag_context = retriever.format_context(relevant_docs)

        if relevant_docs:
            logger.info(
                f"RAG нашёл {len(relevant_docs)} документов: "
                + ", ".join(f"'{d.title}'" for d in relevant_docs)
            )
        else:
            logger.info("RAG: подходящих документов не найдено")

        return await ollama_client.generate_hint(
            user_question=question,
            code=code,
            rag_context=rag_context,
            language="Python",
        )
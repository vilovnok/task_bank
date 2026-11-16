# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import threading
import time
import json
import base64
import tempfile
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import requests
from loguru import logger
from PIL import Image

from utils.web_search_tools import (
    get_web_search_config,
    get_url_context_config,
    should_use_responses_endpoint,
)
from utils.model_endpoints import get_model_endpoint


TAGME_LLM_PROXY_URL: str = (os.getenv("TAGME_LLM_PROXY_URL") or "").strip() or \
                           "https://tagme.sberdevices.ru/x/ai/llm/v1/chat/completions"

TAGME_DIRECT_URL: str = (os.getenv("TAGME_DIRECT_URL") or "").strip() or \
                        "https://tagme.sberdevices.ru/ai/llm/v1/chat/completions"

TAGME_DIRECT_FILES_URL: str = (os.getenv("TAGME_DIRECT_FILES_URL") or "").strip() or \
                             "https://tagme.sberdevices.ru/ai/llm/v1/files"

TAGME_IFT_BASE_URL: str = (os.getenv("TAGME_IFT_BASE_URL") or "").strip().rstrip("/") or \
                          "https://tagme.ift.delta.sbrf.ru"

TAGME_IFT_URL: str = (os.getenv("TAGME_IFT_URL") or "").strip() or \
                     f"{TAGME_IFT_BASE_URL}/ai/llm/v1/chat/completions"

TAGME_IFT_FILES_URL: str = (os.getenv("TAGME_IFT_FILES_URL") or "").strip() or \
                           f"{TAGME_IFT_BASE_URL}/ai/llm/v1/files"

TAGME_EMBEDDINGS_PROXY_URL: str = (
                                          os.getenv("TAGME_EMBEDDINGS_PROXY_URL") or ""
                                  ).strip() or "https://tagme.sberdevices.ru/x/ai/llm/v1/embeddings"

DEFAULT_PROXY_CFG_PATH: str = os.path.expanduser(
    (os.getenv("TAGME_PROXY_CFG_PATH") or "").strip() or "~/configs/.ml_config.cfg"
)

# Отдельный cfg только для IFT TagMe proxy:
# - вызовы модели через TAGME_IFT_URL;
# - загрузка файлов через TAGME_IFT_FILES_URL.
# Обычная отправка разметки/работа с не-IFT TagMe остается на DEFAULT_PROXY_CFG_PATH.
TAGME_IFT_PROXY_CFG_PATH: str = os.path.expanduser(
    (os.getenv("TAGME_IFT_PROXY_CFG_PATH") or "").strip() or "~/configs/.ml_config_ift.cfg"
)

GIGAHF_PROXY_CFG_PATH: str = os.path.expanduser(
    (os.getenv("GIGAHF_PROXY_CFG_PATH") or "").strip() or "~/configs/.ml_config.cfg"
)

VIDEO_PROXY_MAX_JSON_MB = float(os.getenv("VIDEO_PROXY_MAX_JSON_MB", "0.5"))
VIDEO_PROXY_MAX_JSON_BYTES = int(VIDEO_PROXY_MAX_JSON_MB * 1024 * 1024)

VIDEO_DOWNLOAD_TIMEOUT_SEC = float(os.getenv("VIDEO_DOWNLOAD_TIMEOUT_SEC", "60"))

IMG_DOWNLOAD_TIMEOUT_SEC = float(os.getenv("IMG_DOWNLOAD_TIMEOUT_SEC", "20"))
IMG_MAX_SIDE = int(os.getenv("IMG_MAX_SIDE", "512"))
IMG_JPEG_QUALITY = int(os.getenv("IMG_JPEG_QUALITY", "60"))

AUDIO_DOWNLOAD_TIMEOUT_SEC = float(os.getenv("AUDIO_DOWNLOAD_TIMEOUT_SEC", "30"))
AUDIO_PROXY_MAX_JSON_MB = float(os.getenv("AUDIO_PROXY_MAX_JSON_MB", "10"))

GEMINI_VIDEO_MAX_HEIGHT = int(os.getenv("GEMINI_VIDEO_MAX_HEIGHT", "350"))
GEMINI_VIDEO_TARGET_FPS = int(os.getenv("GEMINI_VIDEO_TARGET_FPS", "5"))

# Минимальный таймаут (сек) для моделей Gemini: они часто длинно "думают"
# над видео/изображениями, поэтому берём с запасом — не меньше 15 минут.
GEMINI_MIN_TIMEOUT_SEC = float(os.getenv("GEMINI_MIN_TIMEOUT_SEC", "900"))

GIGAHF_QWEN_MODELS = {
    "Qwen3-VL-235B-A22B-Instruct",
    "Qwen2.5-VL-72B-Instruct-AWQ",
    "Qwen3-VL-235B-A22B-Instruct-AWQ"
}

GIGAHF_TEXT_MODELS = {
    "DeepSeek-V3.2",
    "Qwen3-Next-80B-A3B-Thinking",
    "Qwen3-235B-A22B-Instruct-2507",
    "gpt-oss-120b"
}

TAGME_MODELS = {
    "DeepSeek-R1",
    "GigaChat",
    "GigaChat-2",
    "GigaChat-3-Ultra",
    "GigaChat-3-Lightning",
    "GigaChat-2-Pro",
    "GigaChat-Pro",
    "GigaChat-2-Max",
    "GigaChat-3-Pro",
    "GigaChat-Max",
    "Qwen",
    "GigaChat-3.5-432B-A28B"
}


def _is_gemini_model(model: str) -> bool:
    m = (model or "").strip().lower()
    return "gemini" in m


def _is_kimi_model(model: str) -> bool:
    m = (model or "").strip().lower()
    return "kimi" in m



_EFFORT_ALIASES = {
    "off": "none",
    "disabled": "none",
    "minimal": "low",
}
_GEMINI_THINKING_LEVEL = {
    "none": "low",
    "low": "low",
    "medium": "low",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}
_CLAUDE_EFFORT = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "max",
    "max": "max",
}
# Для Claude с effort=max весь max_tokens может уйти на рассуждения и content будет пустым.
CLAUDE_MAX_EFFORT_MIN_TOKENS = 16000


def normalize_reasoning_effort(effort: Any) -> Optional[str]:
    """Приводит значение из конфига к виду none/low/medium/high/xhigh/max. Пусто -> None."""
    if effort is None:
        return None
    e = str(effort).strip().lower()
    if not e:
        return None
    return _EFFORT_ALIASES.get(e, e)


def build_reasoning_params(model: str, effort: Optional[str]) -> Tuple[Dict[str, Any], Optional[int]]:
    """
    Возвращает (доп. поля payload, минимальный max_tokens или None) для управления
    reasoning на vsellm в зависимости от семейства модели:
      - GPT (/responses):   reasoning={"effort": ...}
      - Gemini (chat):      thinking_level=low|high
      - Claude (chat):      reasoning_effort=low|medium|high|max
      - DeepSeek (chat):    только выключение: thinking={"type": "disabled"}
    Для остальных семейств ничего не добавляется.
    """
    e = normalize_reasoning_effort(effort)
    if e is None:
        return {}, None

    m = (model or "").strip().lower()

    if "gemini" in m:
        level = _GEMINI_THINKING_LEVEL.get(e)
        return ({"thinking_level": level} if level else {}), None

    if "claude" in m or "anthropic" in m:
        level = _CLAUDE_EFFORT.get(e)
        if not level:
            return {}, None
        return {"reasoning_effort": level}, (CLAUDE_MAX_EFFORT_MIN_TOKENS if level == "max" else None)

    if "deepseek" in m:
        if e == "none":
            return {"thinking": {"type": "disabled"}}, None
        return {}, None

    if _is_gpt_responses_capable(m):
        return {"reasoning": {"effort": e}}, None

    return {}, None



_GPT_RESPONSES_MIN_VERSION = (5, 6)
_GPT_VERSION_RE = re.compile(r"(?:^|[/\-_])gpt-(\d+)(?:\.(\d+))?")


def _is_gpt_responses_capable(model_lower: str) -> bool:
    if get_model_endpoint(model_lower) == "responses":
        return True
    match = _GPT_VERSION_RE.search(model_lower)
    if not match:
        return False
    version = (int(match.group(1)), int(match.group(2) or 0))
    return version >= _GPT_RESPONSES_MIN_VERSION


def get_proxy_token(proxy_cfg_path: str) -> str:
    try:
        from crowd_sdk.tagme import TagmeClientAdvanced
    except Exception as e:
        raise RuntimeError(
            "crowd_sdk.tagme.TagmeClientAdvanced недоступен — не могу получить proxy token"
        ) from e

    client = TagmeClientAdvanced(proxy_cfg_path)
    token = getattr(getattr(client, "client", None), "token", None)
    if not token:
        raise RuntimeError("Не получилось получить proxy token: client.client.token пустой")
    return str(token)


def _extract_usage_from_json(j: dict[str, Any]) -> Dict[str, int]:
    usage = j.get("usage") or {}

    # Унификация usage под один формат ключей (prompt_tokens /
    # completion_tokens / total_tokens), который ожидает _add_usage в боте.
    #
    # Причина: два разных формата ответа прокси:
    #   - chat/completions (Gemini, Anthropic, Moonshot, GigaChat): usage
    #     содержит prompt_tokens / completion_tokens / total_tokens;
    #   - responses (OpenAI GPT-5.6): usage содержит input_tokens /
    #     output_tokens / total_tokens. Прежний код читал только
    #     prompt_tokens/completion_tokens — для GPT они отсутствуют,
    #     поэтому статистика по токенам у GPT всегда была нулевой.
    #
    # Маппинг полей (семантически эквивалентны):
    #   input_tokens   -> prompt_tokens     (токены запроса)
    #   output_tokens  -> completion_tokens (токены сгенерированного ответа)
    #   total_tokens   -> total_tokens      (сумма; одинаковый ключ в обоих)
    #
    # Обратная совместимость: chat-модели имеют prompt_tokens/completion_tokens,
    # поэтому первый .get() сработает и фолбэк до input_tokens/output_tokens
    # не выполняется — поведение chat-моделей не меняется.

    prompt = usage.get("prompt_tokens")
    if prompt is None:
        # GPT /responses: берём input_tokens как аналог prompt_tokens.
        prompt = usage.get("input_tokens")

    completion = usage.get("completion_tokens")
    if completion is None:
        # GPT /responses: берём output_tokens как аналог completion_tokens.
        completion = usage.get("output_tokens")

    total = usage.get("total_tokens")
    if total is None:
        # OpenAI /responses обычно шлёт total_tokens, но на случай,
        # если поле отсутствует, считаем сами из input+output, чтобы
        # статистика не обнулялась.
        total = (int(prompt or 0)) + (int(completion or 0))

    return {
        "prompt_tokens": int(prompt or 0),
        "completion_tokens": int(completion or 0),
        "total_tokens": int(total or 0),
    }


def _extract_text_from_json(j: dict[str, Any]) -> str:
    try:
        choice = (j.get("choices") or [None])[0] or {}
        msg = choice.get("message") or {}

        # Основной OpenAI-like формат: choices[0].message.content.
        # Для Qwen/некоторых gateway иногда встречаются text/answer/reasoning_content.
        for msg_key in ("content", "text", "answer", "raw_text", "response", "reasoning_content", "reasoning"):
            content = msg.get(msg_key)

            if isinstance(content, list):
                parts: List[str] = []
                for it in content:
                    if isinstance(it, dict):
                        for part_key in ("text", "content", "value"):
                            if it.get(part_key):
                                parts.append(str(it[part_key]))
                                break
                    elif isinstance(it, str):
                        parts.append(it)
                text = "\n".join([p.strip() for p in parts if p and p.strip()]).strip()
                if text:
                    return text

            if isinstance(content, dict):
                for part_key in ("text", "content", "value"):
                    val = content.get(part_key)
                    if val:
                        return str(val).strip()

            if content:
                text = str(content).strip()
                if text:
                    return text

        # Еще один частый вариант: choices[0].text.
        choice_text = choice.get("text")
        if choice_text:
            return str(choice_text).strip()

        return ""
    except Exception:
        return ""


def _extract_reasoning_from_json(j: dict) -> str:
    """Извлекает reasoning_content из choices[0].message (для моделей с think)."""
    try:
        choice = (j.get("choices") or [None])[0] or {}
        msg = choice.get("message") or {}
        return str(msg.get("reasoning_content") or "")
    except Exception:
        return ""


def _extract_reasoning_tokens_from_json(j: Any) -> int:
    """reasoning-токены: usage.output_tokens_details (/responses) или usage.completion_tokens_details (chat)."""
    try:
        usage = (j or {}).get("usage") or {}
        for details_key in ("output_tokens_details", "completion_tokens_details"):
            details = usage.get(details_key) or {}
            val = details.get("reasoning_tokens")
            if val is not None:
                return int(val or 0)
    except Exception:
        pass
    return 0


def _extract_reasoning_text_from_json(j: Any) -> str:
    """
    Текст reasoning, если он приходит: choices[0].message.reasoning_content / reasoning (chat)
    или summary в блоках type=reasoning из output (/responses).
    """
    try:
        if not isinstance(j, dict):
            return ""
        choice = (j.get("choices") or [None])[0] or {}
        msg = choice.get("message") or {}
        for key in ("reasoning_content", "reasoning"):
            val = msg.get(key)
            if isinstance(val, str) and val.strip():
                return val
        output = j.get("output")
        if isinstance(output, list):
            parts: List[str] = []
            for item in output:
                if not isinstance(item, dict) or item.get("type") != "reasoning":
                    continue
                summary = item.get("summary")
                if isinstance(summary, list):
                    for s in summary:
                        t = s.get("text") if isinstance(s, dict) else None
                        if t:
                            parts.append(str(t))
                elif isinstance(summary, str) and summary:
                    parts.append(summary)
            return "\n".join(parts).strip()
    except Exception:
        pass
    return ""


def _extract_applied_effort_from_json(j: Any) -> str:
    """Эхо применённого effort есть только у GPT /responses: response.reasoning.effort."""
    try:
        r = (j or {}).get("reasoning")
        if isinstance(r, dict):
            return str(r.get("effort") or "")
    except Exception:
        pass
    return ""


def _extract_text_from_gigahf_content(content: Any) -> str:
    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, dict):
        if content.get("text"):
            return str(content["text"]).strip()
        if content.get("content"):
            return _extract_text_from_gigahf_content(content.get("content"))
        return ""

    if isinstance(content, list):
        parts: List[str] = []
        for it in content:
            if isinstance(it, str):
                s = it.strip()
                if s:
                    parts.append(s)
            elif isinstance(it, dict):
                if it.get("text"):
                    s = str(it["text"])
                    if s:
                        parts.append(s)
                elif it.get("content"):
                    s = _extract_text_from_gigahf_content(it.get("content"))
                    if s:
                        parts.append(s)
        return "\n".join(parts).strip()

    return ""


def _extract_text_from_gigahf_json(j: dict[str, Any]) -> str:
    """
    Формат GigaHF (aihf) может отличаться в зависимости от модели/роута.
    Поддерживаем несколько вариантов структуры ответа.
    """
    try:
        if not isinstance(j, dict):
            return ""

        for top_key in ("answer", "raw_text", "text", "output_text", "response"):
            top_val = j.get(top_key)
            top_text = _extract_text_from_gigahf_content(top_val)
            if top_text:
                return top_text

        choices = j.get("choices") or []
        if not choices:
            return ""
        choice0 = choices[0] or {}

        messages = choice0.get("messages") or []
        for msg in messages:
            text = _extract_text_from_gigahf_content((msg or {}).get("content"))
            if text:
                return text

        message = choice0.get("message") or {}
        text = _extract_text_from_gigahf_content(message.get("content"))
        if text:
            return text

        delta = choice0.get("delta") or {}
        text = _extract_text_from_gigahf_content(delta.get("content"))
        if text:
            return text

        text = _extract_text_from_gigahf_content(choice0.get("text"))
        if text:
            return text

        return ""
    except Exception:
        logger.exception("Failed to parse GigaHF response text")
        return ""


def _extract_by_path(obj: Any, path: Tuple[str, ...]) -> Any:
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _extract_text_from_responses_json(j: dict[str, Any]) -> str:
    """
    OpenAI /responses API: текст лежит в output[].content[].text
    (где part.type == 'output_text'). Этот формат отличается от
    chat/completions (choices[0].message.content), поэтому стандартные
    extractor'ы его не видят и ответ получается пустым.

    Срабатывает только когда output — список (GPT /responses);
    для chat-моделей output — dict или отсутствует, вернёт "".
    """
    try:
        if not isinstance(j, dict):
            return ""
        output = j.get("output")
        if not isinstance(output, list):
            return ""
        parts: List[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    t = part.get("text") or part.get("content")
                    if t:
                        parts.append(str(t))
            elif isinstance(content, str) and content:
                parts.append(content)
        return "\n".join(p.strip() for p in parts if p and p.strip()).strip()
    except Exception:
        return ""


def _extract_text_from_any_json(j: dict[str, Any]) -> str:
    """
    Универсальный extractor для разных proxy-форматов.

    Почему нужен:
    - обычный TagMe/VseLLM часто возвращает OpenAI-like choices[0].message.content;
    - IFT proxy иногда отдает уже готовый ответ в top-level/вложенных полях
      answer/text/response/output/result/data;
    - OpenAI /responses отдаёт текст в output[].content[].text;
    - если текст не распарсить, внешний цикл разметки видит пустой answer/raw_text
      и может повторно вызывать модель, хотя raw-ответ уже пришел.
    """
    if not isinstance(j, dict):
        return ""

    # 1. Основной OpenAI-like формат.
    text = _extract_text_from_json(j)
    if text:
        return text

    # 1.5. OpenAI /responses format: output[].content[].text
    text = _extract_text_from_responses_json(j)
    if text:
        return text

    # 2. Форматы, похожие на GigaHF/AIHF и часть IFT-ответов.
    text = _extract_text_from_gigahf_json(j)
    if text:
        return text

    # 3. Частые вложенные варианты IFT/прокси-ответов.
    candidate_paths: List[Tuple[str, ...]] = [
        ("result", "answer"),
        ("result", "raw_text"),
        ("result", "text"),
        ("result", "content"),
        ("result", "response"),
        ("data", "answer"),
        ("data", "raw_text"),
        ("data", "text"),
        ("data", "content"),
        ("data", "response"),
        ("output", "answer"),
        ("output", "raw_text"),
        ("output", "text"),
        ("output", "content"),
        ("output", "response"),
    ]

    for path in candidate_paths:
        val = _extract_by_path(j, path)
        text = _extract_text_from_gigahf_content(val)
        if text:
            return text

    return ""


def _extract_finish_reason_from_json(j: dict[str, Any]) -> str:
    if not isinstance(j, dict):
        return ""

    for key in ("finish_reason", "finishReason", "stop_reason", "stopReason", "status", "state"):
        val = j.get(key)
        if val is not None:
            return str(val).strip()

    choices = j.get("choices") or []
    if choices:
        choice0 = choices[0] or {}
        for key in ("finish_reason", "finishReason", "stop_reason", "stopReason", "status", "state"):
            val = choice0.get(key)
            if val is not None:
                return str(val).strip()

    for path in (("result", "finish_reason"), ("result", "status"), ("data", "finish_reason"), ("data", "status")):
        val = _extract_by_path(j, path)
        if val is not None:
            return str(val).strip()

    return ""


def _is_complete_response(j: dict[str, Any], text: Optional[str]) -> bool:
    """
    Возвращаем True, если ответ можно считать завершенным для внешнего цикла разметки.
    Если текст распарсился — это уже завершенный ответ для наших целей.
    """
    if text and str(text).strip():
        return True

    if not isinstance(j, dict):
        return False

    for key in ("complete", "completed", "is_complete", "isCompleted"):
        val = j.get(key)
        if isinstance(val, bool):
            return val
        if isinstance(val, str) and val.strip().lower() in {"true", "yes", "1", "complete", "completed", "done", "success"}:
            return True

    finish_reason = _extract_finish_reason_from_json(j).strip().lower()
    return finish_reason in {"stop", "complete", "completed", "done", "success", "end_turn"}


def _content_to_plain_text(content: Any) -> str:
    """Преобразует content из разных message-форматов в обычный текст."""
    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, dict):
        for key in ("text", "content", "value"):
            val = content.get(key)
            text = _content_to_plain_text(val)
            if text:
                return text
        return ""

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            text = _content_to_plain_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()

    return str(content).strip()


def _build_ift_text_messages(
        *,
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        user_text: Optional[str] = None,
        extra_text: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Для IFT text route используем тот же простой формат, что уже стабильно работает
    в IFT image route: один user-message с цельным текстом промпта.

    Это особенно важно для Qwen через IFT proxy: классический набор system/user
    может вернуть raw-ответ, но внешний цикл разметки не всегда видит корректное
    завершение/answer.
    """
    prompt_parts: List[str] = []

    if messages:
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "").strip()
            content_text = _content_to_plain_text(msg.get("content"))
            if not content_text:
                continue

            # Не вставляем технические role-labels в промпт, но сохраняем порядок сообщений.
            # System-инструкцию просто кладем первой частью промпта.
            prompt_parts.append(content_text)
    else:
        if system_prompt:
            prompt_parts.append(str(system_prompt).strip())
        if user_text:
            prompt_parts.append(str(user_text).strip())

    if extra_text:
        prompt_parts.append(str(extra_text).strip())

    prompt = "\n\n".join([p for p in prompt_parts if p]).strip()
    if not prompt:
        prompt = "Выполни разметку согласно инструкции."

    return [{"role": "user", "content": prompt}]


def _post_proxy(*, proxy_url: str, token: str, payload: dict[str, Any], timeout_sec: float) -> requests.Response:
    return requests.post(
        proxy_url,
        json=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        verify=False,
        timeout=(60.0, float(timeout_sec)),
    )


def _retry_call(n: int, fn, *, label: str, delay_sec: float = 0.0):
    last_exc: Optional[BaseException] = None
    for i in range(1, n + 1):
        try:
            logger.info("{} attempt {}/{}", label, i, n)
            return fn()
        except Exception as e:
            last_exc = e
            logger.warning("{} attempt {}/{} failed: {}", label, i, n, e)
            if delay_sec and i < n:
                time.sleep(delay_sec)
    assert last_exc is not None
    raise last_exc


def _clean_media_url(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s or None


def _resolve_media_ref(ref: Any, row: Any = None) -> Any:
    """Имя столбца -> ссылка из строки данных.

    Если передан row (dict / pandas.Series — ОДНА строка) и ref совпадает с названием
    его столбца, берём ссылку из row[ref]. Иначе ref возвращается как есть (обычная
    ссылка, data:-URL и т.д.), поэтому вызовы без row работают как раньше.
    Пустая ячейка (None / NaN / '') -> None.
    Работает одинаково для image_url / video_url / audio_url.
    """
    if ref is None or row is None or not isinstance(ref, str):
        return ref
    key = ref.strip()
    try:
        if key not in row:
            return ref
        val = row[key]
    except Exception:
        return ref
    if hasattr(val, "iloc"):
        raise ValueError(
            f"call_vsellm(row=...): ожидается ОДНА строка (dict / pandas.Series), "
            f"а по ключу {key!r} вернулся целый столбец"
        )
    if val is None or (isinstance(val, float) and val != val):
        return None
    s = str(val).strip()
    return s or None


def _call_proxy_with_refresh(
        *,
        proxy_url: str,
        proxy_token: str,
        proxy_cfg_path: str,
        payload: dict[str, Any],
        timeout_sec: float,
) -> tuple[dict[str, Any], str]:
    r = _post_proxy(proxy_url=proxy_url, token=proxy_token, payload=payload, timeout_sec=timeout_sec)

    if r.status_code in (401, 403):
        logger.warning(
            "Proxy token expired (status={}), refreshing token using cfg_path={}",
            r.status_code,
            proxy_cfg_path,
        )
        proxy_token = get_proxy_token(proxy_cfg_path)
        logger.info("Proxy token refreshed (len={})", len(proxy_token))

        r = _post_proxy(proxy_url=proxy_url, token=proxy_token, payload=payload, timeout_sec=timeout_sec)

    # ВАЖНО: логируем тело ошибки, чтобы понимать причину 400/422
    if r.status_code >= 400:
        logger.error(
            "Proxy request failed: status={} url={} body={}",
            r.status_code,
            proxy_url,
            r.text[:4000],
        )

    r.raise_for_status()
    return r.json(), proxy_token


def _json_size_bytes(obj: Any) -> int:
    try:
        return len(json.dumps(obj, ensure_ascii=False).encode("utf-8"))
    except Exception:
        return 0


def _download_image_bytes(url: str, timeout_sec: float = IMG_DOWNLOAD_TIMEOUT_SEC) -> bytes:
    r = requests.get(url, timeout=float(timeout_sec), verify=False)
    r.raise_for_status()
    return r.content or b""


def _resize_keep_aspect(img: Image.Image, max_side: int) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    return img.resize((nw, nh))


def _image_to_data_url_jpeg(img: Image.Image, quality: int) -> str:
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=int(quality), optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _download_image_as_data_url_max_compress(image_url: str) -> str:
    b = _download_image_bytes(image_url, timeout_sec=IMG_DOWNLOAD_TIMEOUT_SEC)
    if not b:
        raise RuntimeError("Downloaded image is empty")
    img = Image.open(BytesIO(b))
    img = _resize_keep_aspect(img, IMG_MAX_SIDE)
    return _image_to_data_url_jpeg(img, IMG_JPEG_QUALITY)


def _data_image_url_to_data_url_max_compress(data_url: str) -> str:
    s = (data_url or "").strip()
    if not s.startswith("data:image/") or ";base64," not in s:
        raise ValueError("Not a base64 data:image URL")

    b64 = s.split(";base64,", 1)[1]
    raw = base64.b64decode(b64)

    img = Image.open(BytesIO(raw))
    img = _resize_keep_aspect(img, IMG_MAX_SIDE)
    return _image_to_data_url_jpeg(img, IMG_JPEG_QUALITY)


def _download_video_bytes(url: str, timeout_sec: float = VIDEO_DOWNLOAD_TIMEOUT_SEC) -> bytes:
    r = requests.get(url, timeout=float(timeout_sec), verify=False)
    r.raise_for_status()
    return r.content or b""


def _video_bytes_to_data_uri_mp4(video_bytes: bytes) -> str:
    b64 = base64.b64encode(video_bytes).decode("utf-8")
    return f"data:video/mp4;base64,{b64}"


def _compress_video_bytes_opencv(
        video_bytes: bytes,
        *,
        max_height: int,
        target_fps: int,
) -> bytes:
    try:
        import cv2 as cv
    except Exception as e:
        raise RuntimeError("OpenCV (cv2) is required for video compression fallback") from e

    in_path = None
    out_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            in_path = f.name
            f.write(video_bytes)

        out_fd, out_path = tempfile.mkstemp(suffix=".mp4")
        os.close(out_fd)

        cap = cv.VideoCapture(in_path)
        if not cap.isOpened():
            raise RuntimeError("Failed to open input video for compression")

        fps_in = cap.get(cv.CAP_PROP_FPS) or 0
        w_in = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) or 0)
        h_in = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) or 0)
        if w_in <= 0 or h_in <= 0:
            cap.release()
            raise RuntimeError("Failed to read input video dimensions")

        if h_in > max_height:
            scale = max_height / float(h_in)
            w_out = max(2, int(w_in * scale))
            h_out = max(2, int(h_in * scale))
        else:
            w_out, h_out = w_in, h_in

        if w_out % 2:
            w_out -= 1
        if h_out % 2:
            h_out -= 1
        w_out = max(2, w_out)
        h_out = max(2, h_out)

        fps_out = max(1, int(target_fps) if target_fps > 0 else (int(fps_in) if fps_in else 4))

        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        writer = cv.VideoWriter(out_path, fourcc, float(fps_out), (w_out, h_out))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Failed to open output writer for compression")

        keep_every = 1
        if fps_in and fps_out and fps_in > fps_out:
            keep_every = max(1, int(round(float(fps_in) / float(fps_out))))

        frame_idx = 0
        written = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if keep_every > 1 and (frame_idx % keep_every) != 0:
                frame_idx += 1
                continue
            frame_idx += 1

            if (w_out, h_out) != (w_in, h_in):
                frame = cv.resize(frame, (w_out, h_out), interpolation=cv.INTER_AREA)

            writer.write(frame)
            written += 1

        cap.release()
        writer.release()

        if written <= 0:
            raise RuntimeError("No frames written during compression")

        with open(out_path, "rb") as f:
            return f.read()

    finally:
        for p in (in_path, out_path):
            if p:
                try:
                    os.remove(p)
                except Exception:
                    pass


_payload_config_logged_keys: set = set()
_payload_config_lock = threading.Lock()


def _redact_payload_for_log(obj: Any) -> Any:
    """Красит data:image/... url в заглушку с длиной — чтобы не дампить base64 в лог."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "url" and isinstance(v, str) and v.startswith("data:image/"):
                out[k] = f"<base64 image redacted, len={len(v)}>"
            else:
                out[k] = _redact_payload_for_log(v)
        return out
    if isinstance(obj, list):
        return [_redact_payload_for_log(x) for x in obj]
    return obj


def _payload_image_free_signature(payload: Dict[str, Any]) -> str:
    """Сигнатура payload без image-data.

    Картинка меняется от файла к файлу, остальной конфиг — нет.
    По этой сигнатуре логируем конфиг один раз на задачу:
    повторные запросы с тем же конфигом не дампят payload повторно.
    """

    def _strip(obj: Any, parent_key: str = "") -> Any:
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if k == "url" and parent_key == "image_url":
                    out[k] = "<image>"
                else:
                    out[k] = _strip(v, parent_key=k)
            return out
        if isinstance(obj, list):
            return [_strip(x, parent_key=parent_key) for x in obj]
        return obj

    return json.dumps(_strip(payload), ensure_ascii=False, sort_keys=True)


def _log_request_config_once(*, route: str, url: str, model: str, payload: Dict[str, Any]) -> None:
    """Логирует полный конфиг запроса один раз для уникального (route + конфиг).

    Конфиг = всё, кроме image-data (она меняется от файла к файлу).
    Повторные запросы в рамках той же задачи не повторяют дамп payload.
    """
    key = (route, _payload_image_free_signature(payload))
    with _payload_config_lock:
        if key in _payload_config_logged_keys:
            return
        _payload_config_logged_keys.add(key)
    logger.info(
        "{} IMAGE REQUEST | url={} model={} payload={}",
        route,
        url,
        model,
        json.dumps(_redact_payload_for_log(payload), ensure_ascii=False, indent=2),
    )


def _log_image_parsed(*, route: str, text: Optional[str], usage: Dict[str, int], raw: Any, elapsed: Optional[float] = None) -> None:
    """Унифицированный per-request итог разбора ответа: короткий PARSED + дамп при пустом ответе.

    elapsed (сек) — latency HTTP-вызова; None если не измерялось.
    """
    elapsed_str = f"{elapsed:.2f}s" if elapsed is not None else "n/a"
    logger.info(
        "{} IMAGE PARSED | answer_len={} usage={} raw_keys={} elapsed={}",
        route,
        len(text or ""),
        usage,
        list(raw.keys()) if isinstance(raw, dict) else type(raw),
        elapsed_str,
    )
    if not text:
        logger.error(
            "{} IMAGE EMPTY TEXT | raw_response_full={}",
            route,
            json.dumps(raw, ensure_ascii=False, indent=2) if isinstance(raw, dict) else raw,
        )


class VseLLMProxyClient:
    def __init__(
            self,
            *,
            proxy_cfg_path: str | None = None,
            proxy_url: str | None = None,
            proxy_token: str | None = None,
            request_timeout: float = 120.0,
    ):
        self.proxy_cfg_path = (proxy_cfg_path or DEFAULT_PROXY_CFG_PATH).strip()
        self.proxy_url = (proxy_url or TAGME_LLM_PROXY_URL).strip()
        self.request_timeout = float(request_timeout)

        self._proxy_token = (proxy_token or "").strip() or get_proxy_token(self.proxy_cfg_path)
        self._lock = threading.Lock()

    def _call(
            self,
            payload: dict[str, Any],
            proxy_url: Optional[str] = None,
    ) -> dict[str, Any]:
        # URL передаётся per-call, чтобы не мутировать self.proxy_url
        # синглтон-клиента вне локока (гонка при конкурентных вызовах
        # chat/responses моделей на одном _default_client).
        target = proxy_url or self.proxy_url
        with self._lock:
            j, self._proxy_token = _call_proxy_with_refresh(
                proxy_url=target,
                proxy_token=self._proxy_token,
                proxy_cfg_path=self.proxy_cfg_path,
                payload=payload,
                timeout_sec=self.request_timeout,
            )
        return j

    def call_text(
            self,
            *,
            model: str,
            messages: List[Dict[str, Any]],
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            top_p: Optional[float] = None,
            top_k: Optional[float] = None,
            api_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], Dict[str, int], Any]:
        endpoint = get_model_endpoint(model)
        # Если задан reasoning_effort, build_reasoning_params положил в api_kwargs
        # {"reasoning": {"effort": ...}} - такой запрос возможен только на /responses.
        # Без reasoning_effort этого ключа нет, и маршрут остаётся прежним.
        if (
                endpoint != "responses"
                and api_kwargs
                and isinstance(api_kwargs.get("reasoning"), dict)
                and "effort" in api_kwargs["reasoning"]
        ):
            endpoint = "responses"

        payload: Dict[str, Any] = {
            "model": model,
            "input" if endpoint == "responses" else "messages": messages,
        }

        # GPT-5 reasoning models на /responses режектят temperature —
        # отправляем только для chat-эндпоинтов (gemini/anthropic/moonshot).
        if endpoint != "responses" and temperature is not None:
            payload["temperature"] = float(temperature)
        if endpoint == "responses":
            if max_tokens is not None:
                payload["max_output_tokens"] = int(max_tokens)
            else:
                payload["max_output_tokens"] = 1024
        else:
            if max_tokens is not None:
                payload["max_tokens"] = int(max_tokens)
        if endpoint != "responses":
            if top_p is not None:
                payload["top_p"] = float(top_p)
            if top_k is not None:
                payload["top_k"] = float(top_k)

        if api_kwargs:
            for k, v in api_kwargs.items():
                if v is not None:
                    payload[k] = v

        # OpenAI /responses: встроенный web_search принимает только "type".
        # Поле "name" валидно для gemini/anthropic на /chat/completions,
        # но на /responses вызывает 400 invalid_request_error. Чистим только GPT.
        if endpoint == "responses" and isinstance(payload.get("tools"), list):
            cleaned_tools: List[Dict[str, Any]] = []
            for t in payload["tools"]:
                if not isinstance(t, dict):
                    continue
                ct = dict(t)
                ct.pop("name", None)
                cleaned_tools.append(ct)
            payload["tools"] = cleaned_tools

        # URL считается per-call и передаётся в _call, чтобы не мутировать
        # self.proxy_url синглтона вне локока (гонка при конкурентных вызовах
        # chat/responses моделей на одном _default_client).
        target_url = self.proxy_url
        if endpoint == "responses" and "/chat/completions" in target_url:
            target_url = target_url.replace("/chat/completions", "/responses")
            logger.info(
                "OpenAI GPT-5.6: using responses endpoint {}",
                target_url,
            )

        j = self._call(payload, proxy_url=target_url)

        usage = _extract_usage_from_json(j)
        text = _extract_text_from_any_json(j)
        return (text or None), usage, j

    def call_embeddings(
            self,
            *,
            model: str,
            inputs: str | List[str],
            proxy_url: Optional[str] = None,
            api_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[List[float]], Dict[str, int], Any]:
        """
        Получает эмбеддинги через TagMe/VseLLM proxy.

        Для авторизации используется тот же TagMe proxy token,
        который извлекается из ml_config.cfg.
        """
        target_url = (
                proxy_url
                or TAGME_EMBEDDINGS_PROXY_URL
        ).strip()

        if not target_url:
            raise ValueError("Embeddings proxy URL is empty")

        payload: Dict[str, Any] = {
            "model": model,
            "input": inputs,
        }

        if api_kwargs:
            for key, value in api_kwargs.items():
                if value is not None:
                    payload[key] = value

        with self._lock:
            response_json, self._proxy_token = _call_proxy_with_refresh(
                proxy_url=target_url,
                proxy_token=self._proxy_token,
                proxy_cfg_path=self.proxy_cfg_path,
                payload=payload,
                timeout_sec=self.request_timeout,
            )

        data = response_json.get("data")

        if not isinstance(data, list):
            raise RuntimeError(
                "Unexpected embeddings response: "
                f"{json.dumps(response_json, ensure_ascii=False)[:2000]}"
            )

        embeddings: List[List[float]] = []

        for index, item in enumerate(data):
            if not isinstance(item, dict):
                raise RuntimeError(
                    f"Embeddings response data[{index}] "
                    f"is not an object: {item!r}"
                )

            vector = item.get("embedding")

            if not isinstance(vector, list):
                raise RuntimeError(
                    f"Embeddings response data[{index}] "
                    f"has no embedding: {item!r}"
                )

            embeddings.append(
                [float(value) for value in vector]
            )

        usage = _extract_usage_from_json(response_json)

        return embeddings, usage, response_json

    def call_image(
            self,
            *,
            model: str,
            system_prompt: Optional[str],
            image_url: str,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            top_p: Optional[float] = None,
            extra_text: Optional[str] = None,
            user_text: Optional[str] = None,
            api_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], Dict[str, int], Any]:

        user_text = _image_user_text(system_prompt, user_text, extra_text)

        is_gemini = _is_gemini_model(model)
        is_kimi = _is_kimi_model(model)
        force_base64_image = is_gemini or is_kimi

        def _build_messages_with_image_url(img_ref: str) -> List[Dict[str, Any]]:
            msgs: List[Dict[str, Any]] = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            user_content: List[Dict[str, Any]] = []
            if user_text:
                user_content.append({"type": "text", "text": user_text})
            user_content.append({"type": "image_url", "image_url": {"url": img_ref}})
            msgs.append({"role": "user", "content": user_content})
            return msgs

        def _build_payload(msgs: List[Dict[str, Any]]) -> Dict[str, Any]:
            payload: Dict[str, Any] = {"model": model, "messages": msgs}

            if temperature is not None:
                payload["temperature"] = float(temperature)
            if max_tokens is not None:
                payload["max_tokens"] = int(max_tokens)
            if top_p is not None:
                payload["top_p"] = float(top_p)

            if api_kwargs:
                for k, v in api_kwargs.items():
                    if v is not None:
                        payload[k] = v

            return payload

        # === ДИНАМИЧЕСКОЕ СЖАТИЕ ИЗОБРАЖЕНИЙ ===
        def _dynamic_image_data_url_from_bytes(raw_bytes: bytes) -> str:
            """
            1) Пытаемся отправить картинку без уменьшения размера (orig side, качество ~95).
            2) Если payload >= 0.5MB, пробуем последовательно более агрессивные варианты:
               - max_side=700, quality=60 (как ты просил),
               - далее ступенчато уменьшаем max_side/quality,
               пока не влезем < 0.5MB или не исчерпаем пресеты.
            Если ни один не влез — берём тот, у которого payload минимальный.
            """
            img = Image.open(BytesIO(raw_bytes))

            presets: List[Tuple[int, int]] = [
                (max(img.size), 95),  # "почти без сжатия", первый тест
                (700, 60),  # явный requested preset
                (512, 60),
                (384, 60),
                (384, 40),
                (256, 40),
            ]

            best_data_url: Optional[str] = None
            best_mb: float = 1e9

            for max_side, quality in presets:
                logger.info("IMAGE dynamic compression try: max_side={} quality={}", max_side, quality)

                if max(img.size) > max_side:
                    img_resized = _resize_keep_aspect(img, max_side)
                else:
                    img_resized = img

                data_url = _image_to_data_url_jpeg(img_resized, quality)
                tmp_msgs = _build_messages_with_image_url(data_url)
                tmp_payload = _build_payload(tmp_msgs)
                mb_size = _json_size_bytes(tmp_payload) / 1024.0 / 1024.0

                logger.info(
                    "IMAGE payload size with max_side={} quality={} -> {:.3f}MB",
                    max_side, quality, mb_size
                )

                if mb_size < best_mb:
                    best_mb = mb_size
                    best_data_url = data_url

                if mb_size < float(VIDEO_PROXY_MAX_JSON_MB):
                    logger.info(
                        "IMAGE dynamic compression fit: max_side={} quality={} payload={:.3f}MB",
                        max_side, quality, mb_size
                    )
                    return data_url

            if best_data_url is None:
                raise RuntimeError("Dynamic image compression failed to produce any data_url")

            logger.warning(
                "Cannot fit image payload under {:.3f}MB; best={:.3f}MB (will use best)",
                float(VIDEO_PROXY_MAX_JSON_MB), best_mb
            )
            return best_data_url

        if force_base64_image:
            logger.info(
                "{} image: dynamic IMAGE_URL base64 with payload<= {:.2f}MB",
                "Kimi" if is_kimi else "Gemini",
                VIDEO_PROXY_MAX_JSON_MB
            )

            # поддержка data:image/...;base64,... без requests.get
            if isinstance(image_url, str) and image_url.strip().startswith("data:image/"):
                s = image_url.strip()
                if not s.startswith("data:image/") or ";base64," not in s:
                    raise ValueError("Not a base64 data:image URL")
                b64 = s.split(";base64,", 1)[1]
                raw = base64.b64decode(b64)
            else:
                raw = _download_image_bytes(image_url, timeout_sec=IMG_DOWNLOAD_TIMEOUT_SEC)

            if not raw:
                raise RuntimeError("Downloaded image is empty")

            data_url = _dynamic_image_data_url_from_bytes(raw)
            payload = _build_payload(_build_messages_with_image_url(data_url))

            _call_t0 = time.time()
            j = _retry_call(3, lambda: self._call(payload), label=("KIMI IMAGE (dynamic b64 image_url)" if is_kimi else "GEMINI IMAGE (dynamic b64 image_url)"))
            _call_elapsed = time.time() - _call_t0
            usage = _extract_usage_from_json(j)
            text = _extract_text_from_any_json(j)
            _log_image_parsed(route="VSELLM", text=text, usage=usage, raw=j, elapsed=_call_elapsed)
            return (text or None), usage, j

        payload_url = _build_payload(_build_messages_with_image_url(image_url))

        _log_request_config_once(route="VSELLM", url=self.proxy_url, model=model, payload=payload_url)

        try:
            _call_t0 = time.time()
            j = _retry_call(3, lambda: self._call(payload_url), label="IMAGE URL")
            _call_elapsed = time.time() - _call_t0
            usage = _extract_usage_from_json(j)
            text = _extract_text_from_any_json(j)
            _log_image_parsed(route="VSELLM", text=text, usage=usage, raw=j, elapsed=_call_elapsed)
            return (text or None), usage, j
        except Exception as e1:
            logger.warning("IMAGE URL failed, switching to dynamic base64: {}", e1)

        if isinstance(image_url, str) and image_url.strip().startswith("data:image/"):
            s = image_url.strip()
            if not s.startswith("data:image/") or ";base64," not in s:
                raise ValueError("Not a base64 data:image URL")
            b64 = s.split(";base64,", 1)[1]
            raw = base64.b64decode(b64)
        else:
            raw = _download_image_bytes(image_url, timeout_sec=IMG_DOWNLOAD_TIMEOUT_SEC)

        if not raw:
            raise RuntimeError("Downloaded image is empty")

        data_url = _dynamic_image_data_url_from_bytes(raw)
        payload_b64 = _build_payload(_build_messages_with_image_url(data_url))

        _call_t0 = time.time()
        j = _retry_call(3, lambda: self._call(payload_b64), label="IMAGE BASE64 (dynamic image_url)")
        _call_elapsed = time.time() - _call_t0
        usage = _extract_usage_from_json(j)
        text = _extract_text_from_any_json(j)
        _log_image_parsed(route="VSELLM", text=text, usage=usage, raw=j, elapsed=_call_elapsed)
        return (text or None), usage, j

    def call_audio(
            self,
            *,
            model: str,
            system_prompt: Optional[str],
            audio_url: str,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            top_p: Optional[float] = None,
            extra_text: Optional[str] = None,
            user_text: Optional[str] = None,
            api_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], Dict[str, int], Any]:
        """Аудио-анализ через прокси TagMe/VseLLM (input_audio с инлайн-base64)"""
        messages = _build_audio_messages(system_prompt, user_text, extra_text, audio_url)

        payload: Dict[str, Any] = {"model": model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        if top_p is not None:
            payload["top_p"] = float(top_p)

        if api_kwargs:
            for k, v in api_kwargs.items():
                if v is not None:
                    payload[k] = v

        payload_mb = _json_size_bytes(payload) / 1024.0 / 1024.0
        if payload_mb > AUDIO_PROXY_MAX_JSON_MB:
            raise RuntimeError(
                "Audio payload {:.3f}MB exceeds proxy limit {:.3f}MB; "
                "use a shorter or downsampled audio".format(
                    payload_mb, AUDIO_PROXY_MAX_JSON_MB)
            )

        _call_t0 = time.time()
        j = _retry_call(3, lambda: self._call(payload), label="AUDIO input_audio (base64)")
        _call_elapsed = time.time() - _call_t0
        usage = _extract_usage_from_json(j)
        text = _extract_text_from_any_json(j)
        _log_image_parsed(route="VSELLM", text=text, usage=usage, raw=j, elapsed=_call_elapsed)
        return (text or None), usage, j

    def call_video_full(
            self,
            *,
            model: str,
            video_url: str,
            prompt_text: str,
            temperature: Optional[float] = None,
            timeout_sec: int = 120,
            api_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], Dict[str, int], Any]:
        prompt = (prompt_text or "").strip() or "Опиши, что происходит в этом видео. Будь конкретным."

        is_gemini = _is_gemini_model(model)

        def _build_payload_url(video_ref: str) -> Dict[str, Any]:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "video_url", "video_url": {"url": video_ref}},
                    ],
                }
            ]
            payload: Dict[str, Any] = {"model": model, "messages": messages}
            payload["temperature"] = float(temperature if temperature is not None else 0.2)

            if api_kwargs:
                for k, v in api_kwargs.items():
                    if v is not None:
                        payload[k] = v

            return payload

        def _build_payload_file(data_uri: str) -> Dict[str, Any]:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "file", "file": {"file_id": data_uri}},
                    ],
                }
            ]
            payload: Dict[str, Any] = {"model": model, "messages": messages}
            payload["temperature"] = float(temperature if temperature is not None else 0.2)

            if api_kwargs:
                for k, v in api_kwargs.items():
                    if v is not None:
                        payload[k] = v

            return payload

        def _payload_size_mb(obj: Any) -> float:
            try:
                return len(json.dumps(obj, ensure_ascii=False).encode("utf-8")) / 1024 / 1024
            except Exception:
                return 0.0

        def _compress_video_to_fit_payload_mb(video_bytes: bytes, *, limit_mb: float) -> str:
            HEIGHTS = [120]
            FPS = [2, 1]

            best_data_uri: Optional[str] = None
            best_mb = 1e9

            for height in HEIGHTS:
                for fps in FPS:
                    logger.info("FAST ladder try: height={} fps={}", height, fps)

                    comp_bytes = _compress_video_bytes_opencv(
                        video_bytes,
                        max_height=height,
                        target_fps=fps,
                    )
                    data_uri = _video_bytes_to_data_uri_mp4(comp_bytes)

                    payload_try = _build_payload_file(data_uri)
                    mb_size = _payload_size_mb(payload_try)

                    if mb_size < best_mb:
                        best_mb = mb_size
                        best_data_uri = data_uri

                    if mb_size <= float(limit_mb):
                        logger.info(
                            "Gemini video fit: height={} fps={} payload={:.3f}MB",
                            height, fps, mb_size
                        )
                        return data_uri

            if best_data_uri is None:
                raise RuntimeError("Gemini video compression failed")

            raise RuntimeError(
                f"Cannot fit payload under {limit_mb:.3f}MB; best={best_mb:.3f}MB"
            )

        old_timeout = self.request_timeout
        self.request_timeout = float(timeout_sec)

        try:
            # ============================================================
            # 1. Сначала ВСЕГДА пробуем отправить видео по URL.
            #
            # Для Gemini это теперь первый transport:
            # {
            #   "type": "video_url",
            #   "video_url": {"url": "..."}
            # }
            #
            # Для не-Gemini поведение остается прежним:
            # пробуем URL и, если он падает, ошибку прокидываем наружу.
            # ============================================================

            url_error: Optional[BaseException] = None

            if not (
                    isinstance(video_url, str)
                    and video_url.strip().startswith("data:video/")
            ):
                try:
                    payload = _build_payload_url(video_url)

                    j = _retry_call(
                        3,
                        lambda: self._call(payload),
                        label=(
                            "GEMINI VIDEO URL"
                            if is_gemini
                            else "VIDEO URL"
                        ),
                    )

                    usage = _extract_usage_from_json(j)
                    text = _extract_text_from_any_json(j)
                    return (text or None), usage, j

                except Exception as e:
                    url_error = e

                    logger.warning(
                        "VIDEO URL failed: model={} is_gemini={} error={}",
                        model,
                        is_gemini,
                        e,
                    )

                    # Для не-Gemini fallback в base64 сейчас не включаем.
                    # Старое поведение сохраняется: если URL не сработал —
                    # вызов падает.
                    if not is_gemini:
                        raise

            else:
                logger.info(
                    "VIDEO input is already data:video URL; skip video_url attempt and use file/base64 route"
                )

            # ============================================================
            # 2. Fallback в base64/file делаем ТОЛЬКО для Gemini.
            # ============================================================

            if not is_gemini:
                raise RuntimeError(
                    "VIDEO URL failed and base64 fallback is enabled only for Gemini"
                ) from url_error

            logger.warning(
                "Gemini video URL failed, fallback to FILE base64 with adaptive compression "
                "to payload <= {:.2f}MB",
                VIDEO_PROXY_MAX_JSON_MB,
            )

            # Если на вход уже пришел data:video/...;base64,...
            # то ничего не скачиваем и не сжимаем, просто отправляем как file_id.
            if isinstance(video_url, str) and video_url.strip().startswith("data:video/"):
                logger.info("VIDEO is already base64 data URL, sending as file_id")

                data_uri = video_url.strip()

                def _one_call_direct():
                    payload = _build_payload_file(data_uri)
                    return self._call(payload)

                j = _retry_call(
                    3,
                    _one_call_direct,
                    label="GEMINI VIDEO FALLBACK (data_url file)",
                )

                usage = _extract_usage_from_json(j)
                text = _extract_text_from_any_json(j)
                return (text or None), usage, j

            raw_bytes = _download_video_bytes(
                video_url,
                timeout_sec=VIDEO_DOWNLOAD_TIMEOUT_SEC,
            )

            if not raw_bytes:
                raise RuntimeError("Downloaded video is empty") from url_error

            def _one_call():
                data_uri = _compress_video_to_fit_payload_mb(
                    raw_bytes,
                    limit_mb=VIDEO_PROXY_MAX_JSON_MB,
                )

                payload = _build_payload_file(data_uri)

                mb_size = _payload_size_mb(payload)

                if mb_size > float(VIDEO_PROXY_MAX_JSON_MB):
                    raise RuntimeError(
                        f"Payload still too large: "
                        f"{mb_size:.3f}MB > "
                        f"{float(VIDEO_PROXY_MAX_JSON_MB):.3f}MB"
                    )

                return self._call(payload)

            j = _retry_call(
                3,
                _one_call,
                label="GEMINI VIDEO FALLBACK (file b64 <=0.5MB payload)",
            )

            usage = _extract_usage_from_json(j)
            text = _extract_text_from_any_json(j)
            return (text or None), usage, j

        finally:
            self.request_timeout = old_timeout


_default_client: Optional[VseLLMProxyClient] = None
try:
    _default_client = VseLLMProxyClient(
        proxy_cfg_path=DEFAULT_PROXY_CFG_PATH,
        proxy_url=TAGME_LLM_PROXY_URL,
        request_timeout=180.0,
    )
except Exception as e:
    logger.warning("VseLLMProxyClient init failed: {}", e)
    _default_client = None



_clients_cache: dict[tuple[str, str, float], VseLLMProxyClient] = {}
_clients_cache_lock = threading.Lock()


def _get_client(
        *,
        proxy_cfg_path: str | None = None,
        proxy_url: str | None = None,
        request_timeout: float = 180.0,
) -> VseLLMProxyClient:
    if _default_client is not None and proxy_cfg_path is None and proxy_url is None:
        return _default_client

    key = (
        proxy_cfg_path or DEFAULT_PROXY_CFG_PATH,
        proxy_url or TAGME_LLM_PROXY_URL,
        float(request_timeout),
    )
    with _clients_cache_lock:
        client = _clients_cache.get(key)
        if client is None:
            client = VseLLMProxyClient(
                proxy_cfg_path=key[0],
                proxy_url=key[1],
                request_timeout=key[2],
            )
            _clients_cache[key] = client
        return client


def call_vsellm_text(
        *,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[float] = None,
        api_kwargs: Optional[Dict[str, Any]] = None,
        request_timeout: float = 180.0,
        proxy_cfg_path: str | None = None,
        proxy_url: str | None = None,
) -> Tuple[Optional[str], Dict[str, int], Any]:
    client = _get_client(proxy_cfg_path=proxy_cfg_path, proxy_url=proxy_url, request_timeout=request_timeout)
    return client.call_text(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        api_kwargs=api_kwargs,
    )
def call_vsellm_embeddings(
        *,
        model: str,
        inputs: str | List[str],
        api_kwargs: Optional[Dict[str, Any]] = None,
        request_timeout: float = 180.0,
        proxy_cfg_path: Optional[str] = None,
        proxy_url: Optional[str] = None,
) -> Tuple[List[List[float]], Dict[str, int], Any]:
    client = _get_client(
        proxy_cfg_path=proxy_cfg_path,
        request_timeout=request_timeout,
    )

    return client.call_embeddings(
        model=model,
        inputs=inputs,
        proxy_url=proxy_url,
        api_kwargs=api_kwargs,
    )


def call_vsellm_image(
        *,
        model: str,
        system_prompt: Optional[str],
        image_url: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        extra_text: Optional[str] = None,
        user_text: Optional[str] = None,
        api_kwargs: Optional[Dict[str, Any]] = None,
        request_timeout: float = 180.0,
        proxy_cfg_path: str | None = None,
        proxy_url: str | None = None,
) -> Tuple[Optional[str], Dict[str, int], Any]:
    client = _get_client(proxy_cfg_path=proxy_cfg_path, proxy_url=proxy_url, request_timeout=request_timeout)
    return client.call_image(
        model=model,
        system_prompt=system_prompt,
        image_url=image_url,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        extra_text=extra_text,
        user_text=user_text,
        api_kwargs=api_kwargs,
    )


def call_vsellm_audio(
        *,
        model: str,
        system_prompt: Optional[str],
        audio_url: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        extra_text: Optional[str] = None,
        user_text: Optional[str] = None,
        api_kwargs: Optional[Dict[str, Any]] = None,
        request_timeout: float = 180.0,
        proxy_cfg_path: str | None = None,
        proxy_url: str | None = None,
) -> Tuple[Optional[str], Dict[str, int], Any]:
    client = _get_client(proxy_cfg_path=proxy_cfg_path, proxy_url=proxy_url, request_timeout=request_timeout)
    return client.call_audio(
        model=model,
        system_prompt=system_prompt,
        audio_url=audio_url,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        extra_text=extra_text,
        user_text=user_text,
        api_kwargs=api_kwargs,
    )


def call_vsellm_video_full(
        *,
        model: str,
        video_url: str,
        prompt_text: str,
        temperature: Optional[float] = None,
        timeout_sec: int = 180,
        api_kwargs: Optional[Dict[str, Any]] = None,
        proxy_cfg_path: str | None = None,
        proxy_url: str | None = None,
) -> Tuple[Optional[str], Dict[str, int], Any]:
    if _is_gemini_model(model):
        timeout_sec = max(int(timeout_sec), int(GEMINI_MIN_TIMEOUT_SEC))

    client = _get_client(proxy_cfg_path=proxy_cfg_path, proxy_url=proxy_url, request_timeout=float(timeout_sec))
    return client.call_video_full(
        model=model,
        video_url=video_url,
        prompt_text=prompt_text,
        temperature=temperature,
        timeout_sec=timeout_sec,
        api_kwargs=api_kwargs,
    )


def _image_user_text(
        system_prompt: Optional[str],
        user_text: Optional[str],
        extra_text: Optional[str],
) -> Optional[str]:
    """Text-часть user-сообщения для image-запроса (или None = image-only).

    - user_text задан → используем его;
    - иначе при заданном system_prompt → None (задача в system, text-часть убираем);
    - иначе → заглушка "Проанализируй изображение согласно инструкции.".
    extra_text (если есть) дописывается как дополнительные текстовые данные.
    """
    if user_text:
        base: Optional[str] = str(user_text).strip()
    elif system_prompt:
        base = None
    else:
        base = "Проанализируй изображение согласно инструкции."
    extra = str(extra_text).strip() if extra_text else ""
    if not extra:
        return base
    extra_block = "Вот дополнительные текстовые данные для анализа:\n" + extra
    if base:
        return base + "\n\n" + extra_block
    return extra_block


def _build_image_messages(
        system_prompt: Optional[str],
        user_text: Optional[str],
        extra_text: Optional[str],
        image_url: str,
) -> List[Dict[str, Any]]:
    """Собирает messages для image-запроса: system (если есть) + user(text? + image).

    image_url может быть TagMe URL или data URL — для data URL скачивание не делается.
    Text-часть user-сообщения формируется через _image_user_text (может отсутствовать → image-only).
    """
    if isinstance(image_url, str) and image_url.strip().startswith("data:image/"):
        data_url = image_url.strip()
    else:
        data_url = _download_image_as_data_url_max_compress(image_url)

    resolved_user_text = _image_user_text(system_prompt, user_text, extra_text)

    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    user_content: List[Dict[str, Any]] = []
    if resolved_user_text:
        user_content.append({"type": "text", "text": resolved_user_text})
    user_content.append({"type": "image_url", "image_url": {"url": data_url}})
    messages.append({"role": "user", "content": user_content})
    return messages


def _download_audio_bytes(url: str, timeout_sec: float = AUDIO_DOWNLOAD_TIMEOUT_SEC) -> bytes:
    r = requests.get(url, timeout=float(timeout_sec), verify=False)
    r.raise_for_status()
    return r.content or b""


def _audio_format_from_ref(audio_ref: str) -> str:
    """Определяет формат аудио по data-URL mime или расширению. По умолчанию wav."""
    s = (audio_ref or "").strip().lower()
    if s.startswith("data:audio/"):
        fmt = s.split("data:audio/", 1)[1].split(";", 1)[0]
        if fmt:
            return fmt
    for ext in ("flac", "mp3", "ogg", "aac", "wav"):
        if s.endswith("." + ext) or ("." + ext + "?") in s:
            return ext
    return "wav"


def _audio_ref_to_input_audio(audio_ref: str) -> Dict[str, str]:
    """data:audio URL или http URL → часть input_audio (base64)."""
    s = (audio_ref or "").strip()
    if s.startswith("data:audio/"):
        if ";base64," not in s:
            raise ValueError("Not a base64 data:audio URL")
        b64 = s.split(";base64,", 1)[1]
        return {"data": b64, "format": _audio_format_from_ref(s)}
    raw = _download_audio_bytes(s)
    if not raw:
        raise RuntimeError("Downloaded audio is empty")
    b64 = base64.b64encode(raw).decode("ascii")
    return {"data": b64, "format": _audio_format_from_ref(s)}


def _audio_user_text(
        system_prompt: Optional[str],
        user_text: Optional[str],
        extra_text: Optional[str],
) -> Optional[str]:
    """Text-часть user-сообщения для audio-запроса (или None = audio-only)."""
    if user_text:
        base: Optional[str] = str(user_text).strip()
    elif system_prompt:
        base = None
    else:
        base = "Проанализируйте аудио согласно инструкции."
    extra = str(extra_text).strip() if extra_text else ""
    if not extra:
        return base
    extra_block = "Вот дополнительные текстовые данные для анализа:\n" + extra
    if base:
        return base + "\n\n" + extra_block
    return extra_block


def _build_audio_messages(
        system_prompt: Optional[str],
        user_text: Optional[str],
        extra_text: Optional[str],
        audio_url: str,
) -> List[Dict[str, Any]]:
    """Собирает messages для audio-запроса: system (если есть) + user(text? + audio)."""
    input_audio = _audio_ref_to_input_audio(audio_url)
    resolved_user_text = _audio_user_text(system_prompt, user_text, extra_text)

    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    user_content: List[Dict[str, Any]] = []
    if resolved_user_text:
        user_content.append({"type": "text", "text": resolved_user_text})
    user_content.append({"type": "input_audio", "input_audio": input_audio})
    messages.append({"role": "user", "content": user_content})
    return messages


def _call_gigahf_qwen_image(
        *,
        model: str,
        system_prompt: Optional[str],
        image_url: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra_text: Optional[str] = None,
        user_text: Optional[str] = None,
        timeout_sec: float = 180.0,
) -> Dict[str, Any]:
    """
    Специальный путь для Qwen*-VL через GigaHF:
    https://tagme.pd.sberdevices.ru/x/aihf/llm/v1/chat/completions
    """
    base_url = os.getenv("TAGME_GIGAHF_BASE_URL", "https://tagme.sberdevices.ru").rstrip("/")
    url = f"{base_url}/x/aihf/llm/v1/chat/completions"

    proxy_cfg_path = GIGAHF_PROXY_CFG_PATH
    token = get_proxy_token(proxy_cfg_path)

    messages = _build_image_messages(system_prompt, user_text, extra_text, image_url)

    payload: Dict[str, Any] = {"model": model, "messages": messages}
    if temperature is not None:
        payload["temperature"] = float(temperature)
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "accept": "application/json",
    }

    _log_request_config_once(route="GIGAHF", url=url, model=model, payload=payload)

    try:
        _call_t0 = time.time()
        r = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=(60.0, float(timeout_sec)),
            verify=False,
        )
        _call_elapsed = time.time() - _call_t0

        raw_body = r.text

        if r.status_code >= 400:
            logger.error(
                "GIGAHF IMAGE FAILED | status={} body={}",
                r.status_code,
                raw_body,
            )

        r.raise_for_status()

        try:
            j = r.json()
        except Exception:
            logger.exception("GIGAHF RESPONSE IS NOT VALID JSON | body={}", raw_body)
            raise

    except Exception:
        logger.exception("GIGAHF IMAGE CALL EXCEPTION")
        raise

    text = _extract_text_from_gigahf_json(j)
    usage = _extract_usage_from_json(j)

    _log_image_parsed(route="GIGAHF", text=text, usage=usage, raw=j, elapsed=_call_elapsed)

    return {"raw_text": text or "", "answer": text or "", "usage": usage, "raw": j}


def _call_gigahf_image(
        *,
        model: str,
        system_prompt: Optional[str],
        image_url: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra_text: Optional[str] = None,
        user_text: Optional[str] = None,
        timeout_sec: float = 180.0,
) -> Dict[str, Any]:
    """
    Общий путь для image-моделей через GigaHF/AIHF.
    Использует ту же схему, что и старый Qwen*-VL путь, но не требует
    заранее поддерживать полный список моделей.
    """
    return _call_gigahf_qwen_image(
        model=model,
        system_prompt=system_prompt,
        image_url=image_url,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_text=extra_text,
        user_text=user_text,
        timeout_sec=timeout_sec,
    )


def _call_gigahf_text(
        *,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout_sec: float = 180.0,
) -> Dict[str, Any]:
    base_url = os.getenv("TAGME_GIGAHF_BASE_URL", "https://tagme.sberdevices.ru").rstrip("/")
    url = f"{base_url}/x/aihf/llm/v1/chat/completions"

    proxy_cfg_path = GIGAHF_PROXY_CFG_PATH
    token = get_proxy_token(proxy_cfg_path)

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }

    if temperature is not None:
        payload["temperature"] = float(temperature)

    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "accept": "application/json",
    }

    r = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=float(timeout_sec),
        verify=False,
    )

    r.raise_for_status()
    j = r.json()

    text = _extract_text_from_gigahf_json(j)
    usage = _extract_usage_from_json(j)

    return {
        "raw_text": text or "",
        "answer": text or "",
        "usage": usage,
        "raw": j,
    }

def _image_ref_to_bytes_for_upload(image_ref: str) -> tuple[bytes, str, str]:
    """
    Поддерживает:
    - data:image/...;base64,...
    - http/https URL

    Возвращает:
    - image bytes
    - filename
    - mime_type
    """
    s = (image_ref or "").strip()
    if not s:
        raise ValueError("Empty image_ref")

    if s.startswith("data:image/") and ";base64," in s:
        header, b64 = s.split(";base64,", 1)
        mime_type = header.replace("data:", "").strip().lower()

        if "png" in mime_type:
            filename = "img.png"
        elif "webp" in mime_type:
            filename = "img.webp"
        elif "jpeg" in mime_type or "jpg" in mime_type:
            filename = "img.jpg"
            mime_type = "image/jpeg"
        else:
            filename = "img.jpg"
            mime_type = "image/jpeg"

        return base64.b64decode(b64), filename, mime_type

    if s.startswith("http://") or s.startswith("https://"):
        r = requests.get(s, timeout=IMG_DOWNLOAD_TIMEOUT_SEC, verify=False)
        r.raise_for_status()

        lower = s.lower()
        if lower.endswith(".png"):
            filename = "img.png"
            mime_type = "image/png"
        elif lower.endswith(".webp"):
            filename = "img.webp"
            mime_type = "image/webp"
        else:
            filename = "img.jpg"
            mime_type = "image/jpeg"

        return r.content or b"", filename, mime_type

    raise ValueError(f"Unsupported image_ref format: {s[:100]}")


def _upload_file_to_tagme_proxy(
        *,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        proxy_cfg_path: str | None = None,
        files_url: str | None = None,
        timeout_sec: float = 180.0,
) -> str:
    """
    Загружает файл в TagMe proxy /files и возвращает file_id.
    Аналог giga.upload_file(f), но через proxy endpoint:
    https://tagme.sberdevices.ru/ai/llm/v1/files
    """
    if not file_bytes:
        raise RuntimeError("_upload_file_to_tagme_proxy: file_bytes is empty")

    cfg_path = proxy_cfg_path or DEFAULT_PROXY_CFG_PATH
    token = get_proxy_token(cfg_path)
    target_files_url = (files_url or TAGME_DIRECT_FILES_URL).strip()

    headers = {
        "Authorization": f"Bearer {token}",
        "accept": "application/json",
    }

    files = {
        "file": (filename, BytesIO(file_bytes), mime_type),
    }

    # purpose добавляем как наиболее типичный формат для /files.
    # Если endpoint его игнорирует — ок. Если требует — он будет на месте.
    data = {
        "purpose": "general",
    }

    logger.info(
        "Uploading image to TagMe proxy files endpoint: url={} filename={} mime={} bytes={}",
        target_files_url,
        filename,
        mime_type,
        len(file_bytes),
    )

    r = requests.post(
        target_files_url,
        headers=headers,
        files=files,
        data=data,
        verify=False,
        timeout=(60.0, float(timeout_sec)),
    )

    if r.status_code >= 400:
        logger.error(
            "TagMe proxy file upload failed: status={} url={} body={}",
            r.status_code,
            target_files_url,
            r.text[:4000],
        )

    r.raise_for_status()

    uploaded = r.json()
    logger.info("TagMe proxy file upload response: {}", uploaded)

    file_id = (
        uploaded.get("id_")
        or uploaded.get("id")
        or uploaded.get("file_id")
        or uploaded.get("uid")
    )

    if not file_id:
        raise RuntimeError(f"TagMe proxy file upload returned no file id: {uploaded}")

    return str(file_id)


def _call_tagme_gigachat_image_proxy(
        *,
        model: str,
        system_prompt: Optional[str],
        image_url: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        extra_text: Optional[str] = None,
        user_text: Optional[str] = None,
        api_kwargs: Optional[Dict[str, Any]] = None,
        timeout_sec: float = 180.0,
        proxy_cfg_path: str | None = None,
        chat_url: str | None = None,
        files_url: str | None = None,
) -> Dict[str, Any]:
    """
    GigaChat image через TagMe proxy по схеме как IFT:

    1. Картинку загружаем в:
       POST https://tagme.sberdevices.ru/ai/llm/v1/files

    2. В chat/completions отправляем:
       messages=[{
           "role": "user",
           "content": prompt,
           "attachments": [file_id]
       }]
    """

    if not image_url:
        raise ValueError("_call_tagme_gigachat_image_proxy: image_url is required")

    # 1. Готовим bytes картинки
    img_bytes, filename, mime_type = _image_ref_to_bytes_for_upload(image_url)

    # 2. Загружаем картинку в /files
    file_id = _upload_file_to_tagme_proxy(
        file_bytes=img_bytes,
        filename=filename,
        mime_type=mime_type,
        proxy_cfg_path=proxy_cfg_path,
        files_url=files_url,
        timeout_sec=timeout_sec,
    )

    logger.info("TagMe proxy uploaded image file_id={}", file_id)

    # 3. Собираем текст промпта
    prompt_parts: List[str] = []

    if system_prompt:
        prompt_parts.append(str(system_prompt).strip())

    if user_text:
        prompt_parts.append(str(user_text).strip())

    if extra_text:
        prompt_parts.append(str(extra_text).strip())

    prompt = "\n\n".join([p for p in prompt_parts if p]).strip()

    if not prompt:
        prompt = "Опиши изображение."

    # 4. Собираем messages в GigaChat-native формате attachments
    messages: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": prompt,
            "attachments": [file_id],
        }
    ]

    if api_kwargs is None:
        api_kwargs = {}

    api_kwargs["profanity_check"] = False

    target_chat_url = (chat_url or TAGME_DIRECT_URL).strip()

    logger.info(
        "Calling GigaChat image through TagMe proxy with attachments: model={} file_id={} chat_url={}",
        model,
        file_id,
        target_chat_url,
    )

    # 5. Отправляем обычный chat/completions, но уже с attachments
    text, usage, raw = call_vsellm_text(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        api_kwargs=api_kwargs,
        request_timeout=float(timeout_sec),
        proxy_cfg_path=proxy_cfg_path,
        proxy_url=target_chat_url,
    )

    return {
        "raw_text": text or "",
        "answer": text or "",
        "usage": usage,
        "raw": raw,
    }


def call_vsellm(
        *,
        model: str,
        service: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        user_text: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[float] = None,
        image: Optional[bool] = None,
        video: Optional[bool] = None,
        audio: Optional[bool] = None,
        image_url: Optional[str] = None,
        video_url: Optional[str] = None,
        audio_url: Optional[str] = None,
        extra_text: Optional[str] = None,
        timeout_sec: int = 180,
        api_kwargs: Optional[Dict[str, Any]] = None,
        proxy_cfg_path: str | None = None,
        proxy_url: str | None = None,
        enable_thinking: Optional[bool] = None,
        web_search: bool = False,
        url_context: bool = False,
        # Управление reasoning (только service="vsellm"): none/low/medium/high/xhigh/max.
        # None/пусто -> ничего не добавляется, поведение прежнее.
        reasoning_effort: Optional[str] = None,
        # Одна строка данных (dict / pandas.Series). Если задана, то в image_url / video_url /
        # audio_url можно передать НАЗВАНИЕ СТОЛБЦА — ссылка возьмётся из row[<столбец>].
        row: Optional[Any] = None,
) -> Dict[str, Any]:
    image_url = _clean_media_url(_resolve_media_ref(image_url, row))
    video_url = _clean_media_url(_resolve_media_ref(video_url, row))
    audio_url = _clean_media_url(_resolve_media_ref(audio_url, row))

    # Для Gemini гарантуем таймаут не меньше GEMINI_MIN_TIMEOUT_SEC (15 мин),
    # даже если вызывающий код передал меньшее значение.
    if _is_gemini_model(model):
        timeout_sec = max(float(timeout_sec), GEMINI_MIN_TIMEOUT_SEC)

    service_norm = (service or "").strip().lower()

    force_gigahf = service_norm == "gigahf"
    force_ift = service_norm == "ift"
    force_vsellm = service_norm == "vsellm"
    force_direct = service_norm == "direct"

    supported_services = {"", "gigahf", "ift", "vsellm", "direct"}
    if service_norm not in supported_services:
        raise ValueError(
            f"Unsupported service={service!r}. "
            f"Supported services: gigahf, ift, vsellm, direct"
        )

    if force_gigahf:
        _gigahf_base = (os.getenv("TAGME_GIGAHF_BASE_URL") or "https://tagme.sberdevices.ru").rstrip("/")
        logger.info(
            "GIGAHF SERVICE ROUTE: model={} image={} video={} url={}",
            model,
            bool(image),
            bool(video),
            f"{_gigahf_base}/x/aihf/llm/v1/chat/completions",
        )

        if video:
            raise NotImplementedError(
                "service='gigahf' пока поддержан только для text/image. "
                "Для video нужно отдельно добавить GigaHF video route."
            )

        if image:
            if not image_url:
                raise ValueError("call_vsellm(service='gigahf', image=True): image_url is required")

            return _call_gigahf_image(
                model=model,
                system_prompt=system_prompt,
                image_url=image_url,
                temperature=temperature,
                max_tokens=20000,
                extra_text=extra_text,
                user_text=user_text,
                timeout_sec=float(timeout_sec),
            )

        if messages is None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_text or ""})

        return _call_gigahf_text(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_sec=float(timeout_sec),
        )

    if force_ift:
        # Для IFT TagMe используем отдельный cfg/token.
        # Это касается только вызова модели и загрузки файлов в IFT proxy.
        # Внешняя отправка разметки в TagMe должна оставаться на обычном DEFAULT_PROXY_CFG_PATH
        # и не должна неявно переключаться на .ml_config_ift.
        ift_proxy_cfg_path = proxy_cfg_path or TAGME_IFT_PROXY_CFG_PATH

        logger.info(
            "IFT SERVICE ROUTE: model={} image={} video={} url={} cfg_path={}",
            model,
            bool(image),
            bool(video),
            TAGME_IFT_URL,
            ift_proxy_cfg_path,
        )

        if video:
            raise NotImplementedError(
                "service='ift' пока поддержан только для text/image. "
                "Для video нужно отдельно добавить IFT video route."
            )

        if image:
            if not image_url:
                raise ValueError("call_vsellm(service='ift', image=True): image_url is required")

            return _call_tagme_gigachat_image_proxy(
                model=model,
                system_prompt=system_prompt,
                image_url=image_url,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                extra_text=extra_text,
                user_text=user_text,
                api_kwargs=api_kwargs,
                timeout_sec=float(timeout_sec),
                proxy_cfg_path=ift_proxy_cfg_path,
                chat_url=TAGME_IFT_URL,
                files_url=TAGME_IFT_FILES_URL,
            )

        # IFT text route: передаём role:system + role:user раздельно.
        if messages is None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": (user_text or "") + ("\n\n" + extra_text if extra_text else "")})

        logger.info(
            "IFT TEXT REQUEST CONTENT | model={} messages={}",
            model,
            json.dumps(messages, ensure_ascii=False),
        )

        if api_kwargs is None:
            api_kwargs = {}
        api_kwargs["profanity_check"] = False

        if enable_thinking is not None:
            api_kwargs["reasoning"] = enable_thinking
            api_kwargs["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

        logger.info(
            "IFT TEXT REQUEST API_KWARGS | model={} api_kwargs={}",
            model,
            json.dumps(api_kwargs, ensure_ascii=False, default=str),
        )

        text, usage, raw = call_vsellm_text(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            api_kwargs=api_kwargs,
            request_timeout=float(timeout_sec),
            proxy_cfg_path=ift_proxy_cfg_path,
            proxy_url=TAGME_IFT_URL,
        )

        # IFT text route: иногда raw-ответ есть, но стандартный extractor не видит текст.
        # Тогда внешний цикл разметки получает пустой answer/raw_text и запускает модель повторно.
        # Достаем текст еще раз универсальным extractor'ом и явно возвращаем complete=True,
        # если текст найден.
        text = text or _extract_text_from_any_json(raw)
        finish_reason = _extract_finish_reason_from_json(raw)
        complete = _is_complete_response(raw, text)
        reasoning = _extract_reasoning_from_json(raw)

        logger.info(
            "IFT TEXT PARSED | answer_len={} complete={} finish_reason={} raw_keys={}",
            len(text or ""),
            complete,
            finish_reason,
            list(raw.keys()) if isinstance(raw, dict) else type(raw),
        )

        logger.info(
            "IFT TEXT RESPONSE CONTENT | model={} answer={}",
            model,
            text or "",
        )

        if not text:
            logger.error(
                "IFT TEXT EMPTY ANSWER | raw_response_full={}",
                json.dumps(raw, ensure_ascii=False, indent=2) if isinstance(raw, dict) else raw,
            )

        return {
            "raw_text": text or "",
            "answer": text or "",
            "think": reasoning,
            "complete": complete,
            "finish_reason": finish_reason,
            "usage": usage,
            "raw": raw,
        }

    if force_vsellm or force_direct:
        selected_proxy_url = (
            TAGME_LLM_PROXY_URL
            if force_vsellm
            else TAGME_DIRECT_URL
        )

        logger.info(
            "EXPLICIT SERVICE ROUTE: service={} model={} image={} video={} audio={} url={}",
            service_norm,
            model,
            bool(image),
            bool(video),
            bool(audio),
            selected_proxy_url,
        )

        # ============================================================
        # VIDEO
        # ============================================================
        if video:
            if not video_url:
                raise ValueError(
                    f"call_vsellm(service={service_norm!r}, video=True): "
                    "video_url is required"
                )

            prompt_text = (system_prompt or "").strip()

            if user_text:
                prompt_text = (
                                  prompt_text + "\n\n"
                                  if prompt_text
                                  else ""
                              ) + str(user_text).strip()

            if extra_text:
                prompt_text = (
                                  prompt_text + "\n\n"
                                  if prompt_text
                                  else ""
                              ) + str(extra_text).strip()

            text, usage, raw = call_vsellm_video_full(
                model=model,
                video_url=video_url,
                prompt_text=prompt_text,
                temperature=temperature,
                timeout_sec=timeout_sec,
                api_kwargs=api_kwargs,
                proxy_cfg_path=proxy_cfg_path,
                proxy_url=selected_proxy_url,
            )

            return {
                "raw_text": text or "",
                "answer": text or "",
                "usage": usage,
                "raw": raw,
            }

        # ============================================================
        # IMAGE
        # ============================================================
        if image:
            if not image_url:
                raise ValueError(
                    f"call_vsellm(service={service_norm!r}, image=True): "
                    "image_url is required"
                )

            # Для direct + TagMe/GigaChat image сохраняем корректную
            # схему с предварительной загрузкой файла и attachments.
            #
            # Для vsellm специально не применяем этот путь:
            # картинка отправляется через image_url/base64 в x/ai/llm.
            if force_direct and model in TAGME_MODELS:
                return _call_tagme_gigachat_image_proxy(
                    model=model,
                    system_prompt=system_prompt,
                    image_url=image_url,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    extra_text=extra_text,
                    user_text=user_text,
                    api_kwargs=api_kwargs,
                    timeout_sec=float(timeout_sec),
                    proxy_cfg_path=proxy_cfg_path,
                    chat_url=TAGME_DIRECT_URL,
                    files_url=TAGME_DIRECT_FILES_URL,
                )

            text, usage, raw = call_vsellm_image(
                model=model,
                system_prompt=system_prompt,
                image_url=image_url,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                extra_text=extra_text,
                user_text=user_text,
                api_kwargs=api_kwargs,
                request_timeout=float(timeout_sec),
                proxy_cfg_path=proxy_cfg_path,
                proxy_url=selected_proxy_url,
            )

            return {
                "raw_text": text or "",
                "answer": text or "",
                "usage": usage,
                "raw": raw,
            }

        # ============================================================
        # AUDIO
        # ============================================================
        if audio:
            if not audio_url:
                raise ValueError(
                    f"call_vsellm(service={service_norm!r}, audio=True): "
                    "audio_url is required"
                )

            text, usage, raw = call_vsellm_audio(
                model=model,
                system_prompt=system_prompt,
                audio_url=audio_url,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                extra_text=extra_text,
                user_text=user_text,
                api_kwargs=api_kwargs,
                request_timeout=float(timeout_sec),
                proxy_cfg_path=proxy_cfg_path,
                proxy_url=selected_proxy_url,
            )

            return {
                "raw_text": text or "",
                "answer": text or "",
                "usage": usage,
                "raw": raw,
            }

        # ============================================================
        # TEXT
        # ============================================================
        if messages is None:
            messages = []

            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt,
                })

            messages.append({
                "role": "user",
                "content": user_text or "",
            })

        api_kwargs = dict(api_kwargs or {})

        # Для direct-маршрута сохраняем текущую настройку TagMe-моделей.
        if force_direct and model in TAGME_MODELS:
            api_kwargs["profanity_check"] = False

        # ==== ОБРАБОТКА WEB_SEARCH (ТОЛЬКО VseLLM) ====
        if web_search:
            tool_config = get_web_search_config(model)
            if tool_config:
                # Добавляем tools (гарантируем существование api_kwargs)
                api_kwargs = api_kwargs or {}
                if "tools" not in api_kwargs or api_kwargs["tools"] is None:
                    api_kwargs["tools"] = []
                api_kwargs["tools"].append(tool_config)
                # Gemini grounding: mark server tool usage for web_search
                model_lower = (model or "").lower()
                if "gemini" in model_lower:
                    su = api_kwargs.setdefault("server_tool_use", {})
                    su["web_search_requests"] = 1
                # Endpoint routing будет выполнен в методе call_text
            else:
                if os.getenv("VSLLM_WEBLOG", "0") in {"1", "true", "yes"}:
                    logger.warning(
                        "VSLLM web_search requested for model=%s but not supported", model
                    )

        # ==== ОБРАБОТКА URL_CONTEXT (ТОЛЬКО VseLLM, google/gemini) ====
        if url_context:
            url_ctx_config = get_url_context_config(model)
            if url_ctx_config:
                api_kwargs = api_kwargs or {}
                if "tools" not in api_kwargs or api_kwargs["tools"] is None:
                    api_kwargs["tools"] = []
                api_kwargs["tools"].append(url_ctx_config)
            else:
                if os.getenv("VSLLM_WEBLOG", "0") in {"1", "true", "yes"}:
                    logger.warning(
                        "VSLLM url_context requested for model=%s but not supported", model
                    )

        # ==== reasoning_effort (только service=vsellm и только если задан) ====
        reasoning_on = False
        if force_vsellm and normalize_reasoning_effort(reasoning_effort) is not None:
            reasoning_on = True
            r_params, r_min_tokens = build_reasoning_params(model, reasoning_effort)
            for k, v in r_params.items():
                api_kwargs.setdefault(k, v)
            if r_min_tokens and (max_tokens is None or int(max_tokens) < r_min_tokens):
                logger.info(
                    "reasoning_effort=max: max_tokens {} -> {} (model={})",
                    max_tokens, r_min_tokens, model,
                )
                max_tokens = r_min_tokens

        text, usage, raw = call_vsellm_text(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            api_kwargs=api_kwargs,
            request_timeout=float(timeout_sec),
            proxy_cfg_path=proxy_cfg_path,
            proxy_url=selected_proxy_url,
        )

        if reasoning_on:
            try:
                _msg = ((raw.get("choices") or [{}])[0] or {}).get("message") or {}
                logger.info(
                    "REASONING DIAG | model={} empty_answer={} finish_reason={!r} "
                    "completion_tokens={} reasoning_tokens={} max_tokens={} "
                    "message_keys={} top_keys={}",
                    model,
                    not bool(text),
                    _extract_finish_reason_from_json(raw),
                    (usage or {}).get("completion_tokens"),
                    _extract_reasoning_tokens_from_json(raw),
                    max_tokens,
                    sorted(_msg.keys()) if isinstance(_msg, dict) else type(_msg).__name__,
                    sorted(raw.keys()) if isinstance(raw, dict) else type(raw).__name__,
                )
            except Exception as _diag_err:
                logger.warning("REASONING DIAG failed: {}", _diag_err)

        result = {
            "raw_text": text or "",
            "answer": text or "",
            "think": _extract_reasoning_from_json(raw),
            "usage": usage,
            "raw": raw,
        }
        if reasoning_on:
            result["reasoning_tokens"] = _extract_reasoning_tokens_from_json(raw)
            result["reasoning"] = _extract_reasoning_text_from_json(raw)
            result["reasoning_effort_applied"] = _extract_applied_effort_from_json(raw)
        return result

    if image and model in TAGME_MODELS:
        if not image_url:
            raise ValueError("call_vsellm(image=True, model in TAGME_MODELS): image_url is required")

        logger.info("TAGME GIGACHAT IMAGE VIA PROXY FILE UPLOAD: model={}", model)

        return _call_tagme_gigachat_image_proxy(
            model=model,
            system_prompt=system_prompt,
            image_url=image_url,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            extra_text=extra_text,
            user_text=user_text,
            api_kwargs=api_kwargs,
            timeout_sec=float(timeout_sec),
            proxy_cfg_path=proxy_cfg_path,
        )

    if model in TAGME_MODELS:
        if messages is None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_text or ""})

        if api_kwargs is None:
            api_kwargs = {}
        api_kwargs["profanity_check"] = False

        text, usage, raw = call_vsellm_text(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            api_kwargs=api_kwargs,
            request_timeout=float(timeout_sec),
            proxy_cfg_path=proxy_cfg_path,
            proxy_url=TAGME_DIRECT_URL
        )
        logger.info("TAGME MODEL")
        return {"raw_text": text or "", "answer": text or "", "usage": usage, "raw": raw}

    if audio:
        if not audio_url:
            raise ValueError("call_vsellm(audio=True): audio_url is required")
        text, usage, raw = call_vsellm_audio(
            model=model,
            system_prompt=system_prompt,
            audio_url=audio_url,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            extra_text=extra_text,
            user_text=user_text,
            api_kwargs=api_kwargs,
            request_timeout=float(timeout_sec),
            proxy_cfg_path=proxy_cfg_path,
            proxy_url=proxy_url,
        )
        return {"raw_text": text or "", "answer": text or "", "usage": usage, "raw": raw}

    if video:
        if not video_url:
            raise ValueError("call_vsellm(video=True): video_url is required")

        prompt_text = (system_prompt or "").strip()
        if extra_text:
            prompt_text = (prompt_text + "\n\n" if prompt_text else "") + extra_text

        text, usage, raw = call_vsellm_video_full(
            model=model,
            video_url=video_url,
            prompt_text=prompt_text,
            temperature=temperature,
            timeout_sec=timeout_sec,
            api_kwargs=api_kwargs,
            proxy_cfg_path=proxy_cfg_path,
            proxy_url=proxy_url,
        )
        return {"raw_text": text or "", "answer": text or "", "usage": usage, "raw": raw}

    if image:
        if not image_url:
            raise ValueError("call_vsellm(image=True): image_url is required")

        if model in GIGAHF_QWEN_MODELS:
            result = _call_gigahf_qwen_image(
                model=model,
                system_prompt=system_prompt,
                image_url=image_url,
                temperature=temperature,
                max_tokens=10000,
                extra_text=extra_text,
                user_text=user_text,
                timeout_sec=float(timeout_sec),
            )
            return result

        if "qwen" in model:
            # Для Qwen через VseLLM: отправляем image + text как для Gemini
            messages = _build_image_messages(system_prompt, user_text, extra_text, image_url)

            # Web-search только для текстовых задач, не для изображений
            api_kwargs = dict(api_kwargs or {})

            if web_search and should_use_responses_endpoint(model):
                # Для OpenAI меняем эндпоинт на /responses
                if proxy_url and "/chat/completions" in proxy_url:
                    proxy_url = proxy_url.replace("/chat/completions", "/responses")

            text, usage, raw = call_vsellm_text(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                api_kwargs=api_kwargs,
                request_timeout=float(timeout_sec),
                proxy_cfg_path=proxy_cfg_path,
                proxy_url=proxy_url,
            )
            return {"raw_text": text or "", "answer": text or "", "usage": usage, "raw": raw}

        text, usage, raw = call_vsellm_image(
            model=model,
            system_prompt=system_prompt,
            image_url=image_url,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            extra_text=extra_text,
            user_text=user_text,
            api_kwargs=api_kwargs,
            request_timeout=float(timeout_sec),
            proxy_cfg_path=proxy_cfg_path,
            proxy_url=proxy_url,
        )
        return {"raw_text": text or "", "answer": text or "", "usage": usage, "raw": raw}

    if messages is None:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_text or ""})

    if model in GIGAHF_TEXT_MODELS:

        if messages is None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_text or ""})

        result = _call_gigahf_text(
            model=model,
            messages=messages,
            temperature=temperature,
            # max_tokens=max_tokens,
            timeout_sec=float(timeout_sec),
        )

        return result

    # ==== ОБРАБОТКА WEB_SEARCH (ТОЛЬКО VseLLM) ====
    api_kwargs = dict(api_kwargs or {})

    if web_search:
        tool_config = get_web_search_config(model)
        if tool_config:
            # Добавляем tools
            if "tools" not in api_kwargs or api_kwargs["tools"] is None:
                api_kwargs["tools"] = []
            api_kwargs["tools"].append(tool_config)

            # Для OpenAI меняем эндпоинт на /responses
            if should_use_responses_endpoint(model):
                if proxy_url and "/chat/completions" in proxy_url:
                    proxy_url = proxy_url.replace("/chat/completions", "/responses")
        else:
            # Модель не поддерживает web_search, но параметр был передан
            # Это не ошибка, просто игнорируем
            pass

    # ==== ОБРАБОТКА URL_CONTEXT (ТОЛЬКО VseLLM, google/gemini) ====
    if url_context:
        url_ctx_config = get_url_context_config(model)
        if url_ctx_config:
            if "tools" not in api_kwargs or api_kwargs["tools"] is None:
                api_kwargs["tools"] = []
            api_kwargs["tools"].append(url_ctx_config)
        else:
            # Модель не поддерживает url_context, но параметр был передан
            # Это не ошибка, просто игнорируем
            pass

    text, usage, raw = call_vsellm_text(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k,
        api_kwargs=api_kwargs,
        request_timeout=float(timeout_sec),
        proxy_cfg_path=proxy_cfg_path,
        proxy_url=proxy_url,
    )
    return {"raw_text": text or "", "answer": text or "", "usage": usage, "raw": raw}

def call_vsellm_chain(
        *,
        instr: str,
        user_input: str,
        model: str,
        service: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[float] = None,
        timeout_sec: int = 180,
        system_prompt: Optional[str] = None,
        api_kwargs: Optional[Dict[str, Any]] = None,
        proxy_cfg_path: str | None = None,
        proxy_url: str | None = None,

        # NEW: media для chain
        image: bool = False,
        video: bool = False,
        audio: bool = False,
        image_url: Optional[str] = None,
        video_url: Optional[str] = None,
        audio_url: Optional[str] = None,
        extra_text: Optional[str] = None,

        # first — media только на первом шаге chain
        # all — media на каждом шаге chain
        chain_media_mode: str = "first",

        # NEW: web_search для chain
        web_search: bool = False,

        # NEW: url_context для chain
        url_context: bool = False,

        # reasoning_effort (только service="vsellm"); None -> поведение прежнее
        reasoning_effort: Optional[str] = None,

        # Одна строка данных: в image_url / video_url / audio_url можно передать название столбца
        row: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Выполняет цепочку из блоков:

        <PROMPT_1>...</PROMPT_1>
        <PROMPT_2>...</PROMPT_2>

    Каждый шаг вызывается через текущий универсальный call_vsellm(),
    поэтому поддерживаются те же маршруты:

    - service="vsellm"
    - service="direct"
    - service="ift"
    - service="gigahf"
    - автоматический роутинг без service

    Теперь chain поддерживает не только text, но и image/video.

    chain_media_mode:
    - "first": image/video передаются только на первый шаг chain.
      Это режим по умолчанию, он дешевле и обычно логичнее.
    - "all": image/video передаются на каждый шаг chain.
    """

    # Импорт внутри функции позволяет не создавать лишние связи
    # между модулями при загрузке проекта.
    from utils.chain_processor import ChainProcessor

    if not isinstance(instr, str) or not instr.strip():
        raise ValueError(
            "call_vsellm_chain: instr должен содержать "
            "блоки <PROMPT_N>...</PROMPT_N>"
        )

    if not isinstance(user_input, str):
        user_input = json.dumps(
            user_input,
            ensure_ascii=False,
        )

    media_mode = str(chain_media_mode or "first").strip().lower()

    if media_mode not in {"first", "all"}:
        raise ValueError(
            f"Unsupported chain_media_mode={chain_media_mode!r}. "
            "Use 'first' or 'all'."
        )

    # Название столбца -> ссылка (если передан row); дальше в шаги уходят уже готовые ссылки.
    image_url = _clean_media_url(_resolve_media_ref(image_url, row))
    video_url = _clean_media_url(_resolve_media_ref(video_url, row))
    audio_url = _clean_media_url(_resolve_media_ref(audio_url, row))

    # Для Gemini гарантуем таймаут не меньше GEMINI_MIN_TIMEOUT_SEC (15 мин).
    if _is_gemini_model(model):
        timeout_sec = max(float(timeout_sec), GEMINI_MIN_TIMEOUT_SEC)

    chain_cfg: Dict[str, Any] = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "top_k": top_k,
        "timeout_sec": timeout_sec,
    }

    # Накопитель reasoning по шагам chain (заполняется только если задан reasoning_effort).
    chain_reasoning: Dict[str, Any] = {"tokens": 0, "texts": [], "applied": ""}

    def call_chain_step(
            *,
            prompt: str,
            model: str,
            service: Optional[str],
            system_prompt: Optional[str],
            cfg: Dict[str, Any],
            step: int,
            **step_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Адаптер между ChainProcessor и реальным call_vsellm().
        """

        logger.info(
            "CHAIN STEP CALL | step={} model={} service={} prompt_len={}",
            step,
            model,
            service or "default",
            len(prompt),
        )

        # Каждый PROMPT_N является отдельным пользовательским промптом.
        messages: List[Dict[str, Any]] = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })

        messages.append({
            "role": "user",
            "content": prompt,
        })

        use_media_this_step = (
            media_mode == "all"
            or int(step) == 1
        )

        step_image = bool(
            image
            and image_url
            and use_media_this_step
        )

        step_video = bool(
            video
            and video_url
            and use_media_this_step
        )

        step_audio = bool(
            audio
            and audio_url
            and use_media_this_step
        )

        logger.info(
            "CHAIN STEP MEDIA | step={} media_mode={} image={} video={} audio={} image_url={} video_url={} audio_url={}",
            step,
            media_mode,
            step_image,
            step_video,
            step_audio,
            bool(image_url),
            bool(video_url),
            bool(audio_url),
        )

        result = call_vsellm(
            model=model,
            service=service,
            messages=messages,
            system_prompt=system_prompt,
            user_text=prompt,
            temperature=cfg.get("temperature"),
            max_tokens=cfg.get("max_tokens"),
            top_p=cfg.get("top_p"),
            top_k=cfg.get("top_k"),

            image=step_image,
            video=step_video,
            audio=step_audio,
            image_url=image_url if step_image else None,
            video_url=video_url if step_video else None,
            audio_url=audio_url if step_audio else None,
            extra_text=extra_text if use_media_this_step else None,

            timeout_sec=int(
                cfg.get("timeout_sec") or 180
            ),
            api_kwargs=api_kwargs,
            proxy_cfg_path=proxy_cfg_path,
            proxy_url=proxy_url,
            web_search=web_search,
            url_context=url_context,
            reasoning_effort=reasoning_effort,
        )

        if "reasoning_tokens" in result:
            chain_reasoning["tokens"] += int(result.get("reasoning_tokens") or 0)
            r_txt = result.get("reasoning") or ""
            if r_txt:
                chain_reasoning["texts"].append(f"[step {step}] {r_txt}")
            applied = result.get("reasoning_effort_applied") or ""
            if applied:
                chain_reasoning["applied"] = applied

        answer = (
            result.get("answer")
            or result.get("raw_text")
            or ""
        )

        logger.info(
            "CHAIN STEP RESULT | step={} answer_len={} usage={}",
            step,
            len(str(answer)),
            result.get("usage") or {},
        )

        return result

    processor = ChainProcessor(
        instr=instr,
        model=model,
        service=service,
        cfg=chain_cfg,
        call_model=call_chain_step,

        # Важно: сюда не передаём весь instr.
        # instr содержит всю цепочку, а не системный промпт
        # конкретного шага.
        system_prompt=system_prompt,
    )

    result = processor.run_and_build_payload(
        user_input=user_input,
    )

    if normalize_reasoning_effort(reasoning_effort) is not None and (service or "").strip().lower() == "vsellm":
        result["reasoning_tokens"] = chain_reasoning["tokens"]
        result["reasoning"] = "\n---\n".join(chain_reasoning["texts"])
        result["reasoning_effort_applied"] = chain_reasoning["applied"]

    logger.info(
        "CHAIN COMPLETE | model={} service={} completed_steps={} usage={} media_mode={} image={} video={}",
        model,
        service or "default",
        (
            result.get("chain_meta", {})
            .get("completed_steps", 0)
        ),
        result.get("usage") or {},
        media_mode,
        bool(image and image_url),
        bool(video and video_url),
    )

    return result

def _add_usage(acc: dict, delta: dict) -> dict:
    """Складываем usage-словари по ключам."""
    if acc is None:
        acc = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for k in ["prompt_tokens", "completion_tokens", "total_tokens"]:
        acc[k] = acc.get(k, 0) + (delta.get(k, 0) or 0)
    return acc

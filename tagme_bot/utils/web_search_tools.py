# -*- coding: utf-8 -*-
"""
Утилиты для веб-поиска в VseLLM.
Только для текстовых задач с vsellm моделями.
"""

from typing import Any, Dict, Optional
from utils.model_endpoints import get_model_endpoint


def get_web_search_config(model: str) -> Optional[Dict[str, Any]]:
    """
    Возвращает конфигурацию web_search для поддерживаемых моделей.

    Args:
        model: Название модели (например, "google/gemini-3.6-flash")

    Returns:
        None если веб-поиск не поддерживается
        {"type": "..."} для поддерживаемых моделей
    """
    if not model:
        return None

    model_lower = model.lower()

    # Google/Gemini → server-side grounding
    if "google" in model_lower or "gemini" in model_lower:
        return {"type": "web_search_preview", "name": "web_search_preview"}

    # Anthropic/Claude → prompt-инъекция
    elif "anthropic" in model_lower or "claude" in model_lower:
        return {"type": "web_search", "name": "web_search"}

    # OpenAI GPT-5.6 → специальный эндпоинт
    elif "openai" in model_lower and ("gpt-5.6" in model_lower or "gpt-5" in model_lower):
        return {"type": "web_search", "name": "web_search"}

    return None


def get_url_context_config(model: str) -> Optional[Dict[str, Any]]:
    """
    Возвращает конфигурацию url_context для поддерживаемых моделей.

    url_context позволяет модели подгружать содержимое URL из ответа.
    Поддерживается только google/gemini (как в ask_copy.py).

    Args:
        model: Название модели (например, "google/gemini-3.6-flash")

    Returns:
        None если url_context не поддерживается
        {"type": "url_context"} для поддерживаемых моделей
    """
    if not model:
        return None

    model_lower = model.lower()

    # Google/Gemini → url_context (server-side)
    if "google" in model_lower or "gemini" in model_lower:
        return {"type": "url_context"}

    return None


def should_use_responses_endpoint(model: str) -> bool:
    """
    Проверяет, нужно ли использовать /responses эндпоинт для модели.

    Args:
        model: Название модели

    Returns:
        True для OpenAI GPT-5.6, False для остальных
    """
    if not model:
        return False
    model_lower = model.lower()
    return "openai" in model_lower and ("gpt-5.6" in model_lower or "gpt-5" in model_lower)


def extract_web_search_stats(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Извлекает статистику использования веб-поиска из результата.

    Args:
        result: Результат от call_vsellm

    Returns:
        Словарь со статистикой или пустой словарь
    """
    usage = result.get("usage", {})
    if not isinstance(usage, dict):
        return {}

    server_tool_use = usage.get("server_tool_use", {})
    if not isinstance(server_tool_use, dict):
        return {}

    web_search_requests = server_tool_use.get("web_search_requests", 0)
    return {"web_search_requests": web_search_requests} if web_search_requests > 0 else {}
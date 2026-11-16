# -*- coding: utf-8 -*-
"""
Упрощенный модуль для запуска нескольких поисковых движков из бота.

Основная функция:
    run_search_engines(query, search_engines, max_contexts=5)

Пример результата:
{
  "search_engines": [
    {
      "name": "perplexity",
      "contexts": ["...", "...", "...", "...", "..."]
    },
    {
      "name": "brave",
      "contexts": ["...", "...", "...", "...", "..."]
    }
  ]
}

Поддерживаемые search_engines:
- perplexity
- tavily
- tavily_gigasearch
- gigasearch
- yandex / yandex_search
- brave

Переменные окружения:
- PERPLEXITY_API_KEY
- TAVILY_API_KEY
- GIGASEARCH_API_KEY
- BRAVE_API_KEY
"""

from __future__ import annotations

import os
import json
import logging
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import urllib3
from dotenv import load_dotenv


load_dotenv(override=True)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


# ============================================================================
# Настройки
# ============================================================================

DEFAULT_MAX_CONTEXTS = 5
DEFAULT_COUNTRY = "ru"
DEFAULT_SEARCH_DEPTH = "basic"
DEFAULT_TIMEOUT_SEC = 60

SUPPORTED_SEARCH_ENGINES = {
    "perplexity",
    "exa",
    "tavily",
    "tavily_gigasearch",
    "gigasearch",
    "yandex",
    "yandex_search",
    "brave",
}


# ============================================================================
# Общие helpers
# ============================================================================

def _clean_text(text: str) -> str:
    return " ".join((text or "").split())


def _element_text(elem) -> str:
    if elem is None:
        return ""
    return _clean_text("".join(elem.itertext()))


def _normalize_engine_name(name: str) -> str:
    """
    Нормализует названия поисковиков.

    Во внешнем конфиге бота можно писать:
    - yandex

    Внутри старого кода сервис назывался:
    - yandex_search
    """

    name = str(name or "").strip().lower()

    if name == "yandex":
        return "yandex_search"

    return name


def _display_engine_name(name: str) -> str:
    """
    Какое имя возвращать наружу в payload.
    """

    if name == "yandex_search":
        return "yandex"

    return name


def _get_api_key(engine: str) -> Optional[str]:
    env_key_map = {
        "perplexity": "PERPLEXITY_API_KEY",
        "tavily": "TAVILY_API_KEY",
        "tavily_gigasearch": "GIGASEARCH_API_KEY",
        "gigasearch": "GIGASEARCH_API_KEY",
        "yandex_search": "GIGASEARCH_API_KEY",
        "brave": "BRAVE_API_KEY",
        "exa": "EXA_API_KEY"
    }

    env_key = env_key_map.get(engine)
    if not env_key:
        return None

    return os.getenv(env_key)


def _parse_yandex_xml(raw_xml: str) -> List[Dict[str, str]]:
    """
    Парсит XML-ответ Яндекса через GigaGateway.
    """

    if not raw_xml:
        return []

    root = ET.fromstring(raw_xml)
    results: List[Dict[str, str]] = []

    for doc in root.findall(".//doc"):
        title = _element_text(doc.find("title"))
        url = _clean_text(doc.findtext("url", ""))
        domain = _clean_text(doc.findtext("domain", ""))

        passages = doc.findall(".//passage")
        snippet = "\n".join(
            _element_text(passage)
            for passage in passages
            if _element_text(passage)
        )

        extended_text = _element_text(doc.find("./properties/extended-text"))

        results.append(
            {
                "title": title,
                "snippet": snippet,
                "url": url,
                "domain": domain,
                "extended_text": extended_text,
            }
        )

    return results


def _compact_context(*parts: str) -> str:
    """
    Собирает один контекст из title/snippet/content/url.
    Пустые части пропускает.
    """

    cleaned = [_clean_text(p) for p in parts if _clean_text(p)]
    return "\n".join(cleaned)


# ============================================================================
# Поисковые API
# ============================================================================

def _search_perplexity(
    query: str,
    api_key: str,
    max_contexts: int,
    country: str = DEFAULT_COUNTRY,
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
) -> List[str]:
    url = "https://api.perplexity.ai/search"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "query": query,
        "max_results": max_contexts,
        "max_tokens_per_page": 4096,
    }

    if country:
        payload["country"] = country.upper()

    response = requests.post(
        url,
        json=payload,
        headers=headers,
        timeout=timeout_sec,
        verify=False,
    )
    response.raise_for_status()

    data = response.json()
    results = data.get("results", []) or []

    contexts: List[str] = []

    for item in results[:max_contexts]:
        context = _compact_context(
            item.get("title", ""),
            item.get("snippet", ""),
            item.get("url", ""),
        )
        if context:
            contexts.append(context)

    return contexts


def _search_tavily(
    query: str,
    api_key: str,
    max_contexts: int,
    country: str = DEFAULT_COUNTRY,
    search_depth: str = DEFAULT_SEARCH_DEPTH,
    timeout_sec: int = 30,
) -> List[str]:
    url = "https://api.tavily.com/search"

    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": search_depth,
        "max_results": max_contexts,
        "include_answer": False,
        "include_raw_content": False,
        "include_images": False,
        "location": country.upper(),
    }

    response = requests.post(
        url,
        json=payload,
        timeout=timeout_sec,
        verify=False,
    )
    response.raise_for_status()

    data = response.json()
    results = data.get("results", []) or []

    contexts: List[str] = []

    for item in results[:max_contexts]:
        context = _compact_context(
            item.get("title", ""),
            item.get("content", ""),
            item.get("url", ""),
        )
        if context:
            contexts.append(context)

    return contexts


def _search_tavily_gigasearch(
    query: str,
    api_key: str,
    max_contexts: int,
    country: str = DEFAULT_COUNTRY,
    search_depth: str = DEFAULT_SEARCH_DEPTH,
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
) -> List[str]:
    """
    Tavily через GigaGateway.
    Для него используется GIGASEARCH_API_KEY.
    """

    url = "https://gigagateway.ift.eye.salute.ru/tavily/search"

    headers = {
        "accept": "application/json",
        "X-API-Key": api_key,
        "Content-Type": "application/json",
    }

    country_map = {
        "ru": "russia",
        "russia": "russia",
        "us": "united states",
        "usa": "united states",
        "uk": "united kingdom",
        "gb": "united kingdom",
    }

    tavily_country = country_map.get(str(country).lower(), country)

    payload = {
        "auto_parameters": True,
        "country": tavily_country,
        "include_answer": True,
        "include_images": False,
        "include_raw_content": False,
        "include_usage": True,
        "max_results": max_contexts,
        "query": query,
        "search_depth": search_depth,
        "timeout": timeout_sec,
        "topic": "general",
    }

    response = requests.post(
        url,
        json=payload,
        headers=headers,
        timeout=timeout_sec,
        verify=False,
    )
    response.raise_for_status()

    data = response.json()
    results = data.get("results", []) or []

    contexts: List[str] = []

    for item in results[:max_contexts]:
        context = _compact_context(
            item.get("title", ""),
            item.get("content", ""),
            item.get("url", ""),
        )
        if context:
            contexts.append(context)

    answer = data.get("answer", "")
    if answer and len(contexts) < max_contexts:
        contexts.append(_compact_context("ANSWER", answer))

    return contexts[:max_contexts]


def _search_gigasearch(
    query: str,
    api_key: str,
    max_contexts: int,
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
) -> List[str]:
    url = "https://gigagateway.ift.eye.salute.ru/v3/search"

    headers = {
        "accept": "application/json",
        "X-API-Key": api_key,
        "Content-Type": "application/json",
    }

    payload = {
        "query": query,
        "limit": max_contexts,
        "sources": [],
        "useCache": True,
        "search_mode": "fast",
    }

    response = requests.post(
        url,
        json=payload,
        headers=headers,
        timeout=timeout_sec,
        verify=False,
    )
    response.raise_for_status()

    data = response.json()
    results = data.get("documents", []) or []

    contexts: List[str] = []

    for item in results[:max_contexts]:
        context = _compact_context(
            item.get("title_raw", "") or item.get("title", ""),
            item.get("context_raw", "") or item.get("content", ""),
            item.get("url", ""),
        )
        if context:
            contexts.append(context)

    return contexts


def _search_yandex(
    query: str,
    api_key: str,
    max_contexts: int,
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
) -> List[str]:
    """
    Яндекс через GigaGateway.
    Для него используется GIGASEARCH_API_KEY.
    """

    url = "https://gigagateway.ift.eye.salute.ru/yandex/web-search"

    headers = {
        "accept": "application/json",
        "X-API-Key": api_key,
        "Content-Type": "application/json",
    }

    payload = {
        "query": {
            "searchType": "SEARCH_TYPE_RU",
            "queryText": query,
            "familyMode": "FAMILY_MODE_MODERATE",
            "page": "0",
            "fixTypoMode": "FIX_TYPO_MODE_ON",
        },
        "sortSpec": {
            "sortMode": "SORT_MODE_BY_RELEVANCE",
            "sortOrder": "SORT_ORDER_DESC",
        },
        "groupSpec": {
            "groupMode": "GROUP_MODE_FLAT",
            "groupsOnPage": str(max_contexts),
            "docsInGroup": "1",
        },
        "maxPassages": "5",
        "region": "213",
        "l10n": "LOCALIZATION_RU",
        "responseFormat": "FORMAT_XML",
        "useCache": True,
    }

    response = requests.post(
        url,
        json=payload,
        headers=headers,
        timeout=timeout_sec,
        verify=False,
    )
    response.raise_for_status()

    data = response.json()
    raw_xml = data.get("raw_text", "")

    parsed_results = _parse_yandex_xml(raw_xml)

    contexts: List[str] = []

    for item in parsed_results[:max_contexts]:
        context = _compact_context(
            item.get("title", ""),
            item.get("snippet", "") or item.get("extended_text", ""),
            item.get("url", ""),
        )
        if context:
            contexts.append(context)

    return contexts


def _search_exa(
    query: str,
    api_key: str,
    max_contexts: int,
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
    search_type: str = "auto",
    highlight_max_characters: int = 5000,
) -> List[str]:
    """
    Exa Search API.
    Для него используется EXA_API_KEY.

    Берем отдельные контексты из:
    1. highlights[0]
    2. summary
    3. text[:500]
    """

    url = "https://api.exa.ai/search"

    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
    }

    payload = {
        "query": query,
        "userLocation": "RU",
        "numResults": max_contexts,
        "type": search_type,
        "contents": {
            "highlights": {
                "maxCharacters": highlight_max_characters
            }
        },
    }

    response = requests.post(
        url,
        json=payload,
        headers=headers,
        timeout=timeout_sec,
        verify=False,
    )
    response.raise_for_status()

    data = response.json()
    results = data.get("results", []) or []

    contexts: List[str] = []

    for item in results[:max_contexts]:
        title = item.get("title", "")

        highlights = item.get("highlights", []) or []
        snippet = highlights[0] if highlights else ""

        if not snippet:
            snippet = item.get("summary", "")

        if not snippet:
            text_value = item.get("text", "")
            snippet = text_value[:500] if text_value else ""

        url = item.get("url", "")

        context = _compact_context(title, snippet, url)
        if context:
            contexts.append(context)

    return contexts

def _search_brave(
    query: str,
    api_key: str,
    max_contexts: int,
    country: str = DEFAULT_COUNTRY,
    timeout_sec: int = 30,
    search_lang: str = "ru",
    ui_lang: str = "ru-RU",
    safesearch: str = "moderate",
) -> List[str]:
    """
    Brave Search API.
    Для него используется BRAVE_API_KEY.
    """

    url = "https://api.search.brave.com/res/v1/web/search"

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }

    params = {
        "q": query,
        "count": max_contexts,
        "country": country.upper(),
        "search_lang": search_lang,
        "ui_lang": ui_lang,
        "safesearch": safesearch,
        "text_decorations": True,
        "spellcheck": True,
    }

    response = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=timeout_sec,
        verify=False,
    )
    response.raise_for_status()

    data = response.json()
    results = data.get("web", {}).get("results", []) or []

    contexts: List[str] = []

    for item in results[:max_contexts]:
        context = _compact_context(
            item.get("title", ""),
            item.get("description", ""),
            item.get("url", ""),
        )
        if context:
            contexts.append(context)

    return contexts


# ============================================================================
# Единый запуск одного поисковика
# ============================================================================

def _run_one_search_engine(
    query: str,
    engine: str,
    max_contexts: int,
    country: str,
    search_depth: str,
    timeout_sec: int,
    include_errors: bool,
) -> Dict[str, Any]:
    normalized_engine = _normalize_engine_name(engine)
    output_name = _display_engine_name(normalized_engine)

    result: Dict[str, Any] = {
        "name": output_name,
        "contexts": [],
    }

    if normalized_engine not in SUPPORTED_SEARCH_ENGINES:
        error_text = (
            f"Unknown search engine: {engine}. "
            f"Available: {sorted(SUPPORTED_SEARCH_ENGINES)}"
        )
        if include_errors:
            result["error"] = error_text
        return result

    api_key = _get_api_key(normalized_engine)

    if not api_key:
        error_text = f"API key not found for search engine: {output_name}"
        if include_errors:
            result["error"] = error_text
        return result

    try:
        if normalized_engine == "perplexity":
            contexts = _search_perplexity(
                query=query,
                api_key=api_key,
                max_contexts=max_contexts,
                country=country,
                timeout_sec=timeout_sec,
            )

        elif normalized_engine == "tavily":
            contexts = _search_tavily(
                query=query,
                api_key=api_key,
                max_contexts=max_contexts,
                country=country,
                search_depth=search_depth,
                timeout_sec=timeout_sec,
            )

        elif normalized_engine == "exa":
            contexts = _search_exa(
                query=query,
                api_key=api_key,
                max_contexts=max_contexts,
                timeout_sec=timeout_sec,
            )

        elif normalized_engine == "tavily_gigasearch":
            contexts = _search_tavily_gigasearch(
                query=query,
                api_key=api_key,
                max_contexts=max_contexts,
                country=country,
                search_depth=search_depth,
                timeout_sec=timeout_sec,
            )

        elif normalized_engine == "gigasearch":
            contexts = _search_gigasearch(
                query=query,
                api_key=api_key,
                max_contexts=max_contexts,
                timeout_sec=timeout_sec,
            )

        elif normalized_engine == "yandex_search":
            contexts = _search_yandex(
                query=query,
                api_key=api_key,
                max_contexts=max_contexts,
                timeout_sec=timeout_sec,
            )

        elif normalized_engine == "brave":
            contexts = _search_brave(
                query=query,
                api_key=api_key,
                max_contexts=max_contexts,
                country=country,
                timeout_sec=timeout_sec,
            )

        else:
            contexts = []

        result["contexts"] = contexts[:max_contexts]

    except Exception as e:
        logger.exception("Search engine failed: %s", output_name)

        if include_errors:
            result["error"] = f"{type(e).__name__}: {e}"

    return result


# ============================================================================
# Главная функция для импорта в бота
# ============================================================================

def run_search_engines(
    query: str,
    search_engines: List[str],
    max_contexts: int = DEFAULT_MAX_CONTEXTS,
    country: str = DEFAULT_COUNTRY,
    search_depth: str = DEFAULT_SEARCH_DEPTH,
    parallel: bool = True,
    max_workers: Optional[int] = None,
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
    include_errors: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Главная функция для бота.

    На вход:
        query:
            Текстовый запрос пользователя.

        search_engines:
            Список поисковиков из bot_cfg:
            ["perplexity", "tavily_gigasearch", "gigasearch", "yandex", "brave"]

        max_contexts:
            Сколько контекстов вернуть по каждому поисковику.
            По ТЗ — 5.

    На выход:
        {
          "search_engines": [
            {
              "name": "perplexity",
              "contexts": ["...", "...", "...", "...", "..."]
            },
            {
              "name": "brave",
              "contexts": ["...", "...", "...", "...", "..."]
            }
          ]
        }

    Важно:
    - функция не падает из-за ошибки одного поисковика;
    - если include_errors=True, ошибка попадет в поле "error";
    - порядок результатов соответствует порядку search_engines из конфига.
    """

    query = str(query or "").strip()

    if not query:
        raise ValueError("query is empty")

    if not search_engines:
        raise ValueError("search_engines list is empty")

    engines = [str(engine).strip() for engine in search_engines if str(engine).strip()]

    if not engines:
        raise ValueError("search_engines list is empty after normalization")

    if max_workers is None:
        max_workers = len(engines)

    results_by_engine: Dict[str, Dict[str, Any]] = {}

    if parallel and len(engines) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_engine = {
                executor.submit(
                    _run_one_search_engine,
                    query,
                    engine,
                    max_contexts,
                    country,
                    search_depth,
                    timeout_sec,
                    include_errors,
                ): engine
                for engine in engines
            }

            for future in as_completed(future_to_engine):
                engine = future_to_engine[future]

                try:
                    results_by_engine[engine] = future.result()
                except Exception as e:
                    normalized = _normalize_engine_name(engine)
                    output_name = _display_engine_name(normalized)

                    fallback_result = {
                        "name": output_name,
                        "contexts": [],
                    }

                    if include_errors:
                        fallback_result["error"] = f"{type(e).__name__}: {e}"

                    results_by_engine[engine] = fallback_result

    else:
        for engine in engines:
            results_by_engine[engine] = _run_one_search_engine(
                query=query,
                engine=engine,
                max_contexts=max_contexts,
                country=country,
                search_depth=search_depth,
                timeout_sec=timeout_sec,
                include_errors=include_errors,
            )

    ordered_results = [results_by_engine[engine] for engine in engines]

    return {
        "search_engines": ordered_results
    }


# ============================================================================
# Тестовый запуск
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_query = "Кто получил премию IEEE Фрэнка Розенблатта в 2010 году?"

    test_search_engines = [
        "exa"
    ]

    result = run_search_engines(
        query=test_query,
        search_engines=test_search_engines,
        max_contexts=5,
        country="ru",
        parallel=True,
        include_errors=True,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))
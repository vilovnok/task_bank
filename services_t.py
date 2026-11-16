#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Демонстрация запросов к LLM через TagMe.

Эндпоинты:
  direct  — https://tagme.sberdevices.ru/ai/llm/v1/chat/completions
  ift     — https://tagme.ift.delta.sbrf.ru/ai/llm/v1/chat/completions
  vsellm  — https://tagme.sberdevices.ru/x/ai/llm/v1/chat/completions
  aihf    — https://tagme.sberdevices.ru/x/aihf/llm/v1/chat/completions

  Чтобы посмотреть модели:
  https://tagme.sberdevices.ru/ai/llm/v1/models
"""
import json
import os
import warnings

import requests
from crowd_sdk.tagme import TagmeClientAdvanced

warnings.filterwarnings("ignore")  # убираем InsecureRequestWarning

# ─── эндпоинты ────────────────────────────────────────────────────────────────

ENDPOINTS = {
    # Прямой доступ к GigaChat, путь без /x/
    "direct": "https://tagme.sberdevices.ru/ai/llm/v1/chat/completions",

    # IFT-стенд TagMe — другой контур, другой cfg и токен
    "ift":    "https://tagme.ift.delta.sbrf.ru/ai/llm/v1/chat/completions",

    # vsellm-роутер — DeepSeek-R1, Qwen и др., путь /x/ai/
    "vsellm": "https://tagme.sberdevices.ru/x/ai/llm/v1/chat/completions",

    # aihf / GigaHF — тяжёлые open-source модели, путь /x/aihf/
    "aihf":   "https://tagme.sberdevices.ru/x/aihf/llm/v1/chat/completions",
}

# cfg-файлы для каждого стенда
CFG = {
    "direct": "~/configs/.ml_config.cfg",
    "ift":    "~/configs/.ml_config_ift.cfg",   # отдельный токен для IFT
    "vsellm": "~/configs/.ml_config.cfg",
    "aihf":   "~/configs/.ml_config.cfg",
}
# ─── получение токена ─────────────────────────────────────────────────────────

def get_token(service: str) -> str:
    """
    Получает Bearer-токен для нужного стенда через crowd_sdk.

    TagmeClientAdvanced читает cfg-файл (URL стенда, логин, пароль),
    делает запрос на /auth/token и сохраняет токен внутри client.client.token.
    """
    import os
    cfg_path = os.path.expanduser(CFG[service])
    print(f"[auth] cfg: {cfg_path}")

    client = TagmeClientAdvanced(cfg_path)   # авторизация на стенде
    token  = client.client.token             # Bearer-токен

    print(f"[auth] токен получен: …{token[-8:]}")
    return token

# ─── запрос к модели ──────────────────────────────────────────────────────────

def chat(service: str, model: str, user_message: str, system: str = "Ты полезный ассистент."):
    """
    Отправляет сообщение в модель через выбранный сервис.
    Печатает запрос и ответ.
    """
    url   = ENDPOINTS[service]
    token = get_token(service)

    payload = {
        "model":    model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_message},
        ],
        "temperature": 0.3,
        "max_tokens":  512,
    }
    # GigaChat-* (direct и ift) — выключаем проверку цензуры
    if service in ("direct", "ift"):
        payload["profanity_check"] = False

    print(f"\n{'═' * 60}")
    print(f"  Сервис  : {service}  →  {url}")
    print(f"  Модель  : {model}")
    print(f"\n📤 ЗАПРОС:")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=payload,
        verify=False,
        timeout=(60, 180),
    )
    resp.raise_for_status()
    data = resp.json()

    print(f"\n📥 ОТВЕТ (JSON):")
    print(json.dumps(data, ensure_ascii=False, indent=2))

    # Извлечение текста — aihf возвращает choices[0].messages[], остальные choices[0].message
    choices = data.get("choices") or []
    c0 = choices[0] if choices else {}
    if "messages" in c0:                            # формат aihf
        content = (c0["messages"][0] or {}).get("content", "")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content)
    else:                                           # стандартный OpenAI-формат
        content = (c0.get("message") or {}).get("content", "")

    print(f"\n💬 ОТВЕТ МОДЕЛИ:\n  {content}")
    usage = data.get("usage") or {}
    if usage:
        print(f"\n🔢 Токены: {usage}")
    print()

# ══════════════════════════════════════════════════════════════════════════════
#  ТЕСТОВЫЙ ЗАПУСК — меняй сервис, модель и вопрос прямо здесь
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── GigaChat через direct-эндпоинт (/ai/llm/...)
    # chat(
    #     service="direct",
    #     model="GigaChat-2-Max",
    #     user_message="Назови три популярных языка программирования и одну причину их популярности.",
    # )

    # ── GigaChat через IFT-стенд (tagme.ift.delta.sbrf.ru)
    chat(
        service="ift",
        model="Qwen3.8-397b",
        user_message="Что такое IFT-стенд?",
    )

    # ── DeepSeek-R1 через vsellm-роутер (/x/ai/llm/...)
    # chat(
    #     service="vsellm",
    #     model="minimax/minimax-m3",
    #     user_message="Сформируй одно слово",
    # )

    # ── DeepSeek-V3.2 через aihf (/x/aihf/llm/...)
    # chat(
    #     service="aihf",
    #     model="Qwen3.8-27B",
    #     user_message="Напиши функцию на Python для проверки палиндрома.",
    # )

"""RAG-инструмент бота (own-модель `rag_tool`).

Для записи: берёт единственное поле `search_*`, считает его эмбеддинг, ищет top_n
ближайших записей справочника (косинус), добавляет их `description` как `reference` во вход LLM
(и возвращает в `_meta["reference"]`) и вызывает LLM теми же средствами, что и обычная LLM-ветка бота.

Параметры `bot_cfg`: reference, emb_model, emb_service, top_n, llm_model, service (для LLM),
temperature, max_new_tokens, top_p.
Служебные данные (токены LLM, reasoning, think) возвращаются в ключе `_meta`.
"""
import asyncio
import json
import logging
import threading
from typing import Any, Dict, List, Optional

import numpy as np

from utils.config import check_service_policy
from utils.utils_bot import extract_json_and_think
from utils.vsellm_client import (
    TAGME_DIRECT_URL,
    TAGME_EMBEDDINGS_PROXY_URL,
    TAGME_IFT_PROXY_CFG_PATH,
    TAGME_IFT_URL,
    call_vsellm,
    call_vsellm_embeddings,
)

logger = logging.getLogger(__name__)

DEFAULT_EMB_MODEL = "Giga-Embeddings-instruct-10B-A1.8B-0826"
DEFAULT_EMB_SERVICE = "direct"
DEFAULT_TOP_N = 10
SEARCH_PREFIX = "search_"
TEXT_KEY = "description"
META_KEY = "_meta"
EMB_TIMEOUT = 180.0
LLM_TIMEOUT = 600

ORG_BY_STAND = {
    "cloud": "2bdf1e09-59e1-469c-b055-f81891aa3e64",
    "sigma": "2673e31b-8602-48c9-bf5f-8be78bfc6f47",
}

_RAG_INDEX: Optional[Dict[str, Any]] = None
_RAG_LOCK = threading.Lock()


def parse_file_id(reference: Any) -> str:
    """Id файла из id или ссылки вида https://.../files/<id>."""
    ref = str(reference or "").strip()
    if "files/" in ref:
        ref = ref.rsplit("files/", 1)[1]
        for sep in ("?", "#", "/"):
            ref = ref.split(sep, 1)[0]
    ref = ref.strip()
    if not ref:
        raise ValueError("rag_tool: не задан reference (id файла или ссылка)")
    return ref


def _route(service: str) -> tuple[str, Optional[str]]:
    """(url эмбеддингов, путь к cfg или None) для emb_service."""
    if service == "direct":
        return TAGME_DIRECT_URL.replace("/chat/completions", "/embeddings"), None
    if service == "ift":
        return TAGME_IFT_URL.replace("/chat/completions", "/embeddings"), TAGME_IFT_PROXY_CFG_PATH
    if service == "vsellm":
        return TAGME_EMBEDDINGS_PROXY_URL, None
    raise ValueError(f"rag_tool: неизвестный emb_service {service!r} (direct | ift | vsellm)")


async def _download(cfg_path: str, file_id: str, org_id: str) -> Any:
    from crowd_sdk.tagme import TagmeClientAdvanced

    client = TagmeClientAdvanced(cfg_path)
    try:
        return await client.download_dataset_file(file_id, organization_id=org_id)
    finally:
        await client.close()


def _build_index(raw: Any, emb_model: str) -> Dict[str, Any]:
    data = json.loads(raw)
    if not isinstance(data, list) or not data:
        raise ValueError("rag_tool: справочник должен быть непустым JSON-массивом")
    descriptions: List[str] = []
    vectors: List[List[float]] = []
    for i, rec in enumerate(data):
        if not isinstance(rec, dict) or not isinstance(rec.get(TEXT_KEY), str):
            raise ValueError(f"rag_tool: запись {i} без текстового поля {TEXT_KEY!r}")
        vec = rec.get(emb_model)
        if not isinstance(vec, list) or not vec:
            raise ValueError(f"rag_tool: в записи {i} нет эмбеддинга модели {emb_model!r}")
        descriptions.append(rec[TEXT_KEY])
        vectors.append(vec)
    mat = np.asarray(vectors, dtype=np.float32)
    if mat.ndim != 2:
        raise ValueError("rag_tool: эмбеддинги справочника имеют разную размерность")
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("rag_tool: в справочнике есть нулевые эмбеддинги")
    return {"descriptions": descriptions, "matrix": mat / norms}


def _get_index(stand: Any, file_id: str, emb_model: str) -> Dict[str, Any]:
    """Справочник грузится один раз за процесс; присваивается только после успешной загрузки."""
    global _RAG_INDEX
    if _RAG_INDEX is not None:
        return _RAG_INDEX
    with _RAG_LOCK:
        if _RAG_INDEX is not None:
            return _RAG_INDEX
        org_id = ORG_BY_STAND.get(getattr(stand, "name", ""))
        if not org_id:
            raise RuntimeError(f"rag_tool не поддерживается на стенде {getattr(stand, 'name', None)!r}")
        raw = asyncio.run(_download(stand.cfg_path, file_id, org_id))
        index = _build_index(raw, emb_model)
        _RAG_INDEX = index
        logger.info(
            f"rag_tool: справочник загружен | записей={len(index['descriptions'])} "
            f"| dim={index['matrix'].shape[1]}"
        )
        return _RAG_INDEX


def _find_search_text(input_data: Any) -> str:
    if not isinstance(input_data, dict):
        raise ValueError(f"rag_tool: input_data должен быть dict, получено {type(input_data).__name__}")
    keys = [k for k in input_data if isinstance(k, str) and k.startswith(SEARCH_PREFIX)]
    if len(keys) != 1:
        raise ValueError(f"rag_tool: нужно ровно одно поле {SEARCH_PREFIX}*, найдено {len(keys)}: {keys}")
    value = input_data[keys[0]]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"rag_tool: поле {keys[0]!r} пустое или не строка")
    return value


def _embed_query(text: str, emb_model: str, emb_service: str) -> tuple[np.ndarray, int]:
    url, cfg_path = _route(emb_service)
    embs, usage, _ = call_vsellm_embeddings(
        model=emb_model,
        inputs=[text],
        request_timeout=EMB_TIMEOUT,
        proxy_cfg_path=cfg_path,
        proxy_url=url,
    )
    if len(embs) != 1 or not embs[0]:
        raise RuntimeError(f"rag_tool: некорректный ответ эмбеддера: {len(embs)} векторов")
    tokens = int((usage or {}).get("total_tokens") or (usage or {}).get("prompt_tokens") or 0)
    vec = np.asarray(embs[0], dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm == 0:
        raise RuntimeError("rag_tool: нулевой эмбеддинг запроса")
    return vec / norm, tokens


def _parse_top_n(value: Any) -> int:
    if value is None or str(value).strip() == "":
        return DEFAULT_TOP_N
    n = int(value)
    if n < 1:
        raise ValueError(f"rag_tool: top_n должен быть >= 1, получено {n}")
    return n


def run_rag_tool(
        input_data: Any,
        *,
        stand: Any,
        bot_cfg: Dict[str, Any],
        instr: str = "",
        think: bool = False,
        web_search: bool = False,
        url_context: bool = False,
        reasoning_effort: Optional[str] = None,
) -> Dict[str, Any]:
    """Результат: JSON от LLM + `emb_tokens` + служебный `_meta` (в т.ч. `reference`). `input_data` не мутируется."""
    emb_model = str(bot_cfg.get("emb_model") or "").strip() or DEFAULT_EMB_MODEL
    emb_service = str(bot_cfg.get("emb_service") or "").strip().lower() or DEFAULT_EMB_SERVICE
    llm_model = str(bot_cfg.get("llm_model") or "").strip()
    if not llm_model:
        raise ValueError("rag_tool: не задан llm_model")
    llm_service = str(bot_cfg.get("service") or "").strip().lower() or None
    top_n = _parse_top_n(bot_cfg.get("top_n"))

    policy_error = check_service_policy(stand, emb_service)
    if policy_error:
        raise ValueError(f"rag_tool: emb_service: {policy_error}")

    file_id = parse_file_id(bot_cfg.get("reference"))
    search_text = _find_search_text(input_data)

    index = _get_index(stand, file_id, emb_model)
    query, emb_tokens = _embed_query(search_text, emb_model, emb_service)
    matrix = index["matrix"]
    if query.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"rag_tool: размерность запроса {query.shape[0]} != размерности справочника {matrix.shape[1]}"
        )
    scores = matrix @ query
    top_idx = np.argsort(-scores, kind="stable")[: min(top_n, len(scores))]
    reference = [index["descriptions"][int(i)] for i in top_idx]

    enriched = {**input_data, "reference": reference}
    messages = []
    if instr:
        messages.append({"role": "system", "content": instr})
    messages.append({"role": "user", "content": json.dumps(enriched, ensure_ascii=False, indent=2)})

    result = call_vsellm(
        model=llm_model,
        service=llm_service,
        system_prompt=instr,
        messages=messages,
        temperature=bot_cfg.get("temperature"),
        max_tokens=bot_cfg.get("max_new_tokens"),
        top_p=bot_cfg.get("top_p"),
        timeout_sec=LLM_TIMEOUT,
        enable_thinking=bool(think),
        web_search=web_search,
        url_context=url_context,
        reasoning_effort=reasoning_effort,
    )

    ans = (result.get("answer") or result.get("raw_text") or "").strip()
    payload_obj, think_txt = extract_json_and_think(ans)
    if payload_obj is None:
        raise ValueError(f"rag_tool: ответ LLM не является JSON: {ans[:300]!r}")
    payload = payload_obj if isinstance(payload_obj, dict) else {"result": payload_obj}

    think_parts = [t for t in (result.get("think") or "", think_txt or "") if t]
    payload["emb_tokens"] = emb_tokens
    payload[META_KEY] = {
        "usage": result.get("usage") or {},
        "reasoning_tokens": int(result.get("reasoning_tokens") or 0),
        "reasoning": result.get("reasoning") or "",
        "reasoning_effort_applied": result.get("reasoning_effort_applied") or "",
        "think": "\n".join(think_parts),
        "reference": reference,
    }

    return payload

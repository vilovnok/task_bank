# -*- coding: utf-8 -*-
"""
chatbot_classificator.py

Production-пайплайн для TagMe-бота:
- грузит уже обученный LogReg artifact из utils/models/trained_logreg_artifact.pkl;
- грузит справочник из utils/models/spravochnik_patched.xlsx;
- грузит prompt template из utils/models/chatbot_classificator_prompt.txt;
- для входного текста:
    1) получает embedding;
    2) прогоняет LogReg;
    3) если confidence >= threshold — возвращает LogReg класс;
    4) иначе собирает кандидатов LogReg + Guide Retrieval через RRF;
    5) вызывает LLM;
    6) возвращает TagMe payload:
       {"marks": [{"entityId": "<class>"}]}.

Дополнительный режим:
- если во входных данных передан think=true, итоговый payload дополнительно
  содержит источник ответа;
- если ответ получен от классификатора, возвращается его уверенность;
- если ответ получен от LLM, возвращается рассуждение модели;
- при think=false сохраняется прежний компактный формат ответа.

Важно:
- этот модуль НЕ обучает модель;
- обучение остается в отдельном экспериментальном скрипте;
- pkl, справочник и промпт лежат рядом в utils/models.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
import pickle
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from utils.vsellm_client import call_vsellm, call_vsellm_embeddings

load_dotenv()

logger = logging.getLogger(__name__)

# =============================================================================
# PATHS
# =============================================================================

UTILS_DIR = Path(__file__).resolve().parent
MODELS_DIR = UTILS_DIR / "models"

DEFAULT_ARTIFACT_PATH = MODELS_DIR / "trained_logreg_artifact.pkl"
DEFAULT_GUIDE_PATH = MODELS_DIR / "spravochnik_patched.xlsx"
DEFAULT_PROMPT_PATH = MODELS_DIR / "chatbot_classificator_prompt.txt"
DEFAULT_DEPLOY_CONFIG_PATH = MODELS_DIR / "cascade_deploy_config.json"

GUIDE_EMBEDDINGS_CACHE_PATH = (
        MODELS_DIR / "chatbot_classificator_guide_embeddings_cache.pkl"
)

# =============================================================================
# DEFAULT CONFIG
# =============================================================================

DEFAULT_LLM_MODEL = "DeepSeek-V4-Pro"
DEFAULT_DEPLOY_THRESHOLD = 0.6

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"

DEFAULT_LOGREG_TOP_K_FOR_LLM = 20
DEFAULT_GUIDE_RETRIEVAL_TOP_K_FOR_LLM = 20
DEFAULT_MAX_LLM_CANDIDATES = 40

DEFAULT_RRF_K = 60
DEFAULT_RRF_LOGREG_WEIGHT = 1.0
DEFAULT_RRF_GUIDE_WEIGHT = 1.0

DEFAULT_ALWAYS_INCLUDE_CLASS = "bank_axon"

DEFAULT_LLM_TEMPERATURE = 0.0

# Раньше было 64 токена, но теперь модель должна вернуть JSON
# с рассуждением и ответом, поэтому лимит увеличен.
DEFAULT_LLM_MAX_TOKENS = 2000
MIN_LLM_MAX_TOKENS_FOR_REASONING = 2000

DEFAULT_LLM_REQUEST_TIMEOUT = 180
DEFAULT_LLM_RETRIES = 3
DEFAULT_LLM_RETRY_DELAY = 1.0


# Инструкция добавляется в конец внешнего prompt template.
# Она применяется всегда, независимо от значения think во входном запросе.
LLM_JSON_RESPONSE_INSTRUCTION = """
ВАЖНО: выполняй классификацию по следующему порядку:

1. Сначала внимательно проанализируй запрос пользователя.
2. Сравни подходящие классы из переданного списка кандидатов:
   - сопоставь запрос с описаниями и примерами;
   - учти антипримеры;
   - исключи менее подходящие классы.
3. Сформулируй краткое обоснование выбора.
4. Только после проведённого анализа выбери один итоговый класс.

Верни ответ строго в виде одного валидного JSON-объекта:

{
  "reasoning": "Краткое обоснование выбора после сравнения подходящих классов",
  "answer": "Точное название одного итогового класса из списка кандидатов"
}

Требования:
- сначала сформируй reasoning и только затем выбери answer;
- reasoning должно объяснять, почему выбранный класс подходит лучше остальных;
- answer должен содержать точное название только одного класса из списка;
- не придумывай новые классы;
- не добавляй текст до или после JSON;
- не используй Markdown;
- не оборачивай JSON в ```json.
""".strip()

# =============================================================================
# GUIDE COLUMNS
# =============================================================================

GUIDE_INTENT_COL_CANDIDATES = [
    "Интент",
    "intent",
    "class",
    "Класс",
]

GUIDE_SHORT_DESC_COL_CANDIDATES = [
    "Описание (краткое)",
    "Описание",
    "Краткое описание",
    "description",
]

GUIDE_E2E_DESC_COL_CANDIDATES = [
    "е2е описание",
    "e2e описание",
    "е2е описание ",
    "e2e",
    "E2E описание",
]

GUIDE_EXAMPLES_COL_CANDIDATES = [
    "Примеры",
    "examples",
    "Позитивные примеры",
]

GUIDE_ANTI_EXAMPLES_COL_CANDIDATES = [
    "Антипримеры",
    "anti_examples",
    "Анти-примеры",
]

# =============================================================================
# BASIC UTILS
# =============================================================================


def safe_str(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)

    return str(value).strip()


def normalize_spaces(text: Any) -> str:
    return re.sub(r"\s+", " ", safe_str(text)).strip()


def normalize_label(label: Any) -> str:
    return normalize_spaces(label).lower()


def normalize_header(header: Any) -> str:
    text = safe_str(header).replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip().lower()


def find_column(
        df: pd.DataFrame,
        candidates: List[str],
        required: bool = True,
) -> Optional[str]:
    norm_to_original = {
        normalize_header(col): col
        for col in df.columns
    }

    for candidate in candidates:
        norm = normalize_header(candidate)

        if norm in norm_to_original:
            return norm_to_original[norm]

    for candidate in candidates:
        norm = normalize_header(candidate)

        for norm_col, original_col in norm_to_original.items():
            if norm and norm in norm_col:
                return original_col

    if required:
        raise ValueError(
            f"Колонка не найдена. Варианты: {candidates}. "
            f"Есть: {list(df.columns)}"
        )

    return None


def split_examples(value: Any) -> List[str]:
    text = safe_str(value)

    if not text:
        return []

    parts = re.split(r"[;\n]+", text)

    cleaned: List[str] = []
    seen: set = set()

    for part in parts:
        item = normalize_spaces(part)

        if not item or len(item) < 2:
            continue

        key = item.lower()

        if key not in seen:
            seen.add(key)
            cleaned.append(item)

    return cleaned


def text_hash(model: str, text: str) -> str:
    return hashlib.sha256(
        f"{model}\n{text}".encode("utf-8")
    ).hexdigest()


def load_pickle(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден pickle-файл: {path}"
        )

    with path.open("rb") as f:
        return pickle.load(f)


def save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    with tmp.open("wb") as f:
        pickle.dump(obj, f)

    tmp.replace(path)


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    if suffix == ".csv":
        return pd.read_csv(
            path,
            sep=None,
            engine="python",
        )

    raise ValueError(
        f"Неподдерживаемый формат справочника: {path}"
    )


def l2_normalize_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(
        x,
        dtype=np.float32,
    )

    norms = np.linalg.norm(
        x,
        axis=1,
        keepdims=True,
    )

    norms = np.maximum(
        norms,
        1e-12,
    )

    return x / norms


def cap_candidate_labels(
        labels: List[str],
        max_candidates: Optional[int],
        always_include_label: str,
) -> List[str]:
    if max_candidates is None:
        return labels

    if len(labels) <= max_candidates:
        return labels

    always_norm = normalize_label(
        always_include_label
    )

    trimmed = labels[:max_candidates]

    if (
            always_norm in labels
            and always_norm not in trimmed
    ):
        trimmed[-1] = always_norm

    result: List[str] = []
    seen: set = set()

    for label in trimmed:
        if label not in seen:
            result.append(label)
            seen.add(label)

    return result


def parse_bool(
        value: Any,
        default: bool = False,
) -> bool:
    """
    Преобразует входное значение в bool.

    Поддерживает:
    - True / False;
    - 1 / 0;
    - "true" / "false";
    - "yes" / "no";
    - "да" / "нет".
    """

    if value is None:
        return default

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    value_norm = safe_str(value).lower()

    if value_norm in {
        "1",
        "true",
        "yes",
        "y",
        "on",
        "да",
    }:
        return True

    if value_norm in {
        "0",
        "false",
        "no",
        "n",
        "off",
        "нет",
        "",
    }:
        return False

    return default


def extract_think_flag(input_data: Any) -> bool:
    """
    Извлекает параметр think из входных данных.

    Поддерживает, например:

    {
        "text": "с чего начинать инвестирование",
        "think": true
    }

    Также поддерживается think внутри data:

    {
        "data": {
            "think": true,
            ...
        }
    }
    """

    if isinstance(input_data, str):
        text = input_data.strip()

        try:
            parsed = json.loads(text)

            if isinstance(parsed, dict):
                return extract_think_flag(parsed)
        except Exception:
            pass

        return False

    if not isinstance(input_data, dict):
        return False

    if "think" in input_data:
        return parse_bool(
            input_data.get("think"),
            default=False,
        )

    data = input_data.get("data")

    if isinstance(data, dict) and "think" in data:
        return parse_bool(
            data.get("think"),
            default=False,
        )

    return False


def extract_first_json_object(
        text: str,
) -> Optional[Dict[str, Any]]:
    """
    Извлекает первый валидный JSON object из ответа LLM.

    Поддерживает:
    - чистый JSON;
    - JSON внутри ```json ... ```;
    - лишний текст до JSON;
    - лишний текст после JSON.
    """

    cleaned = safe_str(text).strip()

    if not cleaned:
        return None

    cleaned = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```\s*$",
        "",
        cleaned,
    )

    try:
        parsed = json.loads(cleaned)

        if isinstance(parsed, dict):
            return parsed

    except Exception:
        pass

    decoder = json.JSONDecoder()

    for idx, char in enumerate(cleaned):
        if char != "{":
            continue

        try:
            parsed, _ = decoder.raw_decode(
                cleaned[idx:]
            )

            if isinstance(parsed, dict):
                return parsed

        except Exception:
            continue

    return None


def parse_llm_structured_response(
        raw_response: Optional[str],
) -> Dict[str, Any]:
    """
    Разбирает ответ LLM формата:

    {
        "reasoning": "...",
        "answer": "..."
    }

    Если JSON разобрать не удалось, raw_response используется
    как answer для обратной совместимости со старым форматом.
    """

    raw_text = safe_str(raw_response)

    if not raw_text:
        return {
            "raw_response": "",
            "reasoning": "",
            "answer": "",
            "parsed_json": None,
        }

    parsed_json = extract_first_json_object(
        raw_text
    )

    if parsed_json is None:
        return {
            "raw_response": raw_text,
            "reasoning": "",
            "answer": raw_text,
            "parsed_json": None,
        }

    reasoning = safe_str(
        parsed_json.get("reasoning")
        or parsed_json.get("reason")
        or parsed_json.get("explanation")
        or parsed_json.get("analysis")
        or ""
    )

    answer = safe_str(
        parsed_json.get("answer")
        or parsed_json.get("label")
        or parsed_json.get("class")
        or parsed_json.get("intent")
        or ""
    )

    return {
        "raw_response": raw_text,
        "reasoning": reasoning,
        "answer": answer,
        "parsed_json": parsed_json,
    }


# =============================================================================
# INPUT EXTRACTION
# =============================================================================


def extract_input_text(input_data: Any) -> str:
    """
    Достает клиентскую реплику из входных данных TagMe.

    Поддерживает:
    - plain string;
    - JSON с ключами text / INPUT:text / input_text / question / Question;
    - JSON с data.messages.
    """

    if isinstance(input_data, str):
        text = normalize_spaces(input_data)

        try:
            parsed = json.loads(text)

            if isinstance(parsed, dict):
                return extract_input_text(parsed)

        except Exception:
            pass

        return text

    if isinstance(input_data, dict):
        priority_keys = [
            "INPUT:text",
            "input_text",
            "text",
            "client_text",
            "client_message",
            "message",
            "query",
            "user_query",
            "Question",
            "question",
        ]

        for key in priority_keys:
            value = input_data.get(key)

            if isinstance(value, str) and value.strip():
                return normalize_spaces(value)

        data = input_data.get("data")

        if isinstance(data, dict):
            messages = data.get("messages")

            if isinstance(messages, list):
                parts: List[str] = []

                for message in messages:
                    if not isinstance(message, dict):
                        continue

                    msg_text = safe_str(
                        message.get("text", "")
                    )

                    msg_from = safe_str(
                        message.get("from", "")
                    )

                    if msg_text:
                        if msg_from:
                            parts.append(
                                f"{msg_from}: {msg_text}"
                            )
                        else:
                            parts.append(msg_text)

                if parts:
                    return normalize_spaces(
                        "\n".join(parts)
                    )

        # Последний fallback: классифицируем JSON как текст.
        # Служебный параметр think из fallback-текста исключается.
        fallback_data = dict(input_data)
        fallback_data.pop("think", None)

        return normalize_spaces(
            json.dumps(
                fallback_data,
                ensure_ascii=False,
            )
        )

    return normalize_spaces(input_data)


# =============================================================================
# GUIDE PREP
# =============================================================================


def prepare_guide(
        raw_guide_df: pd.DataFrame,
) -> pd.DataFrame:
    intent_col = find_column(
        raw_guide_df,
        GUIDE_INTENT_COL_CANDIDATES,
    )

    short_desc_col = find_column(
        raw_guide_df,
        GUIDE_SHORT_DESC_COL_CANDIDATES,
        required=False,
    )

    e2e_desc_col = find_column(
        raw_guide_df,
        GUIDE_E2E_DESC_COL_CANDIDATES,
        required=False,
    )

    examples_col = find_column(
        raw_guide_df,
        GUIDE_EXAMPLES_COL_CANDIDATES,
        required=False,
    )

    anti_col = find_column(
        raw_guide_df,
        GUIDE_ANTI_EXAMPLES_COL_CANDIDATES,
        required=False,
    )

    rows: List[Dict[str, Any]] = []

    for _, row in raw_guide_df.iterrows():
        label_orig = safe_str(
            row.get(intent_col, "")
        )

        label_norm = normalize_label(
            label_orig
        )

        if not label_norm:
            continue

        rows.append({
            "guide_label": label_norm,
            "guide_label_original": label_orig,
            "short_desc": (
                safe_str(row.get(short_desc_col, ""))
                if short_desc_col
                else ""
            ),
            "e2e_desc": (
                safe_str(row.get(e2e_desc_col, ""))
                if e2e_desc_col
                else ""
            ),
            "examples_text": (
                safe_str(row.get(examples_col, ""))
                if examples_col
                else ""
            ),
            "anti_examples_text": (
                safe_str(row.get(anti_col, ""))
                if anti_col
                else ""
            ),
        })

    guide_df = pd.DataFrame(rows)

    if guide_df.empty:
        raise ValueError(
            "Справочник пустой или не удалось найти колонку интента"
        )

    merged: List[Dict[str, Any]] = []

    for label, group in guide_df.groupby(
            "guide_label",
            sort=True,
    ):
        first = group.iloc[0]

        merged.append({
            "guide_label": label,
            "guide_label_original": first[
                "guide_label_original"
            ],
            "short_desc": "\n".join(
                filter(
                    None,
                    group["short_desc"].tolist(),
                )
            ),
            "e2e_desc": "\n".join(
                filter(
                    None,
                    group["e2e_desc"].tolist(),
                )
            ),
            "examples_text": "\n".join(
                filter(
                    None,
                    group["examples_text"].tolist(),
                )
            ),
            "anti_examples_text": "\n".join(
                filter(
                    None,
                    group["anti_examples_text"].tolist(),
                )
            ),
        })

    return pd.DataFrame(
        merged
    ).reset_index(drop=True)


def make_full_card_text(
        row: pd.Series,
        max_examples: int = 20,
) -> str:
    parts = [
        f"Класс: {safe_str(row['guide_label_original'])}"
    ]

    if safe_str(row.get("short_desc", "")):
        parts.append(
            f"Краткое описание: "
            f"{safe_str(row['short_desc'])}"
        )

    if safe_str(row.get("e2e_desc", "")):
        parts.append(
            f"E2E описание: "
            f"{safe_str(row['e2e_desc'])}"
        )

    examples = split_examples(
        row.get("examples_text", "")
    )[:max_examples]

    if examples:
        parts.append(
            "Примеры:\n"
            + "\n".join(
                f"  - {example}"
                for example in examples
            )
        )

    anti = split_examples(
        row.get("anti_examples_text", "")
    )[:10]

    if anti:
        parts.append(
            "Антипримеры (НЕ подходят):\n"
            + "\n".join(
                f"  - {example}"
                for example in anti
            )
        )

    return "\n".join(parts)


def make_guide_retrieval_text(
        row: pd.Series,
        max_examples: int = 10,
) -> str:
    """
    Retrieval-текст без антипримеров.
    Это повторяет твою логику эксперимента v5.
    """

    parts = [
        f"Класс: "
        f"{safe_str(row['guide_label_original'])}",

        f"Нормализованный класс: "
        f"{safe_str(row['guide_label'])}",
    ]

    if safe_str(row.get("short_desc", "")):
        parts.append(
            f"Краткое описание: "
            f"{safe_str(row['short_desc'])}"
        )

    if safe_str(row.get("e2e_desc", "")):
        parts.append(
            f"E2E описание: "
            f"{safe_str(row['e2e_desc'])}"
        )

    examples = split_examples(
        row.get("examples_text", "")
    )[:max_examples]

    if examples:
        parts.append(
            "Примеры: "
            + " | ".join(examples)
        )

    return "\n".join(parts)


# =============================================================================
# EMBEDDINGS
# =============================================================================


def request_embeddings_batch(
        texts: List[str],
        embedding_model: str,
) -> List[List[float]]:
    """
    Получает embeddings через TagMe/VseLLM proxy.

    Авторизация выполняется через TagmeClientAdvanced:
    токен берётся из ~/configs/ml_config.cfg.

    Отдельный VSELLM_API_KEY больше не требуется.
    """

    clean = [
        normalize_spaces(text) or "[EMPTY_TEXT]"
        for text in texts
    ]

    if not clean:
        return []

    last_error: Optional[Exception] = None

    for attempt in range(1, 4):
        try:
            embeddings, usage, _raw = call_vsellm_embeddings(
                model=embedding_model,
                inputs=clean,
                request_timeout=180.0,
            )

            if len(embeddings) != len(clean):
                raise RuntimeError(
                    "Количество embeddings не совпадает "
                    "с количеством входных текстов: "
                    f"texts={len(clean)}, "
                    f"embeddings={len(embeddings)}"
                )

            if not embeddings:
                raise RuntimeError(
                    "Embeddings proxy вернул пустой список"
                )

            expected_dimension = len(
                embeddings[0]
            )

            if expected_dimension <= 0:
                raise RuntimeError(
                    "Первый embedding имеет нулевую размерность"
                )

            for index, embedding in enumerate(
                    embeddings
            ):
                if not isinstance(embedding, list):
                    raise RuntimeError(
                        f"Embedding {index} не является списком"
                    )

                if len(embedding) != expected_dimension:
                    raise RuntimeError(
                        f"Embedding {index} имеет размерность "
                        f"{len(embedding)}, ожидалась "
                        f"{expected_dimension}"
                    )

            logger.info(
                "Embeddings received through TagMe proxy | "
                "model=%s | texts=%s | dimension=%s | usage=%s",
                embedding_model,
                len(clean),
                expected_dimension,
                usage,
            )

            return embeddings

        except Exception as e:
            last_error = e

            logger.warning(
                "Embeddings attempt %s/3 failed: %s",
                attempt,
                e,
            )

            if attempt < 3:
                time.sleep(
                    0.3 * attempt
                )

    raise RuntimeError(
        "Не удалось получить embeddings через "
        "TagMe/VseLLM proxy после 3 попыток"
    ) from last_error


def get_embeddings_with_cache(
        texts: List[str],
        cache_path: Path,
        embedding_model: str,
) -> np.ndarray:
    if cache_path.exists():
        try:
            cache = load_pickle(
                cache_path
            )

            if not isinstance(cache, dict):
                cache = {}

        except Exception:
            cache = {}

    else:
        cache = {}

    result: List[Optional[List[float]]] = (
            [None] * len(texts)
    )

    missing_by_key: Dict[
        str,
        Dict[str, Any],
    ] = {}

    for idx, text in enumerate(texts):
        norm = (
                normalize_spaces(text)
                or "[EMPTY_TEXT]"
        )

        key = text_hash(
            embedding_model,
            norm,
        )

        cached = cache.get(key)

        if cached and "embedding" in cached:
            result[idx] = cached["embedding"]

        else:
            if key not in missing_by_key:
                missing_by_key[key] = {
                    "key": key,
                    "text": norm,
                    "indices": [],
                }

            missing_by_key[key][
                "indices"
            ].append(idx)

    missing = list(
        missing_by_key.values()
    )

    logger.info(
        "Guide embeddings: total=%s, new=%s, cached=%s",
        len(texts),
        len(missing),
        len(texts) - len(missing),
        )

    batch_size = 64

    for start in range(
            0,
            len(missing),
            batch_size,
    ):
        batch = missing[
            start:start + batch_size
        ]

        embeddings = request_embeddings_batch(
            texts=[
                record["text"]
                for record in batch
            ],
            embedding_model=embedding_model,
        )

        for record, embedding in zip(
                batch,
                embeddings,
        ):
            cache[record["key"]] = {
                "model": embedding_model,
                "text": record["text"],
                "embedding": embedding,
            }

            for idx in record["indices"]:
                result[idx] = embedding

        save_pickle(
            cache,
            cache_path,
        )

        time.sleep(0.3)

    if any(
            embedding is None
            for embedding in result
    ):
        raise RuntimeError(
            "После получения embeddings часть строк осталась пустой"
        )

    return np.array(
        result,
        dtype=np.float32,
    )


# =============================================================================
# MAIN PIPELINE
# =============================================================================


class ChatbotClassificatorPipeline:
    def __init__(
            self,
            artifact_path: Optional[Path] = None,
            guide_path: Optional[Path] = None,
            prompt_path: Optional[Path] = None,
            deploy_config_path: Optional[Path] = None,
    ) -> None:
        self.artifact_path = Path(
            os.getenv(
                "CHATBOT_CLASSIFICATOR_ARTIFACT_PATH",
                str(
                    artifact_path
                    or DEFAULT_ARTIFACT_PATH
                ),
            )
        )

        self.guide_path = Path(
            os.getenv(
                "CHATBOT_CLASSIFICATOR_GUIDE_PATH",
                str(
                    guide_path
                    or DEFAULT_GUIDE_PATH
                ),
            )
        )

        self.prompt_path = Path(
            os.getenv(
                "CHATBOT_CLASSIFICATOR_PROMPT_PATH",
                str(
                    prompt_path
                    or DEFAULT_PROMPT_PATH
                ),
            )
        )

        self.deploy_config_path = Path(
            os.getenv(
                "CHATBOT_CLASSIFICATOR_DEPLOY_CONFIG_PATH",
                str(
                    deploy_config_path
                    or DEFAULT_DEPLOY_CONFIG_PATH
                ),
            )
        )

        self.deploy_config = (
            self._load_deploy_config()
        )

        self.embedding_model = str(
            os.getenv(
                "CHATBOT_CLASSIFICATOR_EMBEDDING_MODEL",
                self.deploy_config.get(
                    "embedding_model",
                    DEFAULT_EMBEDDING_MODEL,
                ),
            )
        )

        self.threshold = float(
            os.getenv(
                "CHATBOT_CLASSIFICATOR_THRESHOLD",
                self.deploy_config.get(
                    "deploy_threshold",
                    DEFAULT_DEPLOY_THRESHOLD,
                ),
            )
        )

        llm_models = (
                self.deploy_config.get(
                    "llm_models"
                )
                or [DEFAULT_LLM_MODEL]
        )

        default_llm_model = (
            llm_models[0]
            if (
                    isinstance(llm_models, list)
                    and llm_models
            )
            else DEFAULT_LLM_MODEL
        )

        self.llm_model = str(
            os.getenv(
                "CHATBOT_CLASSIFICATOR_LLM_MODEL",
                default_llm_model,
            )
        )

        self.llm_temperature = float(
            os.getenv(
                "CHATBOT_CLASSIFICATOR_LLM_TEMPERATURE",
                self.deploy_config.get(
                    "llm_temperature",
                    DEFAULT_LLM_TEMPERATURE,
                ),
            )
        )

        configured_llm_max_tokens = int(
            os.getenv(
                "CHATBOT_CLASSIFICATOR_LLM_MAX_TOKENS",
                self.deploy_config.get(
                    "llm_max_tokens",
                    DEFAULT_LLM_MAX_TOKENS,
                ),
            )
        )

        # JSON с reasoning и answer не должен обрезаться
        # из-за старого лимита в 64 токена.
        self.llm_max_tokens = max(
            configured_llm_max_tokens,
            MIN_LLM_MAX_TOKENS_FOR_REASONING,
        )

        self.llm_request_timeout = int(
            float(
                os.getenv(
                    "CHATBOT_CLASSIFICATOR_LLM_REQUEST_TIMEOUT",
                    self.deploy_config.get(
                        "llm_request_timeout",
                        DEFAULT_LLM_REQUEST_TIMEOUT,
                    ),
                )
            )
        )

        self.llm_retries = int(
            os.getenv(
                "CHATBOT_CLASSIFICATOR_LLM_RETRIES",
                str(DEFAULT_LLM_RETRIES),
            )
        )

        self.logreg_top_k_for_llm = int(
            self.deploy_config.get(
                "logreg_top_k_for_llm",
                DEFAULT_LOGREG_TOP_K_FOR_LLM,
            )
        )

        self.guide_retrieval_top_k_for_llm = int(
            self.deploy_config.get(
                "guide_retrieval_top_k_for_llm",
                DEFAULT_GUIDE_RETRIEVAL_TOP_K_FOR_LLM,
            )
        )

        max_candidates_raw = os.getenv(
            "CHATBOT_CLASSIFICATOR_MAX_LLM_CANDIDATES",
            str(
                self.deploy_config.get(
                    "max_llm_candidates",
                    DEFAULT_MAX_LLM_CANDIDATES,
                )
            ),
        )

        if (
                str(max_candidates_raw)
                        .strip()
                        .lower()
                in {"none", "null", ""}
        ):
            self.max_llm_candidates = None

        else:
            self.max_llm_candidates = int(
                max_candidates_raw
            )

        self.include_all_unseen_by_logreg_train = bool(
            self.deploy_config.get(
                "include_all_unseen_by_logreg_train",
                False,
            )
        )

        self.rrf_k = int(
            self.deploy_config.get(
                "rrf_k",
                DEFAULT_RRF_K,
            )
        )

        self.rrf_logreg_weight = float(
            self.deploy_config.get(
                "rrf_logreg_weight",
                DEFAULT_RRF_LOGREG_WEIGHT,
            )
        )

        self.rrf_guide_weight = float(
            self.deploy_config.get(
                "rrf_guide_weight",
                DEFAULT_RRF_GUIDE_WEIGHT,
            )
        )

        self.always_include_class = str(
            self.deploy_config.get(
                "always_include_class",
                DEFAULT_ALWAYS_INCLUDE_CLASS,
            )
        )

        # "norm" — вернёт a0502.how_to_call_the_bank.
        # "original" — вернёт A0502.How_to_call_the_Bank.
        # Для твоего эксперимента и примера чаще нужен norm.
        self.output_label_mode = os.getenv(
            "CHATBOT_CLASSIFICATOR_OUTPUT_LABEL_MODE",
            "norm",
        ).strip().lower()

        self.artifact = (
            self._load_artifact()
        )

        self.logreg_model = (
            self.artifact["model"]
        )

        self.logreg_train_labels = set(
            str(label)
            for label in self.artifact.get(
                "train_labels",
                [],
            )
        )

        self.guide_df = (
            self._load_guide()
        )

        self.guide_by_label: Dict[
            str,
            pd.Series,
        ] = {
            str(row["guide_label"]): row
            for _, row in self.guide_df.iterrows()
        }

        self.original_by_label: Dict[
            str,
            str,
        ] = {
            str(row["guide_label"]): (
                    safe_str(
                        row["guide_label_original"]
                    )
                    or str(row["guide_label"])
            )
            for _, row in self.guide_df.iterrows()
        }

        self.valid_labels_norm = (
            self._build_valid_labels_norm()
        )

        self.prompt_template = (
            self._load_prompt_template()
        )

        self.guide_embeddings_norm = (
            self._load_guide_embeddings_norm()
        )

        logger.info(
            "ChatbotClassificator loaded | "
            "threshold=%s | "
            "llm_model=%s | "
            "llm_max_tokens=%s | "
            "guide_labels=%s | "
            "artifact=%s",
            self.threshold,
            self.llm_model,
            self.llm_max_tokens,
            len(self.guide_df),
            self.artifact_path,
        )

    def _load_deploy_config(
            self,
    ) -> Dict[str, Any]:
        if not self.deploy_config_path.exists():
            logger.warning(
                "Deploy config не найден, "
                "используются defaults: %s",
                self.deploy_config_path,
            )

            return {}

        with self.deploy_config_path.open(
                "r",
                encoding="utf-8",
        ) as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(
                "Deploy config должен быть JSON object: "
                f"{self.deploy_config_path}"
            )

        return data

    def _load_artifact(
            self,
    ) -> Dict[str, Any]:
        artifact = load_pickle(
            self.artifact_path
        )

        if (
                isinstance(artifact, dict)
                and "model" in artifact
        ):
            return artifact

        # На случай, если когда-нибудь сохранишь
        # в pkl только sklearn Pipeline.
        return {
            "model": artifact,
            "train_labels": self._get_model_classes(
                artifact
            ),
        }

    def _load_guide(
            self,
    ) -> pd.DataFrame:
        if not self.guide_path.exists():
            raise FileNotFoundError(
                f"Не найден справочник: {self.guide_path}"
            )

        raw_guide_df = read_table(
            self.guide_path
        )

        guide_df = prepare_guide(
            raw_guide_df
        )

        guide_df[
            "guide_retrieval_text"
        ] = guide_df.apply(
            make_guide_retrieval_text,
            axis=1,
        )

        return guide_df

    def _load_prompt_template(
            self,
    ) -> str:
        if not self.prompt_path.exists():
            raise FileNotFoundError(
                f"Не найден prompt-файл: {self.prompt_path}. "
                f"Создай chatbot_classificator_prompt.txt "
                f"с плейсхолдерами "
                f"{{classes_block}}, "
                f"{{class_names_str}}, "
                f"{{input_text}}."
            )

        template = self.prompt_path.read_text(
            encoding="utf-8"
        )

        required_placeholders = [
            "{classes_block}",
            "{class_names_str}",
            "{input_text}",
        ]

        missing = [
            placeholder
            for placeholder in required_placeholders
            if placeholder not in template
        ]

        if missing:
            raise ValueError(
                f"В prompt-файле не хватает "
                f"плейсхолдеров: {missing}. "
                f"Файл: {self.prompt_path}"
            )

        return template

    def _load_guide_embeddings_norm(
            self,
    ) -> np.ndarray:
        embeddings = get_embeddings_with_cache(
            texts=self.guide_df[
                "guide_retrieval_text"
            ].tolist(),
            cache_path=GUIDE_EMBEDDINGS_CACHE_PATH,
            embedding_model=self.embedding_model,
        )

        return l2_normalize_matrix(
            embeddings
        )

    def _build_valid_labels_norm(
            self,
    ) -> Dict[str, str]:
        valid: Dict[str, str] = {}

        for _, row in self.guide_df.iterrows():
            label_norm = str(
                row["guide_label"]
            )

            label_original = safe_str(
                row["guide_label_original"]
            )

            valid[
                normalize_label(label_original)
            ] = label_norm

            valid[
                normalize_label(label_norm)
            ] = label_norm

        return valid

    @staticmethod
    def _get_model_classes(
            model: Any,
    ) -> List[str]:
        if (
                hasattr(model, "named_steps")
                and "logreg" in model.named_steps
        ):
            return [
                str(class_name)
                for class_name
                in model.named_steps[
                    "logreg"
                ].classes_
            ]

        if hasattr(model, "classes_"):
            return [
                str(class_name)
                for class_name
                in model.classes_
            ]

        return []

    def _embed_query(
            self,
            text: str,
    ) -> np.ndarray:
        embeddings = request_embeddings_batch(
            texts=[text],
            embedding_model=self.embedding_model,
        )

        return np.array(
            embeddings,
            dtype=np.float32,
        )

    def _logreg_predict_with_proba(
            self,
            query_embedding: np.ndarray,
    ) -> Tuple[
        str,
        float,
        List[str],
        List[float],
    ]:
        proba = self.logreg_model.predict_proba(
            query_embedding
        )

        classes = self._get_model_classes(
            self.logreg_model
        )

        if not classes:
            raise RuntimeError(
                "Не удалось получить classes_ "
                "из LogReg модели"
            )

        row = proba[0]
        order = np.argsort(-row)

        ranked_labels = [
            str(classes[int(index)])
            for index in order
        ]

        ranked_scores = [
            float(row[int(index)])
            for index in order
        ]

        return (
            ranked_labels[0],
            ranked_scores[0],
            ranked_labels,
            ranked_scores,
        )

    def _rank_guides(
            self,
            query_embedding: np.ndarray,
    ) -> Tuple[
        List[str],
        List[float],
    ]:
        query_norm = l2_normalize_matrix(
            query_embedding
        )

        sim = (
                query_norm
                @ self.guide_embeddings_norm.T
        )

        guide_labels = self.guide_df[
            "guide_label"
        ].astype(str).tolist()

        order = np.argsort(-sim[0])

        ranked_labels = [
            guide_labels[int(index)]
            for index in order
        ]

        ranked_scores = [
            float(sim[0][int(index)])
            for index in order
        ]

        return (
            ranked_labels,
            ranked_scores,
        )

    def _select_llm_candidates(
            self,
            logreg_ranked_labels: List[str],
            logreg_scores: List[float],
            guide_ranked_labels: List[str],
            guide_scores: List[float],
    ) -> Tuple[
        List[pd.Series],
        List[Dict[str, Any]],
    ]:
        candidate_info: Dict[
            str,
            Dict[str, Any],
        ] = {}

        def ensure(
                label: str,
        ) -> Dict[str, Any]:
            if label not in candidate_info:
                candidate_info[label] = {
                    "label": label,
                    "sources": set(),
                    "rrf_score": 0.0,
                    "logreg_rank": None,
                    "logreg_score": None,
                    "guide_rank": None,
                    "guide_score": None,
                    "manual_reason": "",
                }

            return candidate_info[label]

        for rank, label in enumerate(
                logreg_ranked_labels[
                    :self.logreg_top_k_for_llm
                ],
                start=1,
        ):
            if label not in self.guide_by_label:
                continue

            info = ensure(label)

            info["sources"].add(
                "logreg_top_k"
            )

            info["logreg_rank"] = rank

            info["logreg_score"] = (
                float(logreg_scores[rank - 1])
                if rank - 1 < len(logreg_scores)
                else None
            )

            info["rrf_score"] += (
                    self.rrf_logreg_weight
                    / (self.rrf_k + rank)
            )

        for rank, label in enumerate(
                guide_ranked_labels[
                    :self.guide_retrieval_top_k_for_llm
                ],
                start=1,
        ):
            if label not in self.guide_by_label:
                continue

            info = ensure(label)

            info["sources"].add(
                "guide_retrieval_top_k"
            )

            info["guide_rank"] = rank

            info["guide_score"] = (
                float(guide_scores[rank - 1])
                if rank - 1 < len(guide_scores)
                else None
            )

            info["rrf_score"] += (
                    self.rrf_guide_weight
                    / (self.rrf_k + rank)
            )

        sorted_labels = sorted(
            candidate_info.keys(),
            key=lambda label: (
                -candidate_info[label][
                    "rrf_score"
                ],
                (
                    candidate_info[label][
                        "logreg_rank"
                    ]
                    if candidate_info[label][
                           "logreg_rank"
                       ] is not None
                    else 10**9
                ),
                (
                    candidate_info[label][
                        "guide_rank"
                    ]
                    if candidate_info[label][
                           "guide_rank"
                       ] is not None
                    else 10**9
                ),
                label,
            ),
        )

        result_labels: List[str] = []
        seen: set = set()

        for label in sorted_labels:
            if label not in seen:
                result_labels.append(label)
                seen.add(label)

        always_norm = normalize_label(
            self.always_include_class
        )

        if (
                always_norm in self.guide_by_label
                and always_norm not in seen
        ):
            info = ensure(always_norm)

            info["sources"].add(
                "always_include"
            )

            info["manual_reason"] = (
                "always_include"
            )

            result_labels.append(
                always_norm
            )

            seen.add(always_norm)

        if self.include_all_unseen_by_logreg_train:
            guide_labels = self.guide_df[
                "guide_label"
            ].astype(str).tolist()

            for label in guide_labels:
                if (
                        label
                        not in self.logreg_train_labels
                        and label not in seen
                ):
                    info = ensure(label)

                    info["sources"].add(
                        "unseen_by_logreg_train"
                    )

                    info["manual_reason"] = (
                        "unseen_by_logreg_train"
                    )

                    result_labels.append(
                        label
                    )

                    seen.add(label)

        result_labels = cap_candidate_labels(
            labels=result_labels,
            max_candidates=self.max_llm_candidates,
            always_include_label=(
                self.always_include_class
            ),
        )

        candidate_rows: List[
            pd.Series
        ] = []

        candidate_meta: List[
            Dict[str, Any]
        ] = []

        for label in result_labels:
            row = self.guide_by_label.get(
                label
            )

            if row is None:
                continue

            info = candidate_info.get(
                label,
                {},
            )

            sources = sorted(
                list(
                    info.get(
                        "sources",
                        set(),
                    )
                )
            )

            candidate_rows.append(row)

            candidate_meta.append({
                "label": label,
                "label_original": safe_str(
                    row.get(
                        "guide_label_original",
                        "",
                    )
                ),
                "sources": sources,
                "rrf_score": round(
                    float(
                        info.get(
                            "rrf_score",
                            0.0,
                        )
                    ),
                    8,
                ),
                "logreg_rank": info.get(
                    "logreg_rank"
                ),
                "logreg_score": (
                    round(
                        float(
                            info[
                                "logreg_score"
                            ]
                        ),
                        8,
                    )
                    if info.get(
                        "logreg_score"
                    ) is not None
                    else None
                ),
                "guide_rank": info.get(
                    "guide_rank"
                ),
                "guide_score": (
                    round(
                        float(
                            info[
                                "guide_score"
                            ]
                        ),
                        8,
                    )
                    if info.get(
                        "guide_score"
                    ) is not None
                    else None
                ),
                "manual_reason": info.get(
                    "manual_reason",
                    "",
                ),
            })

        return (
            candidate_rows,
            candidate_meta,
        )

    def _build_llm_prompt(
            self,
            input_text: str,
            candidate_rows: List[pd.Series],
    ) -> str:
        classes_block = "\n\n---\n\n".join(
            make_full_card_text(row)
            for row in candidate_rows
        )

        class_names_str = "\n".join(
            f"- {safe_str(row['guide_label_original'])}"
            for row in candidate_rows
        )

        prompt = self.prompt_template

        prompt = prompt.replace(
            "{classes_block}",
            classes_block,
        )

        prompt = prompt.replace(
            "{class_names_str}",
            class_names_str,
        )

        prompt = prompt.replace(
            "{input_text}",
            input_text,
        )

        # Инструкция с reasoning и answer добавляется всегда.
        prompt = (
                prompt.rstrip()
                + "\n\n"
                + LLM_JSON_RESPONSE_INSTRUCTION
        )

        return prompt

    def _call_vsellm_with_thinking(
            self,
            prompt: str,
    ) -> Dict[str, Any]:
        """
        Вызывает call_vsellm.

        Для DeepSeek-V4-Pro пытается передать:
            enable_thinking=True

        Поддерживает разные версии call_vsellm:
        - user_text=...;
        - messages=[...];
        - enable_thinking=True;
        - extra_body={"enable_thinking": True}.
        """

        base_kwargs: Dict[str, Any] = {
            "model": self.llm_model,
            "service": "ift",
            "temperature": self.llm_temperature,
            "max_tokens": self.llm_max_tokens,
            "timeout_sec": self.llm_request_timeout,
        }

        model_norm = self.llm_model.strip().lower()

        enable_thinking = model_norm.startswith(
            "deepseek-v4-pro"
        )

        try:
            signature = inspect.signature(
                call_vsellm
            )

            parameters = (
                signature.parameters
            )

            supports_kwargs = any(
                parameter.kind
                == inspect.Parameter.VAR_KEYWORD
                for parameter
                in parameters.values()
            )

        except Exception:
            parameters = {}
            supports_kwargs = True

        if "user_text" in parameters:
            base_kwargs["user_text"] = prompt

        elif "messages" in parameters:
            base_kwargs["messages"] = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]

        else:
            # Сохраняем совместимость с текущей версией,
            # где ранее использовался user_text.
            base_kwargs["user_text"] = prompt

        if enable_thinking:
            base_kwargs["api_kwargs"] = {"enable_thinking": True}

        try:
            return call_vsellm(
                **base_kwargs
            )

        except TypeError as first_error:
            # Fallback для версий call_vsellm,
            # у которых inspect.signature не отразил
            # реальную сигнатуру или используется wrapper.

            fallback_kwargs: Dict[str, Any] = {
                "model": self.llm_model,
                "service": "ift",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "temperature": (
                    self.llm_temperature
                ),
                "max_tokens": (
                    self.llm_max_tokens
                ),
                "timeout_sec": (
                    self.llm_request_timeout
                ),
            }

            if enable_thinking:
                fallback_kwargs[
                    "api_kwargs"
                ] = {"enable_thinking": True}

            try:
                return call_vsellm(
                    **fallback_kwargs
                )

            except TypeError:
                # Последний fallback без enable_thinking.
                # Нужен только для обратной совместимости
                # со старой версией клиента.
                fallback_kwargs.pop(
                    "enable_thinking",
                    None,
                )

                logger.warning(
                    "call_vsellm не принял "
                    "enable_thinking=True. "
                    "Выполняется совместимый вызов "
                    "без этого параметра. "
                    "Первая ошибка: %s",
                    first_error,
                )

                return call_vsellm(
                    **fallback_kwargs
                )

    def _call_llm_ift(
            self,
            prompt: str,
    ) -> Optional[Dict[str, Any]]:
        last_error: Optional[
            Exception
        ] = None

        for attempt in range(
                1,
                self.llm_retries + 1,
        ):
            try:
                result = (
                    self._call_vsellm_with_thinking(
                        prompt
                    )
                )

                raw_answer = (
                        result.get("answer")
                        or result.get("raw_text")
                        or ""
                )

                if not isinstance(
                        raw_answer,
                        str,
                ):
                    raw_answer = json.dumps(
                        raw_answer,
                        ensure_ascii=False,
                    )

                raw_answer = (
                    raw_answer.strip()
                )

                if not raw_answer:
                    return None

                return parse_llm_structured_response(
                    raw_answer
                )

            except Exception as e:
                last_error = e

                logger.warning(
                    "LLM call failed attempt "
                    "%s/%s: %s",
                    attempt,
                    self.llm_retries,
                    e,
                )

                if attempt < self.llm_retries:
                    time.sleep(
                        DEFAULT_LLM_RETRY_DELAY
                        * attempt
                    )

        logger.error(
            "LLM failed after retries: %s",
            last_error,
        )

        return None

    def _normalize_llm_response(
            self,
            raw_response: Optional[str],
    ) -> Optional[str]:
        if not raw_response:
            return None

        response_norm = normalize_label(
            raw_response.strip()
        )

        if response_norm in self.valid_labels_norm:
            return self.valid_labels_norm[
                response_norm
            ]

        for (
                label_orig_norm,
                label_norm,
        ) in self.valid_labels_norm.items():
            if (
                    label_orig_norm
                    and (
                    label_orig_norm
                    in response_norm
                    or response_norm
                    in label_orig_norm
            )
            ):
                return label_norm

        return None

    def _label_for_tagme(
            self,
            label_norm: str,
    ) -> str:
        if self.output_label_mode == "original":
            return self.original_by_label.get(
                label_norm,
                label_norm,
            )

        return label_norm

    def classify(
            self,
            input_data: Any,
    ) -> Dict[str, Any]:
        input_text = extract_input_text(
            input_data
        )

        if not input_text:
            fallback_label = normalize_label(
                self.always_include_class
            )

            return {
                "input_text": "",
                "final_label_norm": fallback_label,
                "final_entity_id": self._label_for_tagme(
                    fallback_label
                ),
                "route": "empty_input_fallback",
                "answer_source": "fallback",
                "classifier_confidence": None,
                "logreg_top1_proba": None,
                "llm_raw_response": None,
                "llm_reasoning": None,
                "llm_answer": None,
                "llm_parsed_label": None,
            }

        query_embedding = self._embed_query(
            input_text
        )

        (
            logreg_pred,
            top1_proba,
            logreg_ranked_labels,
            logreg_scores,
        ) = self._logreg_predict_with_proba(
            query_embedding=query_embedding,
        )

        high_confidence = (
                top1_proba >= self.threshold
        )

        if high_confidence:
            final_label = logreg_pred

            return {
                "input_text": input_text,
                "final_label_norm": final_label,
                "final_entity_id": self._label_for_tagme(
                    final_label
                ),
                "route": "logreg_high_conf",
                "answer_source": "classifier",
                "classifier_confidence": float(
                    top1_proba
                ),
                "logreg_top1_proba": float(
                    top1_proba
                ),
                "llm_raw_response": None,
                "llm_reasoning": None,
                "llm_answer": None,
                "llm_parsed_label": None,
            }

        (
            guide_ranked_labels,
            guide_scores,
        ) = self._rank_guides(
            query_embedding
        )

        (
            candidates,
            candidate_meta,
        ) = self._select_llm_candidates(
            logreg_ranked_labels=(
                logreg_ranked_labels
            ),
            logreg_scores=logreg_scores,
            guide_ranked_labels=(
                guide_ranked_labels
            ),
            guide_scores=guide_scores,
        )

        if not candidates:
            final_label = logreg_pred

            return {
                "input_text": input_text,
                "final_label_norm": final_label,
                "final_entity_id": self._label_for_tagme(
                    final_label
                ),
                "route": (
                    "no_llm_candidates_"
                    "fallback_to_logreg"
                ),
                "answer_source": "classifier",
                "classifier_confidence": float(
                    top1_proba
                ),
                "logreg_top1_proba": float(
                    top1_proba
                ),
                "llm_raw_response": None,
                "llm_reasoning": None,
                "llm_answer": None,
                "llm_parsed_label": None,
            }

        llm_prompt = self._build_llm_prompt(
            input_text=input_text,
            candidate_rows=candidates,
        )

        llm_result = self._call_llm_ift(
            llm_prompt
        )

        llm_raw_response = (
            llm_result.get("raw_response")
            if llm_result
            else None
        )

        llm_reasoning = (
            llm_result.get("reasoning")
            if llm_result
            else None
        )

        llm_answer = (
            llm_result.get("answer")
            if llm_result
            else None
        )

        llm_parsed_label = (
            self._normalize_llm_response(
                llm_answer
            )
        )

        if llm_parsed_label is not None:
            final_label = llm_parsed_label
            route = "llm"
            answer_source = "llm"
            classifier_confidence = None

        else:
            final_label = logreg_pred
            route = "llm_fallback_to_logreg"
            answer_source = "classifier"
            classifier_confidence = float(
                top1_proba
            )

        return {
            "input_text": input_text,
            "final_label_norm": final_label,
            "final_entity_id": self._label_for_tagme(
                final_label
            ),
            "route": route,
            "answer_source": answer_source,
            "classifier_confidence": (
                classifier_confidence
            ),
            "logreg_top1_proba": float(
                top1_proba
            ),
            "llm_raw_response": (
                llm_raw_response
            ),
            "llm_reasoning": llm_reasoning,
            "llm_answer": llm_answer,
            "llm_parsed_label": (
                llm_parsed_label
            ),
            "n_candidates": len(candidates),
            "candidate_meta": candidate_meta,
        }


# =============================================================================
# SINGLETON API FOR BOT
# =============================================================================

_PIPELINE: Optional[
    ChatbotClassificatorPipeline
] = None

_PIPELINE_LOCK = threading.Lock()


def get_chatbot_classificator_pipeline(
) -> ChatbotClassificatorPipeline:
    global _PIPELINE

    if _PIPELINE is not None:
        return _PIPELINE

    with _PIPELINE_LOCK:
        if _PIPELINE is None:
            _PIPELINE = (
                ChatbotClassificatorPipeline()
            )

    return _PIPELINE


def classify_chatbot_intent(
        input_data: Any,
) -> Dict[str, Any]:
    """
    При think=false или при отсутствии think возвращает
    прежний TagMe payload:

    {
      "form": {},
      "full_task": "<исходный текст клиента>",
      "marks": [
        {
          "entityId": "<class>"
        }
      ]
    }

    При think=true добавляются диагностические поля.

    Если ответ получен от классификатора:

    {
      "form": {},
      "full_task": "...",
      "marks": [
        {
          "entityId": "<class>"
        }
      ],
      "answer_source": "classifier",
      "classifier_confidence": 0.87
    }

    Если ответ получен от LLM:

    {
      "form": {},
      "full_task": "...",
      "marks": [
        {
          "entityId": "<class>"
        }
      ],
      "answer_source": "llm",
      "reasoning": "Почему модель выбрала этот класс"
    }
    """

    think = extract_think_flag(
        input_data
    )

    pipeline = (
        get_chatbot_classificator_pipeline()
    )

    result = pipeline.classify(
        input_data
    )

    payload: Dict[str, Any] = {
        "form": {},
        "full_task": result["input_text"],
        "marks": [
            {
                "entityId": result[
                    "final_entity_id"
                ]
            }
        ],
    }

    # При think=false сохраняется старый формат.
    if not think:
        return payload

    answer_source = result.get(
        "answer_source",
        "fallback",
    )

    payload["answer_source"] = (
        answer_source
    )

    if answer_source == "classifier":
        payload[
            "classifier_confidence"
        ] = result.get(
            "classifier_confidence"
        )

    elif answer_source == "llm":
        payload["reasoning"] = (
                result.get("llm_reasoning")
                or ""
        )

    return payload


# Backward-compatible alias, если захочешь
# импортировать под другим именем.
call_chatbot_classificator = (
    classify_chatbot_intent
)


if __name__ == "__main__":
    test_payload = {
        "text": "ляля",
        "think": True,
    }

    print(
        json.dumps(
            classify_chatbot_intent(
                test_payload
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
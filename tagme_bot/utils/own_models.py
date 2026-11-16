import json
import re
import time
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import urllib3
from dotenv import load_dotenv
from langchain_gigachat.embeddings import GigaChatEmbeddings
from utils.chatbot_classificator import classify_chatbot_intent

logger = logging.getLogger(__name__)
load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


AVAILABLE_OWN_MODELS = [
    "insurance_gcr_model",
    "chatbot_classificator_2",
    "kandi_consist_v1",
    "kandi_consist_mark_vs_gold",
    "audio_quality_analyzer",
    "rag_tool",
]


MODELS_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_INSURANCE_PKL = MODELS_DIR / "multi_target_logreg_model.pkl"


_INSURANCE_MODEL = None
_GIGA_EMBEDDER: Optional[GigaChatEmbeddings] = None


def _get_cert_path(filename: str) -> str:
    cert_path = Path.home() / "certs" / filename
    if not cert_path.exists():
        raise FileNotFoundError(f"Сертификат не найден: {cert_path}")
    return str(cert_path)


def _init_insurance_model(model_path: str | Path = DEFAULT_INSURANCE_PKL):
    global _INSURANCE_MODEL
    if _INSURANCE_MODEL is None:
        mp = Path(model_path)
        if not mp.is_absolute():
            mp = (MODELS_DIR / mp).resolve()
        if not mp.exists():
            raise FileNotFoundError(f"PKL model not found: {mp}")
        _INSURANCE_MODEL = joblib.load(str(mp))
        logger.info("insurance_gcr_model loaded: %s", mp)
    return _INSURANCE_MODEL


def _init_embedder() -> GigaChatEmbeddings:
    global _GIGA_EMBEDDER
    if _GIGA_EMBEDDER is None:
        _GIGA_EMBEDDER = GigaChatEmbeddings(
            model="EmbeddingsGigaR",
            base_url="https://gigachat-ift.sberdevices.delta.sbrf.ru/v1",
            ca_bundle_file=_get_cert_path("ca.pem"),
            cert_file=_get_cert_path("tls.pem"),
            key_file=_get_cert_path("tls.key"),
            verify_ssl_certs=False,
        )
        logger.info("GigaChatEmbeddings initialized OK")
    return _GIGA_EMBEDDER


def _embed_with_retry(text: str, retries: int = 500, sleep_s: float = 0.1) -> np.ndarray:
    embedder = _init_embedder()
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            emb_list = embedder.embed_documents([text])[0]
            return np.array(emb_list, dtype=float)
        except Exception as e:
            last_exc = e
            if attempt % 50 == 0:
                logger.warning("Embedding retry %s/%s: %s", attempt, retries, e)
            time.sleep(sleep_s)
    raise RuntimeError(f"Не удалось получить эмбеддинг после {retries} попыток: {last_exc}")


_CHUNK_CHARS = 3000
_CHUNK_OVERLAP = 200


def _chunk_text(text: str, chunk_chars: int = _CHUNK_CHARS, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [text]

    chunks = []
    step = max(1, chunk_chars - overlap)
    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_chars].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def embed_text_safely(text: str) -> np.ndarray:
    """
    Возвращает 1 эмбеддинг фиксированной размерности.
    Если текст слишком большой — режем на чанки и усредняем эмбеддинги.
    """
    text = (text or "").strip()
    if not text:
        return _embed_with_retry(" ")

    try:
        return _embed_with_retry(text)
    except Exception:
        chunks = _chunk_text(text)
        if not chunks:
            return _embed_with_retry(" ")
        embs = [_embed_with_retry(ch) for ch in chunks]
        return np.mean(np.stack(embs, axis=0), axis=0)


def transform_insurance_dialog(input_json: dict) -> Dict[str, str]:
    dialog_lines = []
    messages = input_json.get("data", {}).get("messages", [])
    for msg in messages:
        speaker = "CLIENT" if msg.get("from") == "user" else "BOT"
        raw_text = (msg.get("text", "") or "").replace("\n", " ").replace("\\n", " ").strip()
        highlighted = bool(msg.get("highlight", False))
        if highlighted:
            line = f"! {speaker}: {raw_text} !"
        else:
            line = f"{speaker}: {raw_text}"
        dialog_lines.append(line)

    dialog_text = "\n".join(dialog_lines).strip()

    scenario = ""
    for qblock in input_json.get("questions", []):
        title = qblock.get("title", "")
        if isinstance(title, str) and title.startswith("Сценарий:"):
            title_no_esc = title.replace("\\n", " ")
            parts = title_no_esc.split("\n")
            clean_parts = [re.sub(r"\s+", " ", part).strip() for part in parts]
            scenario = "\n".join(clean_parts).strip()
            break

    return {"dialog": dialog_text, "scenario": scenario}


def build_questions_structure(scenario: str, gcr_val: str, scen_corr_val: str, slot_filling_val: str) -> list:
    gcr_options = [{"label": "1", "value": "1"}, {"label": "0.5", "value": "0.5"}, {"label": "0", "value": "0"}]

    gcr_block = {
        "title": scenario,
        "questions": [{
            "name": "gcr",
            "text": "Goal Completion Rate (GCR)",
            "type": "radio",
            "value": gcr_val,
            "options": gcr_options,
            "required": True,
            "valueText": gcr_val,
        }],
    }

    scen_options = [{"label": "Да, это так.", "value": "1"}, {"label": "Нет, это не так.", "value": "0"}]
    slot_options = [{"label": "Да, это так.", "value": "1"}, {"label": "Нет, это не так.", "value": "0"}]

    scen_value_text = "Да, это так." if scen_corr_val == "1" else "Нет, это не так."

    if slot_filling_val == "1":
        slot_value = "1"
        slot_value_text = "Да, это так."
    else:
        slot_value = ""
        slot_value_text = ""

    scen_block = {
        "title": "Выбор сценария",
        "questions": [{
            "name": "scenario_correct",
            "text": "Указанный сценарий подходит под запрос клиенту",
            "type": "radio",
            "value": scen_corr_val,
            "options": scen_options,
            "required": True,
            "valueText": scen_value_text,
            "addition": {
                "visibilityValue": "0",
                "questions": [{
                    "name": "slot_filling_correct",
                    "show": [0],
                    "text": "Ассистент пользуется необходимыми данными, уже предоставленными клиентом в этом или в одном из предыдущих запросов",
                    "type": "radio",
                    "value": slot_value,
                    "options": slot_options,
                    "valueText": slot_value_text,
                }],
            },
        }],
    }

    return [gcr_block, scen_block]


def _normalize_pred_to_str(x: Any) -> str:
    if isinstance(x, str):
        return x
    try:
        v = float(x)
        if v.is_integer():
            return str(int(v))
        return str(v)
    except Exception:
        return str(x)


def run_insurance_gcr_model(input_data: Any, *, model_path: str | Path = DEFAULT_INSURANCE_PKL) -> Dict[str, Any]:
    if isinstance(input_data, str):
        try:
            raw_json = json.loads(input_data)
        except Exception:
            raw_json = None
    elif isinstance(input_data, dict):
        raw_json = input_data
    else:
        raw_json = None

    if raw_json is None:
        return {"dialog": "", "scenario": "", "questions": []}

    transformed = transform_insurance_dialog(raw_json)
    dialog_text = transformed.get("dialog", "")
    scenario = transformed.get("scenario", "")

    if not dialog_text or not scenario:
        return {"dialog": dialog_text, "scenario": scenario, "questions": []}

    model = _init_insurance_model(model_path)

    emb1 = embed_text_safely(dialog_text)
    emb2 = embed_text_safely(scenario)

    diff = np.abs(emb1 - emb2)
    combined = np.concatenate([emb1, emb2, diff]).reshape(1, -1)

    preds = model.predict(combined)[0]
    gcr_pred = _normalize_pred_to_str(preds[0])
    scen_corr_pred = _normalize_pred_to_str(int(preds[1]) if not isinstance(preds[1], str) else preds[1])
    slot_fill_pred = _normalize_pred_to_str(int(preds[2]) if not isinstance(preds[2], str) else preds[2])

    questions_structure = build_questions_structure(
        scenario=scenario,
        gcr_val=gcr_pred,
        scen_corr_val=scen_corr_pred,
        slot_filling_val=slot_fill_pred,
    )

    return {
        "dialog": dialog_text,
        "scenario": scenario,
        "questions": questions_structure,
    }

def _inject_think_into_input(
        input_data: Any,
        think: Optional[bool],
) -> Any:
    """
    Добавляет внешний параметр think во вход классификатора.

    Поддерживает:
    - dict;
    - строку с JSON;
    - обычную текстовую строку.

    Если think=None, входные данные не изменяются.
    """
    if think is None:
        return input_data

    think_value = bool(think)

    if isinstance(input_data, dict):
        prepared_input = dict(input_data)
        prepared_input["think"] = think_value
        return prepared_input

    if isinstance(input_data, str):
        stripped = input_data.strip()

        try:
            parsed = json.loads(stripped)

            if isinstance(parsed, dict):
                prepared_input = dict(parsed)
                prepared_input["think"] = think_value
                return prepared_input

        except Exception:
            pass

        return {
            "text": input_data,
            "think": think_value,
        }

    return {
        "text": str(input_data),
        "think": think_value,
    }


def call_own_model(
        model: str,
        input_data: Any,
        **kwargs,
) -> Dict[str, Any]:
    """
    Единая точка входа:

        payload = call_own_model(
            "insurance_gcr_model",
            input_json_or_text,
        )

        payload = call_own_model(
            "chatbot_classificator_2",
            input_json_or_text,
            think=True,
        )
    """
    m = (model or "").strip()

    if m not in AVAILABLE_OWN_MODELS:
        raise ValueError(
            f"Unknown own model: {m}. "
            f"Available: {AVAILABLE_OWN_MODELS}"
        )

    if m == "chatbot_classificator_2":
        chatbot_input = _inject_think_into_input(
            input_data=input_data,
            think=kwargs.get("think"),
        )

        return classify_chatbot_intent(
            chatbot_input
        )

    if m == "insurance_gcr_model":
        return run_insurance_gcr_model(
            input_data,
            model_path=kwargs.get(
                "model_path",
                DEFAULT_INSURANCE_PKL,
            ),
        )

    if m == "rag_tool":
        from utils.reference.rag_tool import run_rag_tool
        return run_rag_tool(input_data, **kwargs)

    if m == "kandi_consist_v1":
        from utils.kandi_consist_v1 import run_kandi_consist_v1
        return run_kandi_consist_v1(input_data)

    if m == "kandi_consist_mark_vs_gold":
        from utils.kandi_consist_mark_vs_gold import run_kandi_consist_mark_vs_gold
        return run_kandi_consist_mark_vs_gold(input_data)

    if m == "audio_quality_analyzer":
        from utils.audio_tools import analyze_audio_quality_from_dict, analyze_audio_quality

        # Поддерживаем три формата входа:
        # 1. Dict с полем INPUT:zaudio_links или audio/path (любой dict)
        # 2. Строка URL (прямой вызов)

        if isinstance(input_data, dict):
            # Используем универсальную функцию из audio_tools
            try:
                result = analyze_audio_quality_from_dict(input_data)
                return result
            except Exception as e:
                logger.exception(f"Audio quality analysis from dict failed: {e}")
                return {
                    "error": "analysis_failed",
                    "message": f"{type(e).__name__}: {e}",
                }

        # Строка - прямой URL
        if isinstance(input_data, str):
            try:
                result = analyze_audio_quality(input_data)
                return result
            except Exception as e:
                logger.exception(f"Audio quality analysis failed: {e}")
                return {
                    "error": "analysis_failed",
                    "message": f"{type(e).__name__}: {e}",
                }

        return {
            "error": "invalid_input_format",
            "message": "Input must be dict with audio key or string URL",
        }

    raise ValueError(
        f"Model is registered but not implemented: {m}"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    example = {
        "data": {"messages": [{"from": "user", "text": "Хочу сменить сценарий", "highlight": True}]},
        "questions": [{"title": "Сценарий: Тестовый сценарий"}],
    }

    payload = call_own_model("insurance_gcr_model", example)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
# -*- coding: utf-8 -*-
import sys
import logging
import http.client
from utils.utils import get_method_and_task_files, download_instruction


http.client.HTTPConnection.debuglevel = 0
http.client.HTTPConnection.debuglevel = 0
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)

logging.getLogger("crowd_sdk.core.utils_agent.http_client").setLevel(logging.ERROR)
logging.getLogger("crowd_sdk").setLevel(logging.ERROR)

logging.getLogger("crowd_sdk").setLevel(logging.WARNING)
logging.getLogger("crowd_sdk.core.utils_agent.http_client").setLevel(logging.WARNING)
logging.getLogger("aiohttp").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

import asyncio
import json
import os
import argparse
import urllib3
import base64
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from crowd_sdk.tagme import TagmeClientAdvanced
from utils.giga_models import call_giga_models
import time
from filelock import FileLock
from utils.config import get_stand
from utils.utils_bot import render_prompt_with_placeholders, wrap_with_initial_data, extract_json_and_think
from utils.vsellm_client import call_vsellm, call_vsellm_chain, _add_usage
from utils.web_search_tools import extract_web_search_stats
from utils.own_models import call_own_model, AVAILABLE_OWN_MODELS
from utils.config import StandConfig, check_service_policy
from utils.relevance_checker import check_relevance
from utils.relevance_checker_video import check_relevance_video
from utils.search_engines_runner import run_search_engines
from datetime import datetime

CONSUMPTION_FILE = "consumption.json"
LOCK_FILE = "consumption.json.lock"


def update_consumption(project_name, org_id, model, usage):
    try:
        usage = usage or {}
        date = datetime.now().strftime("%Y-%m-%d")

        project_name = project_name or "unknown_project"
        model = model or "unknown_model"
        org_id = str(org_id or "unknown_org")

        lock = FileLock(LOCK_FILE)

        with lock:
            if os.path.exists(CONSUMPTION_FILE):
                try:
                    with open(CONSUMPTION_FILE, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    data = {}
            else:
                data = {}

            data.setdefault(date, {})
            data[date].setdefault(project_name, {})
            data[date][project_name].setdefault(org_id, {})
            data[date][project_name][org_id].setdefault(model, {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            })

            data[date][project_name][org_id][model]["prompt_tokens"] += usage.get("prompt_tokens", 0)
            data[date][project_name][org_id][model]["completion_tokens"] += usage.get("completion_tokens", 0)
            data[date][project_name][org_id][model]["total_tokens"] += usage.get("total_tokens", 0)

            with open(CONSUMPTION_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"[CONSUMPTION ERROR] {type(e).__name__}: {e}")


LAST_SUCCESS_TS = time.monotonic()
INSTRUCTION_CACHE: dict[tuple[str, str], str] = {}

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
_log_file_handler = logging.FileHandler("tagme_bot.log", encoding="utf-8")
_log_file_handler.setLevel(logging.INFO)
_log_file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
logging.getLogger().addHandler(_log_file_handler)
logger = logging.getLogger(__name__)

giga_models = ["Qwen3-235b", "GigaChat-2-Reasoning"]
script_models = ["relevance_checker_test_10", "relevance_checker_video"]


async def init_task_static_context(client, task_id: str, org_id: str, token: str, stand: StandConfig):
    """
    ОДИН РАЗ на старте задачи:
    - get_method_and_task_files -> (td, md, files)
    - скачиваем marker_brief если это /files/...
    - достаём bot_cfg/react_cfg
    Возвращаем: (td, md, files, instr_text, bot_cfg, react_cfg)
    """
    result = await get_method_and_task_files(client=client, task_id=task_id, org_id=org_id)
    if not result:
        raise RuntimeError(f"[Task {task_id}] get_method_and_task_files returned None (403?)")

    td, md, files = result

    cfg_get = getattr(md.forms.config, "get", lambda *_: {})

    bot_key = "bot_another" if getattr(stand, "name", "") == "sberdev" else "bot"
    bot_cfg = cfg_get(bot_key) or {}

    react_cfg = cfg_get("react_agent")

    instr = md.marker_brief or ""
    if instr and isinstance(instr, str) and instr.startswith("/files/"):
        cache_key = (org_id, instr)
        cached = INSTRUCTION_CACHE.get(cache_key)
        if cached is not None:
            instr = cached
        else:
            url = stand.host.rstrip("/") + instr
            instr_text = await download_instruction(url, token, org_id)
            INSTRUCTION_CACHE[cache_key] = instr_text
            instr = instr_text

    return td, md, (files or []), (instr or ""), (bot_cfg or {}), react_cfg


def _resize_keep_aspect_local(img: Image.Image, max_side: int = 900) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    return img.resize((nw, nh))


def fix_obs_url(url: str) -> str:
    if not isinstance(url, str):
        return url

    old_prefix = "https://obs.ru-moscow-1.hc.sbercloud.ru/gigaeye-kandinsky-spark"
    new_prefix = "https://gigaeye-kandinsky-spark.obs.ru-moscow-1.hc.sbercloud.ru"

    if url.startswith(old_prefix):
        return url.replace(old_prefix, new_prefix, 1)

    return url


def _image_bytes_to_data_url_jpeg(img_bytes: bytes, max_side: int = 900, quality: int = 70) -> str:
    img = Image.open(BytesIO(img_bytes))
    img = _resize_keep_aspect_local(img, max_side=max_side)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def get_input_value(
        js: dict,
        col_name: str,
        default_names: list[str] | None = None,
):
    """
    Ищет значение в input JSON с учетом INPUT:-префикса.

    Примеры:
    - col_name="video" найдет "video" или "INPUT:video"
    - col_name="INPUT:video" найдет "INPUT:video" или "video"
    - default_names позволяют дополнительно искать старые варианты:
      ["path", "image"] или ["video"]
    """
    if not isinstance(js, dict):
        return None

    name = str(col_name or "").strip()
    candidates = []

    if name:
        candidates.append(name)

        if name.startswith("INPUT:"):
            candidates.append(name.replace("INPUT:", "", 1))
        else:
            candidates.append(f"INPUT:{name}")

    for default_name in default_names or []:
        default_name = str(default_name or "").strip()

        if not default_name:
            continue

        candidates.append(default_name)

        if default_name.startswith("INPUT:"):
            candidates.append(default_name.replace("INPUT:", "", 1))
        else:
            candidates.append(f"INPUT:{default_name}")

    seen = set()

    for candidate in candidates:
        if candidate in seen:
            continue

        seen.add(candidate)

        value = js.get(candidate)

        if value not in (None, ""):
            return value

    return None


def parse_config_bool(
        value,
        default: bool = False,
) -> bool:
    """
    Корректно читает boolean-параметры из bot config.
    """
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    normalized = str(value).strip().lower()

    if normalized in {
        "1",
        "true",
        "yes",
        "y",
        "on",
        "да",
    }:
        return True

    if normalized in {
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


async def process_one_markup_item(
        task_id: str,
        mk,
        client,
        org_id: str,
        td,
        md,
        files,
        instr_template: str,
        bot_cfg: dict,
        react_cfg,
        error_state: dict,
        stand: StandConfig,
):
    """
    Обработка ОДНОГО markup item (mk) внутри фиксированного task_id.
    Всё статичное (md/files/instr/config) уже получено заранее.

    error_state["text"] хранит последнюю причину, почему не получилось сделать COMPLETE.
    """
    global LAST_SUCCESS_TS
    project_id = str(getattr(td, "project_id", None) or "unknown_project")

    # Флаги
    think_flag = parse_config_bool(bot_cfg.get("think"),default=False)
    initial_data_flag = bool(bot_cfg.get("initial_data", False))
    image_flag = bool(bot_cfg.get("image", False))
    chain_flag = bool(bot_cfg.get("chain", False))
    text_flag = bool(bot_cfg.get("text", False))
    placeholders_flag = bool(bot_cfg.get("placeholders", False))
    test_bucket_flag = bool(bot_cfg.get("test_bucket", False))
    video_flag = bool(bot_cfg.get("video", False))
    web_search_flag = parse_config_bool(bot_cfg.get("web_search"), default=False)
    audio_flag = bool(bot_cfg.get("audio", False))
    url_context_flag = parse_config_bool(bot_cfg.get("url_context"), default=False)
    llm_meta_flag = bot_cfg.get("llm_meta", None)

    # Необязательные названия колонок с изображением и видео.
    # Если их нет в конфиге, сохраняется старое поведение:
    # изображение ищем в "path", видео — в "video".
    image_col_name = str(
        bot_cfg.get("image_col_name") or "path"
    ).strip()

    video_col_name = str(
        bot_cfg.get("video_col_name") or "video"
    ).strip()

    audio_col_name = str(
        bot_cfg.get("audio_col_name") or "audio"
    ).strip()

    raw_model = bot_cfg.get("model")
    model = str(raw_model).strip() if raw_model is not None else ""

    service = (bot_cfg.get("service") or "").strip().lower() or None

    policy_error = check_service_policy(stand, service)
    if policy_error:
        logger.error(f"[Task {task_id}] SERVICE POLICY | stand={stand.name} | service={service!r} | {policy_error}")
        payload = {
            "raw_answer": policy_error,
            "error": policy_error,
            "task_id": task_id,
            "assignment_id": getattr(mk, "id", None),
        }
        await client.submit_markup_task(task_id, mk.id, payload)
        LAST_SUCCESS_TS = time.monotonic()
        error_state["text"] = ""
        print(f"[Task {task_id}] COMPLETE with error: {policy_error}")
        return


    reasoning_effort_cfg = str(bot_cfg.get("reasoning_effort") or "").strip().lower() or None
    reasoning_effort_flag = bool(reasoning_effort_cfg) and service == "vsellm"

    logger.info(
        f"[Task {task_id}] BOT ROUTING | "
        f"model={model!r} | "
        f"service={service or 'default'} | "
        f"chain={chain_flag} | "
        f"image_col_name={image_col_name!r} | "
        f"video_col_name={video_col_name!r} | "
        f"audio_col_name={audio_col_name!r}"
    )

    if not model:
        msg = (
            "missing_bot_model: в bot_cfg не найден обязательный параметр 'model'. "
            f"bot_cfg_keys={list((bot_cfg or {}).keys())}"
        )
        logger.error(f"[Task {task_id}] {msg}")
        error_state["text"] = msg

        payload = {
            "raw_answer": "TECH_ERROR",
            "error": msg,
            "task_id": task_id,
            "assignment_id": getattr(mk, "id", None),
        }

        await client.submit_markup_task(task_id, mk.id, payload)
        LAST_SUCCESS_TS = time.monotonic()
        error_state["text"] = ""
        print(f"[Task {task_id}] COMPLETE with error: {msg}")
        return

    file_text = ""
    js = None
    s = ""

    if files:
        target = getattr(mk, "item_id", None) or getattr(mk, "itemId", None)

        # === ДИАГНОСТИКА ===
        logger.info(
            f"[Task {task_id}] assignment={getattr(mk,'id',None)} "
            f"target_item_id={target} "
            f"files_count={len(files)}"
        )

        matched = next((f for f in files if getattr(f, "uid", None) == target), None) if target else None
        if matched is not None:
            chosen = matched
            logger.info(
                f"[Task {task_id}] BRANCH=uid_match: uid={getattr(chosen,'uid',None)} name={getattr(chosen,'name',None)}"
            )
        else:
            msg = f"file_not_found: item_id={target} не найден в списке файлов задачи ({len(files)} файлов)"
            logger.error(f"[Task {task_id}] {msg}")
            error_state["text"] = msg
            payload = {
                "raw_answer": "TECH_ERROR",
                "error": msg,
                "task_id": task_id,
                "assignment_id": getattr(mk, "id", None),
            }
            await client.submit_markup_task(task_id, mk.id, payload)
            LAST_SUCCESS_TS = time.monotonic()
            error_state["text"] = ""
            print(f"[Task {task_id}] COMPLETE with error: {msg}")
            return

        try:
            raw = await client.download_file(task_id, getattr(chosen, "uid"), organization_id=org_id)

            chosen_name = (getattr(chosen, "name", "") or "").lower()
            is_json_content = isinstance(raw, (bytes, bytearray)) and raw.lstrip()[:1] == b"{"
            is_image_file = (not is_json_content) and chosen_name.endswith((".jpg", ".jpeg", ".png", ".webp"))
            is_video_file = (not is_json_content) and chosen_name.endswith((".mp4", ".mov", ".avi", ".mkv", ".webm"))

            if is_image_file:
                if chosen_name.endswith(".png"):
                    mime = "image/png"
                elif chosen_name.endswith(".webp"):
                    mime = "image/webp"
                else:
                    mime = "image/jpeg"

                b64 = base64.b64encode(raw).decode("ascii")
                data_url = f"data:{mime};base64,{b64}"
                js = {image_col_name: data_url}
                file_text = json.dumps(js, ensure_ascii=False, indent=2)
                s = ""
            elif is_video_file:
                b64 = base64.b64encode(raw).decode("ascii")
                data_url = f"data:video/mp4;base64,{b64}"
                js = {video_col_name: data_url}
                file_text = json.dumps(js, ensure_ascii=False, indent=2)
                s = ""
            else:
                s = raw.decode("utf-8")
                try:
                    js = json.loads(s)
                    if "text" in js:
                        file_text = js["text"]
                    elif "data" in js and "messages" in js["data"]:
                        file_text = "\n".join(f"{m['from']}: {m['text']}" for m in js["data"]["messages"])
                    else:
                        file_text = json.dumps(js, ensure_ascii=False, indent=2)
                except Exception as e:
                    logging.warning(f"[Task {task_id}] input is not valid JSON: {e}")
                    file_text = s

        except Exception as e:
            msg = f"download_error: {type(e).__name__}: {e}"
            error_state["text"] = msg
            print(f"[Task {task_id}] download error: {e}")
            return
    else:
        error_state["text"] = "no_files"
        print(f"[Task {task_id}] no files")
        return

    # === DEBUG: что реально пришло из TagMe (файл задания) ===
    if s:
        logger.info(f"[Task {task_id}] INPUT FILE (raw decoded) len={len(s)}")
    logger.info(f"[Task {task_id}] INPUT file_text len={len(file_text)}")

    if isinstance(js, dict):
        keys_preview = list(js.keys())[:50]
        logger.info(f"[Task {task_id}] INPUT JSON detected | keys(top50)={keys_preview}")
        logger.info(f"[Task {task_id}] INPUT JSON full:\n{json.dumps(js, ensure_ascii=False, indent=2)}")
    else:
        if s:
            logger.info(f"[Task {task_id}] INPUT is not JSON. First 2000 chars:\n{s[:2000]}")
        else:
            logger.info(f"[Task {task_id}] INPUT is not JSON. First 2000 chars:\n{str(file_text)[:2000]}")

    user_id = getattr(mk, "user_id", "") or getattr(mk, "userId", "")
    source = {"input_data": file_text, "model_answer": None, "user": user_id}

    instr = instr_template or ""
    usage_acc = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    reasoning_acc = {"tokens": 0, "texts": [], "applied": ""}
    chain_result = None
    rag_extra = {}  # доп. поля входа от rag_tool (reference): для final_user_prompt и initial_data

    if placeholders_flag and instr and isinstance(js, dict):
        rendered_instr = render_prompt_with_placeholders(instr, js)
        logger.info(
            f"[Task {task_id}] SYSTEM PROMPT rendered with placeholders (placeholders=true). "
            f"Length old={len(instr)}, new={len(rendered_instr)}"
        )
        instr = rendered_instr

    # --- REACT/CHAIN ветки---
    if react_cfg:
        agent_conf = react_cfg.copy()
        if "model" not in agent_conf:
            agent_conf["model"] = bot_cfg.get("model", "GigaChat-2-Max")

        print(f"[Task {task_id}] START react_agent")
        from utils.react_agent_bot import run_react_agent
        payload = await asyncio.to_thread(
            run_react_agent,
            agent_conf,
            instr,
            file_text
        )
        payload = wrap_with_initial_data(payload, js, initial_data_flag)
        await client.submit_markup_task(task_id, mk.id, payload)
        LAST_SUCCESS_TS = time.monotonic()
        error_state["text"] = ""
        print(f"[Task {task_id}] COMPLETE react_agent")
        return

    async def fetch_model_answer():
        nonlocal usage_acc, chain_result

        if js is not None:
            base_text = json.dumps(js, ensure_ascii=False, indent=2)
        else:
            base_text = file_text

        question_text = None
        if isinstance(js, dict):
            for key in ("Question", "question"):
                val = js.get(key)
                if isinstance(val, str) and val.strip():
                    question_text = val.strip()
                    break

        if test_bucket_flag and question_text:
            user_prompt_text = question_text
        else:
            user_prompt_text = base_text

        source["user_prompt"] = user_prompt_text

        logger.info(
            f"[Task {task_id}] MODEL INPUT (will be sent to model '{model}'):\n{user_prompt_text}"
        )

        image_value = get_input_value(
            js,
            image_col_name,
            default_names=["path", "image"],
        )

        video_value = get_input_value(
            js,
            video_col_name,
            default_names=["video"],
        )

        audio_value = get_input_value(
            js,
            audio_col_name,
            default_names=["audio", "path"],
        )

        image_mode = bool(
            image_flag
            or (
                    text_flag
                    and image_value
            )
        )

        video_mode = bool(
            video_flag
            and video_value
        )

        audio_mode = bool(
            audio_flag
            and audio_value
        )
        service_is_gigahf = service == "gigahf"
        service_is_ift = service == "ift"
        service_is_vsellm = service == "vsellm"
        service_is_direct = service == "direct"

        explicit_vsellm_service = service in {
            "gigahf",
            "ift",
            "vsellm",
            "direct",
        }

        # ===== SEARCH ENGINES MODEL =====
        if model == "search_engines":
            try:
                logger.info(f"[Task {task_id}] Calling SEARCH ENGINES model")

                if not isinstance(js, dict):
                    msg = "search_engines_input_not_json: input data must be JSON with transformed_query"
                    error_state["text"] = msg
                    logger.error(f"[Task {task_id}] {msg}")
                    return None

                transformed_query = js.get("transformed_query")

                if not isinstance(transformed_query, str) or not transformed_query.strip():
                    msg = "search_engines_missing_transformed_query"
                    error_state["text"] = msg
                    logger.error(
                        f"[Task {task_id}] {msg}. Input keys={list(js.keys())}"
                    )
                    return None

                transformed_query = transformed_query.strip()

                search_engines = bot_cfg.get("search_engines", [])

                if not isinstance(search_engines, list) or not search_engines:
                    msg = "search_engines_empty_config"
                    error_state["text"] = msg
                    logger.error(
                        f"[Task {task_id}] {msg}. bot_cfg.search_engines={search_engines}"
                    )
                    return None

                logger.info(
                    f"[Task {task_id}] SEARCH ENGINES INPUT: transformed_query={transformed_query!r}, "
                    f"engines={search_engines}"
                )

                search_payload = await asyncio.to_thread(
                    run_search_engines,
                    transformed_query,
                    search_engines,
                    5,
                    "ru",
                    "basic",
                    True,
                    None,
                    60,
                    True,
                )

                payload = {
                    "user_query": js.get("user_query", ""),
                    "transformed_query": transformed_query,
                    "correct_answer": js.get("correct_answer", ""),
                    "search_engines": search_payload.get("search_engines", []),
                }

                logger.info(
                    f"[Task {task_id}] SEARCH ENGINES RESULT:\n"
                    f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
                )

                return json.dumps(payload, ensure_ascii=False)

            except Exception as e:
                msg = f"search_engines_failed: {type(e).__name__}: {e}"
                error_state["text"] = msg
                logger.exception(f"[Task {task_id}] search_engines failed")
                return None

        # ===== OWN MODELS =====
        if model in AVAILABLE_OWN_MODELS:
            try:
                logger.info(
                    f"[Task {task_id}] Calling OWN model: {model} | "
                    f"think={think_flag}"
                )

                own_input_data = (
                    js
                    if js is not None
                    else file_text
                )

                own_model_kwargs = {}

                if model == "chatbot_classificator_2":
                    own_model_kwargs["think"] = think_flag

                if model == "rag_tool":
                    own_model_kwargs.update(
                        stand=stand,
                        bot_cfg=bot_cfg,
                        instr=instr,
                        think=think_flag,
                        web_search=web_search_flag,
                        url_context=url_context_flag,
                        reasoning_effort=reasoning_effort_cfg if reasoning_effort_flag else None,
                    )

                result = await asyncio.to_thread(
                    call_own_model,
                    model,
                    own_input_data,
                    **own_model_kwargs,
                )

                # Служебные данные own-модели (токены LLM, reasoning, think) -> учёт бота, из ответа убираем.
                if isinstance(result, dict) and "_meta" in result:
                    meta = result.pop("_meta") or {}
                    usage_acc = _add_usage(usage_acc, meta.get("usage") or {})
                    if reasoning_effort_flag:
                        reasoning_acc["tokens"] += int(meta.get("reasoning_tokens") or 0)
                        if meta.get("reasoning"):
                            reasoning_acc["texts"].append(meta["reasoning"])
                        if meta.get("reasoning_effort_applied"):
                            reasoning_acc["applied"] = meta["reasoning_effort_applied"]
                    if "reference" in meta and isinstance(js, dict):
                        rag_extra["reference"] = meta["reference"]
                        source["user_prompt"] = json.dumps(
                            {**js, **rag_extra}, ensure_ascii=False, indent=2
                        )
                    meta_think = meta.get("think") or ""
                    if meta_think:
                        source["think"] = (
                                source.get("think", "") + ("\n" if source.get("think") else "") + meta_think
                        ).strip()

                logger.info(
                    f"[Task {task_id}] OWN MODEL RESULT:\n"
                    f"{json.dumps(result, ensure_ascii=False, indent=2)}"
                )

                return json.dumps(
                    result,
                    ensure_ascii=False,
                )

            except Exception as e:
                msg = (
                    f"own_model_failed: "
                    f"{type(e).__name__}: {e}"
                )

                error_state["text"] = msg

                logger.exception(
                    f"[Task {task_id}] "
                    f"own model call failed"
                )

                return None

        if explicit_vsellm_service or (
                model
                and model not in giga_models
                and model not in AVAILABLE_OWN_MODELS
                and model not in script_models
        ):
            try:
                messages = []
                if instr:
                    messages.append({"role": "system", "content": instr})
                messages.append({"role": "user", "content": user_prompt_text})

                temperature = bot_cfg.get("temperature", None)
                max_tokens = bot_cfg.get("max_new_tokens", None)
                top_p = bot_cfg.get("top_p", None)
                user_prompt = bot_cfg.get("user_prompt", None)
                extra_text = base_text if text_flag else None

                image_url = None
                video_url = None
                audio_url = None

                # ==== ОБРАБОТКА ИЗОБРАЖЕНИЯ ИЗ TagMe ====
                # js[image_col_name] содержит ссылку или data URL изображения.
                if image_mode and isinstance(js, dict):
                    raw_path = get_input_value(
                        js,
                        image_col_name,
                        default_names=["path", "image"],
                    )
                    if isinstance(raw_path, str):
                        raw_path = fix_obs_url(raw_path)
                        raw_path = raw_path.strip()

                        if raw_path.startswith("https://tagme."):
                            try:
                                uid = raw_path.rstrip("/").split("/")[-1]
                                raw_bytes = await client.download_file(task_id, uid, organization_id=org_id)
                                image_url = _image_bytes_to_data_url_jpeg(raw_bytes)
                                logger.info(
                                    f"[Task {task_id}] Converted TagMe image {uid} to data URL for VseLLM"
                                )
                            except Exception as e:
                                logger.error(
                                    f"[Task {task_id}] Failed to download/convert TagMe image from {raw_path}: {e}"
                                )
                                image_mode = False
                        else:
                            image_url = raw_path

                # ==== ОБРАБОТКА ВИДЕО ====
                if video_mode and isinstance(js, dict):
                    raw_video = get_input_value(
                        js,
                        video_col_name,
                        default_names=["video"],
                    )

                    if isinstance(raw_video, str):
                        raw_video = raw_video.strip()

                        if (
                                raw_video.startswith("data:video/")
                                and "base64," in raw_video
                        ):
                            try:
                                b64 = raw_video.split("base64,", 1)[1]
                                decoded = base64.b64decode(b64).decode(
                                    "utf-8",
                                    errors="ignore",
                                )

                                if decoded.strip().startswith("{"):
                                    j2 = json.loads(decoded)

                                    if isinstance(j2, dict):
                                        extracted_video = get_input_value(
                                            j2,
                                            video_col_name,
                                            default_names=["video"],
                                        )

                                        if (
                                                isinstance(extracted_video, str)
                                                and extracted_video.strip()
                                        ):
                                            raw_video = extracted_video.strip()

                                            logger.warning(
                                                f"[Task {task_id}] FIX: extracted "
                                                f"real video URL from base64 JSON "
                                                f"field {video_col_name!r}"
                                            )

                            except Exception as e:
                                logger.warning(
                                    f"[Task {task_id}] base64 video decode failed: {e}"
                                )

                        video_url = fix_obs_url(raw_video)

                        logger.info(
                            f"[Task {task_id}] VIDEO INPUT | "
                            f"column={video_col_name!r} | "
                            f"value_type={type(raw_video).__name__}"
                        )

                # ==== ОБРАБОТКА АУДИО ====
                if audio_mode and isinstance(js, dict):
                    raw_audio = get_input_value(
                        js,
                        audio_col_name,
                        default_names=["audio", "path"],
                    )

                    if isinstance(raw_audio, str):
                        raw_audio = fix_obs_url(raw_audio).strip()

                        if raw_audio.startswith("https://tagme."):
                            try:
                                uid = raw_audio.rstrip("/").split("/")[-1]
                                raw_bytes = await client.download_file(task_id, uid, organization_id=org_id)
                                _afmt = "wav"
                                _low = raw_audio.lower()
                                for _ext in ("flac", "mp3", "ogg", "aac", "wav"):
                                    if _low.endswith("." + _ext) or ("." + _ext + "?") in _low:
                                        _afmt = _ext
                                        break
                                audio_url = (
                                    "data:audio/{};base64,".format(_afmt)
                                    + base64.b64encode(raw_bytes).decode("ascii")
                                )
                                logger.info(
                                    f"[Task {task_id}] Converted TagMe audio {uid} to data URL for VseLLM"
                                )
                            except Exception as e:
                                logger.error(
                                    f"[Task {task_id}] Failed to download/convert TagMe audio from {raw_audio}: {e}"
                                )
                                audio_mode = False
                        else:
                            audio_url = raw_audio

                if chain_flag:
                    chain_media_mode = str(
                        bot_cfg.get("chain_media_mode") or "first"
                    ).strip().lower()

                    logger.info(
                        f"[Task {task_id}] Calling CHAIN | "
                        f"model={model} | "
                        f"service={service or 'default'} | "
                        f"image={bool(image_mode)} | "
                        f"video={bool(video_mode)} | "
                        f"audio={bool(audio_mode)} | "
                        f"image_url={bool(image_url)} | "
                        f"video_url={bool(video_url)} | "
                        f"audio_url={bool(audio_url)} | "
                        f"chain_media_mode={chain_media_mode}"
                    )

                    result = await asyncio.to_thread(
                        call_vsellm_chain,
                        instr=instr,
                        user_input=user_prompt_text,
                        model=model,
                        service=service,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        top_k=bot_cfg.get("top_k", None),
                        timeout_sec=600,
                        system_prompt=bot_cfg.get(
                            "chain_system_prompt",
                            None,
                        ),

                        # NEW: media для chain
                        image=bool(image_mode),
                        video=bool(video_mode),
                        audio=bool(audio_mode),
                        image_url=image_url,
                        video_url=video_url,
                        audio_url=audio_url,
                        extra_text=extra_text,
                        chain_media_mode=chain_media_mode,

                        # NEW: web_search для chain
                        web_search=web_search_flag,
                        url_context=url_context_flag,
                        reasoning_effort=reasoning_effort_cfg if reasoning_effort_flag else None,
                    )

                    chain_result = result

                else:
                    result = await asyncio.to_thread(
                        call_vsellm,
                        model=model,
                        service=service,
                        system_prompt=instr,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        image=bool(image_mode),
                        video=bool(video_mode),
                        audio=bool(audio_mode),
                        image_url=image_url,
                        video_url=video_url,
                        audio_url=audio_url,
                        extra_text=extra_text,
                        user_text=user_prompt,
                        timeout_sec=600,
                        enable_thinking=True if think_flag else False,
                        web_search=web_search_flag,
                        url_context=url_context_flag,
                        reasoning_effort=reasoning_effort_cfg if reasoning_effort_flag else None,
                    )

                usage_acc = _add_usage(usage_acc, result.get("usage") or {})

                if reasoning_effort_flag:
                    reasoning_acc["tokens"] += int(result.get("reasoning_tokens") or 0)
                    r_txt = result.get("reasoning") or ""
                    if r_txt:
                        reasoning_acc["texts"].append(r_txt)
                    r_applied = result.get("reasoning_effort_applied") or ""
                    if r_applied:
                        reasoning_acc["applied"] = r_applied

                # Добавляем статистику веб-поиска
                web_search_stats = extract_web_search_stats(result)
                if web_search_stats:
                    if "web_search_stats" not in usage_acc:
                        usage_acc["web_search_stats"] = {}
                    usage_acc["web_search_stats"].update(web_search_stats)

                ans = (result.get("answer") or result.get("raw_text") or "").strip()
                think = result.get("think", "")
                if think:
                    source["think"] = (
                            source.get("think", "") + ("\n" if source.get("think") else "") + think
                    ).strip()
                    logger.info(f"[Task {task_id}] [think from model] ({len(think)} chars):\n{think}")
                if ans:
                    return ans

                error_state["text"] = "vsellm_empty_answer"
                return None

            except Exception as e:
                msg = f"vsellm_call_failed: {type(e).__name__}: {e}"
                error_state["text"] = msg
                logger.error(f"[Task {task_id}] VseLLM call failed: {e}")
                return None

        if model in script_models:
            try:
                logger.info(f"[Task {task_id}] Calling script model: {model}")

                text_to_check = user_prompt_text
                instruction_text = instr or ""

                relevance_model = bot_cfg.get("relevance_model", "yandex/gpt5-pro")
                relevance_temperature = bot_cfg.get("relevance_temperature", 0.75)
                relevance_max_tokens = bot_cfg.get("relevance_max_tokens", 10000)

                logger.info(
                    f"[Task {task_id}] Relevance params: model={relevance_model}, "
                    f"temperature={relevance_temperature}, max_tokens={relevance_max_tokens}"
                )

                ban_words_list = bot_cfg.get("ban_words", [])
                forbidden_words_set = set(w.lower() for w in ban_words_list) if ban_words_list else None

                logger.info(
                    f"[Task {task_id}] Using ban_words from config: {len(ban_words_list) if ban_words_list else 0} words"
                )

                if model == "relevance_checker_video":
                    result = await asyncio.to_thread(
                        check_relevance_video,
                        text=text_to_check,
                        instruction=instruction_text,
                        model=relevance_model,
                        temperature=relevance_temperature,
                        max_tokens=relevance_max_tokens,
                        forbidden_words=forbidden_words_set
                    )
                else:
                    result = await asyncio.to_thread(
                        check_relevance,
                        text=text_to_check,
                        instruction=instruction_text,
                        model=relevance_model,
                        temperature=relevance_temperature,
                        max_tokens=relevance_max_tokens,
                        forbidden_words=forbidden_words_set
                    )

                logger.info(
                    f"[Task {task_id}] RELEVANCE RESULT: {result}"
                )

                raw_usage = result.get("usage") or {}
                if not isinstance(raw_usage, dict):
                    raw_usage = {
                        "prompt_tokens": getattr(raw_usage, "prompt_tokens", 0) or 0,
                        "completion_tokens": getattr(raw_usage, "completion_tokens", 0) or 0,
                        "total_tokens": getattr(raw_usage, "total_tokens", 0) or 0,
                    }

                usage_acc = _add_usage(usage_acc, raw_usage)

                if result["result"] == "notrelevant":

                    if result.get("forbidden_word"):
                        payload = {
                            "correct": "notrelevant",
                            "text": result.get("forbidden_word"),
                            "model_used": result.get("model_used"),
                            "temperature": result.get("temperature"),
                            "final_decision_by": result.get("final_decision_by"),
                            "destination": result.get("destination"),
                            "direction_words": result.get("direction_words")
                        }
                    else:
                        payload = {
                            "correct": "notrelevant",
                            "model_used": result.get("model_used"),
                            "temperature": result.get("temperature"),
                            "final_decision_by": result.get("final_decision_by"),
                            "destination": result.get("destination"),
                            "direction_words": result.get("direction_words")
                        }

                else:
                    payload = {
                        "correct": "relevant",
                        "model_used": result.get("model_used"),
                        "temperature": result.get("temperature"),
                        "final_decision_by": result.get("final_decision_by"),
                        "destination": result.get("destination"),
                        "direction_words": result.get("direction_words")
                    }

                return json.dumps(payload, ensure_ascii=False)

            except Exception as e:
                msg = f"script_model_failed: {type(e).__name__}: {e}"
                error_state["text"] = msg
                logger.error(f"[Task {task_id}] script model call failed: {e}")
                return None

        # Giga-modes
        if model in giga_models:
            max_attempts = 3

            for attempt in range(1, max_attempts + 1):
                try:
                    messages = []
                    if instr:
                        messages.append({"role": "system", "content": instr})
                    messages.append({"role": "user", "content": user_prompt_text})

                    temperature = bot_cfg.get("temperature", None)
                    max_tokens = bot_cfg.get("max_new_tokens", None)
                    top_p = bot_cfg.get("top_p", None)

                    timeout_sec = 500.0

                    result = await asyncio.to_thread(
                        call_giga_models,
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        timeout=timeout_sec,
                        extract_think=None
                    )

                    usage_acc = _add_usage(usage_acc, result.get("usage") or {})
                    raw_text = result.get("raw_text", "")
                    ans = result.get("answer", "")
                    think = result.get("think", "")

                    logger.info(f"[Task {task_id}] <<< RAW from {model} (len={len(str(raw_text))})")

                    if think:
                        source["think"] = (
                                source.get("think", "") + ("\n" if source.get("think") else "") + think
                        ).strip()

                    if ans:
                        return ans
                    if raw_text and isinstance(raw_text, str):
                        return raw_text.strip()

                    error_state["text"] = f"{model}: empty_answer"
                    return None

                except Exception as e:
                    msg = f"{model} call failed (attempt {attempt}/{max_attempts}): {type(e).__name__}: {e}"
                    error_state["text"] = msg
                    logger.error(f"[Task {task_id}] {model} call failed (attempt {attempt}/{max_attempts}): {e}")
                    if attempt < max_attempts:
                        await asyncio.sleep(2 * attempt if model == "DeepSeek-R1" else 2)
                        continue
                    return None

    payload = None
    last_ans = None
    for _ in range(3):
        ans = await fetch_model_answer()
        if ans is None:
            return
        source["model_answer"] = ans
        last_ans = ans

        logger.info(f"[Task {task_id}] RAW MODEL ANSWER (before json handling):\n{ans}")

        payload_obj, think_txt = extract_json_and_think(ans)
        logger.info(f"ANSWER LEN={len(ans)}")
        if payload_obj is not None:
            if think_txt:
                source["think"] = (
                        source.get("think", "") + ("\n" if source.get("think") else "") + think_txt
                ).strip()
            payload = payload_obj if isinstance(payload_obj, dict) else {"result": payload_obj}
            break

        try:
            parsed = json.loads(ans)
            payload = parsed if isinstance(parsed, dict) else {"result": parsed}
            break
        except Exception as e:
            error_state["text"] = f"json_parse_error: {type(e).__name__}: {e}"
            await asyncio.sleep(2)

    if payload is None:
        raw_text = last_ans or source.get("model_answer")
        payload = {"raw_answer": raw_text}
        error_state["text"] = error_state.get("text") or "payload_none"

    if isinstance(payload, dict) and "full_task" in payload:
        payload["full_task"] = file_text

    if think_flag and "think" in source:
        payload["think"] = source["think"]

    if (
            isinstance(payload, dict)
            and model != "search_engines"
            and llm_meta_flag is None
    ):
        if "user_prompt" in source:
            payload["final_user_prompt"] = source["user_prompt"]

    if (
            isinstance(payload, dict)
            and chain_flag
            and llm_meta_flag is not False
            and isinstance(chain_result, dict)
    ):
        payload["llm_meta"] = {
            "type": "chain",
            "model": model,
            "service": service or "default",
            "chain_meta": chain_result.get("chain_meta", {}),
            "prompts": chain_result.get("prompts", {}),
            "answers": chain_result.get("chain_answers", {}),
            "step_usage": chain_result.get("step_usage", {}),
            "usage": chain_result.get("usage", {}),
            "final_answer": (
                    chain_result.get("answer")
                    or chain_result.get("raw_text")
                    or ""
            ),
        }

    if isinstance(payload, dict) and usage_acc:
        if any(usage_acc.get(k, 0) for k in ("prompt_tokens", "completion_tokens", "total_tokens")):
            payload["tokens_prompt"] = usage_acc.get("prompt_tokens", 0)
            payload["tokens_completion"] = usage_acc.get("completion_tokens", 0)
            payload["tokens_total"] = usage_acc.get("total_tokens", 0)

    if isinstance(payload, dict) and reasoning_effort_flag:
        payload["reasoning_effort"] = reasoning_effort_cfg
        if reasoning_acc["applied"]:
            payload["reasoning_effort_applied"] = reasoning_acc["applied"]
        payload["reasoning_tokens"] = reasoning_acc["tokens"]
        if reasoning_acc["texts"]:
            payload["reasoning"] = "\n---\n".join(reasoning_acc["texts"])

    logger.info(f"[Task {task_id}] ===== FINAL PAYLOAD TO MARKUP =====")
    logger.info(f"[Task {task_id}] Keys: {list(payload.keys()) if isinstance(payload, dict) else type(payload)}")
    if isinstance(payload, dict):
        for k, v in payload.items():
            if k == "think":
                logger.info(f"[Task {task_id}] [think] ({len(str(v))} chars):\n{v}")
            elif k == "raw_answer":
                logger.info(f"[Task {task_id}] [raw_answer] ({len(str(v))} chars):\n{v}")
            else:
                logger.info(f"[Task {task_id}] [{k}]: {v}")
    logger.info(f"[Task {task_id}] ===================================")
    payload = wrap_with_initial_data(
        payload, {**js, **rag_extra} if isinstance(js, dict) else js, initial_data_flag
    )

    await client.submit_markup_task(task_id, mk.id, payload)
    try:
        if any(usage_acc.get(k, 0) for k in ("prompt_tokens", "completion_tokens", "total_tokens")):
            update_consumption(
                project_name=project_id,
                org_id=org_id,
                model=model,
                usage=usage_acc,
            )
    except Exception as e:
        logger.error(f"[Task {task_id}] Failed to update consumption: {e}")
    LAST_SUCCESS_TS = time.monotonic()
    error_state["text"] = ""
    print(f"[Task {task_id}] COMPLETE")


async def run_single_task_loop(client, org_id: str, task_id: str, stand, max_idle_sec: int = 300):
    """
    1) Один раз: td/md/files/instr/config
    2) Цикл: get_markup_task(task_id) -> download нужного файла -> обработка -> submit
    """
    global LAST_SUCCESS_TS
    LAST_SUCCESS_TS = time.monotonic()

    os.environ["ORG_ID"] = org_id

    td, md, files, instr_template, bot_cfg, react_cfg = await init_task_static_context(
        client=client,
        task_id=task_id,
        org_id=org_id,
        token=client.client.token,
        stand=stand
    )

    logger.info(f"[Task {task_id}] INIT OK | project_id={getattr(td, 'project_id', None)} | files={len(files)}")
    logger.info(f"[Task {task_id}] bot_cfg keys={list((bot_cfg or {}).keys())} react_cfg={'yes' if react_cfg else 'no'}")

    current_assignment_id: str | None = None
    assignment_start_ts = time.monotonic()
    TECH_AFTER_SEC = 240

    error_state = {"text": ""}
    exit_reason = "idle_timeout"

    while True:
        try:
            mk = await client.get_markup_task(task_id)
        except Exception as e:
            logger.error(f"[Task {task_id}] get_markup_task failed: {e}")
            error_state["text"] = f"get_markup_task_failed: {type(e).__name__}: {e}"
            if time.monotonic() - LAST_SUCCESS_TS > max_idle_sec:
                logger.warning(
                    f"[Task {task_id}] get_markup_task падает и нет успешных COMPLETE больше {max_idle_sec} секунд — выходим."
                )
                exit_reason = "idle_timeout"
                break
            await asyncio.sleep(2)
            continue

        mk_id = getattr(mk, "id", None) or ""
        if not mk_id:
            error_state["text"] = "empty_assignment_id"
            if time.monotonic() - LAST_SUCCESS_TS > max_idle_sec:
                logger.warning(f"[Task {task_id}] Нет успешных COMPLETE больше {max_idle_sec} секунд — выходим.")
                exit_reason = "idle_timeout"
                break
            await asyncio.sleep(1)
            continue

        if mk_id != (current_assignment_id or ""):
            current_assignment_id = mk_id
            assignment_start_ts = time.monotonic()
            error_state["text"] = ""

        last_success_before = LAST_SUCCESS_TS

        # Обновляем список файлов если item_id текущего задания не найден в кеше
        target = getattr(mk, "item_id", None) or getattr(mk, "itemId", None)
        if target and not any(getattr(f, "uid", None) == target for f in files):
            logger.warning(f"[Task {task_id}] target={target} не найден в кеше ({len(files)} файлов), запрашиваем свежий список...")
            fresh = await client.get_task_files(task_id=task_id, organization_id=org_id)
            if fresh:
                files = fresh
                logger.info(f"[Task {task_id}] Список файлов обновлён: {len(files)} файлов")

        try:
            await process_one_markup_item(
                task_id=task_id,
                mk=mk,
                client=client,
                org_id=org_id,
                td=td,
                md=md,
                files=files,
                instr_template=instr_template,
                bot_cfg=bot_cfg,
                react_cfg=react_cfg,
                error_state=error_state,
                stand=stand,
            )
        except Exception as e:
            if "already finished" in str(e).lower():
                logger.warning(
                    f"[Task {task_id}] assignment {getattr(mk, 'id', None)} already finished "
                    f"(likely claimed by another running instance) — выходим."
                )
                exit_reason = "already_finished"
                break
            logger.exception(f"[Task {task_id}] processing crashed: {e}")
            error_state["text"] = f"processing_crashed: {type(e).__name__}: {e}"
            await asyncio.sleep(1)

        if (LAST_SUCCESS_TS == last_success_before) and (time.monotonic() - assignment_start_ts > TECH_AFTER_SEC):
            try:
                err_text = (error_state.get("text") or "").strip()
                if not err_text:
                    err_text = f"no_complete>{TECH_AFTER_SEC}s"

                logger.warning(
                    f"[Task {task_id}] assignment {mk_id} no COMPLETE > {TECH_AFTER_SEC} sec — TECH SUBMIT | error={err_text}"
                )

                payload = {
                    "raw_answer": "TECH_ERROR",
                    "error": err_text,
                    "task_id": task_id,
                    "assignment_id": mk_id,
                }

                await client.submit_markup_task(task_id, mk_id, payload)
                LAST_SUCCESS_TS = time.monotonic()
                assignment_start_ts = time.monotonic()
                error_state["text"] = ""
                await asyncio.sleep(0.5)
                continue
            except Exception as e:
                if "already finished" in str(e).lower():
                    logger.warning(
                        f"[Task {task_id}] tech submit: assignment {mk_id} already finished "
                        f"(likely claimed by another running instance) — выходим."
                    )
                    exit_reason = "already_finished"
                    break
                logger.error(f"[Task {task_id}] tech submit failed for assignment {mk_id}: {e}")
                await asyncio.sleep(1)

        if time.monotonic() - LAST_SUCCESS_TS > max_idle_sec:
            logger.warning(f"[Task {task_id}] Нет успешных COMPLETE больше {max_idle_sec} секунд - выходим.")
            exit_reason = "idle_timeout"
            break

        await asyncio.sleep(0.1)

    return exit_reason


def main():
    parser = argparse.ArgumentParser(description="Аннотатор Tagme (одна task_id + org_id)")
    parser.add_argument("-t", "--task-id", required=True, help="Task ID")
    parser.add_argument("-o", "--org-id", required=True, help="Organization ID")
    parser.add_argument("--stand", required=True, choices=["cloud", "sigma", "sberdev"], help="Stand name")
    args = parser.parse_args()
    stand = get_stand(args.stand)
    task_id = args.task_id
    org_id = args.org_id

    client = TagmeClientAdvanced(stand.cfg_path)

    exit_reason = asyncio.run(
        run_single_task_loop(client=client, org_id=org_id, task_id=task_id, stand=stand, max_idle_sec=300)
    )
    if exit_reason == "already_finished":
        sys.exit(42)


if __name__ == "__main__":
    main()
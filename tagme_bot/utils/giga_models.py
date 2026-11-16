import os
import time
import json
import logging
import http.client
from pathlib import Path
from typing import Optional, Any, Dict, List, Tuple

from gigachat import GigaChat

logger = logging.getLogger(__name__)


GIGA_MODELS = {"GigaChat-Max", "GigaChat-2-Max", "GigaChat-2-Reasoning", "Qwen"}
DEEPSEEK_MODELS = {"DeepSeek-R1"}


def _enable_http_wire_dump():
    http.client.HTTPConnection.debuglevel = 1
    logging.getLogger("urllib3").setLevel(logging.DEBUG)
    logging.getLogger("urllib3.connectionpool").setLevel(logging.DEBUG)

    root = logging.getLogger()
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(logging.StreamHandler())


def split_think_answer(raw: str) -> Tuple[str, str]:
    if not isinstance(raw, str):
        return "", str(raw)

    s = raw.strip()
    if "<think>" in s and "</think>" in s:
        head, tail = s.split("</think>", 1)
        think = head.replace("<think>", "", 1).strip()
        return think, tail.strip()

    if s.lstrip().startswith("<think>"):
        return s.split("<think>", 1)[1].strip(), ""

    return "", s


def _default_cert_paths_for_model(model: str) -> tuple[str, str, str]:
    home = Path.home()

    if model in DEEPSEEK_MODELS:
        ca = os.getenv("DEEPSEEK_CA_BUNDLE") or str(home / "certs" / "DeepSeek" / "chain.pem")
        cert = os.getenv("DEEPSEEK_CERT_FILE") or str(home / "certs" / "DeepSeek" / "published.pem")
        key = os.getenv("DEEPSEEK_KEY_FILE") or str(home / "certs" / "DeepSeek" / "CI06119293-IFT-RMD.key")
        return ca, cert, key

    ca = os.getenv("GIGA_CA_BUNDLE") or str(home / "certs" / "ca.pem")
    cert = os.getenv("GIGA_CERT_FILE") or str(home / "certs" / "tls.pem")
    key = os.getenv("GIGA_KEY_FILE") or str(home / "certs" / "tls.key")
    return ca, cert, key


def call_giga_models(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    system_prompt: Optional[str] = None,
    base_url: str = "https://gigachat-ift.sberdevices.delta.sbrf.ru/v1",
    timeout: float = 200.0,
    ca_bundle_file: Optional[str] = None,
    cert_file: Optional[str] = None,
    key_file: Optional[str] = None,
    profanity_check: bool = False,
    reasoning_content: bool = True,
    debug_wire: bool = False,
    sleep_after_sec: float = 1.0,
    extract_think: Optional[bool] = None,
) -> Dict[str, Any]:

    if not isinstance(messages, list) or not messages:
        raise ValueError("call_giga_family: messages must be a non-empty list")

    if system_prompt:
        has_system = any(m.get("role") == "system" for m in messages)
        if not has_system:
            messages = [{"role": "system", "content": system_prompt}] + messages

    if ca_bundle_file is None or cert_file is None or key_file is None:
        ca0, cert0, key0 = _default_cert_paths_for_model(model)
        ca_bundle_file = ca_bundle_file or ca0
        cert_file = cert_file or cert0
        key_file = key_file or key0

    if extract_think is None:
        extract_think = (model in {"Qwen", "DeepSeek-R1"})

    prompt_dict: Dict[str, Any] = {
        "messages": messages,
        "profanity_check": profanity_check,
        "reasoning_content": reasoning_content,
    }

    if temperature is not None:
        prompt_dict["temperature"] = float(temperature)
    if top_p is not None:
        prompt_dict["top_p"] = float(top_p)
    if max_tokens is not None:
        prompt_dict["max_tokens"] = int(max_tokens)

    if os.getenv("GIGA_PRINT_REQUEST_JSON") == "1":
        print("\n===== OUTGOING REQUEST BODY (JSON to SDK) =====")
        print(json.dumps(prompt_dict, ensure_ascii=False, indent=2))
        print("================================================\n")

    if debug_wire or os.getenv("GIGA_HTTP_DUMP") == "1":
        _enable_http_wire_dump()

    with GigaChat(
        model=model,
        base_url=base_url,
        ca_bundle_file=ca_bundle_file,
        cert_file=cert_file,
        key_file=key_file,
        timeout=timeout,
    ) as giga:
        resp = giga.chat(prompt_dict)

    if os.getenv("GIGA_PRINT_RESPONSE_OBJ") == "1":
        print("\n===== SDK RESPONSE OBJECT =====")
        print(resp)
        print("================================\n")

    time.sleep(sleep_after_sec)

    # --- Extract usage from API response ---
    usage_obj = getattr(resp, "usage", None)
    usage_dict = {
        "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0) if usage_obj else 0,
        "completion_tokens": getattr(usage_obj, "completion_tokens", 0) if usage_obj else 0,
        "total_tokens": getattr(usage_obj, "total_tokens", 0) if usage_obj else 0,
    }

    raw_text = ""
    try:
        raw_text = resp.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"call_giga_family: failed to extract content: {e}")

    think, answer = ("", raw_text.strip())
    if extract_think:
        think, answer = split_think_answer(raw_text)

    return {
        "raw_text": raw_text,
        "answer": answer,
        "think": think,
        "usage": usage_dict,
    }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)


    system_prompt = "Отвечай кратко и по делу."
    user_text = "Скажи одним предложением, что такое факторинг."


    res = call_giga_models(
        model="GigaChat-2-Max",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],

        temperature=0.2,
        max_tokens=512,
        top_p=0.9,
        extract_think=False,
    )
    print("\n=== GigaChat-2-Max ===")
    print("THINK:", res["think"])
    print("ANSWER:", res["answer"])
    print(res)

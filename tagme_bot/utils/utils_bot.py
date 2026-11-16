import re as _re
import json as _json
import logging
from langchain_core.prompts import PromptTemplate


logger = logging.getLogger(__name__)

vsellm_client = None


def _extract_first_json_block(text: str) -> str | None:
    if not text:
        return None

    start = None
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            break
    if start is None:
        return None

    stack = []
    in_str = False
    esc = False

    for j in range(start, len(text)):
        ch = text[j]

        if in_str:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch in "{[":
            stack.append(ch)
            continue

        if ch in "}]":
            if not stack:
                return None
            open_ch = stack.pop()
            if (open_ch == "{" and ch != "}") or (open_ch == "[" and ch != "]"):
                return None
            if not stack:
                return text[start:j + 1]

    return None


def _strip_json_fences(text: str) -> str:
    if not text:
        return text
    m = _re.search(r"```json\s*(.*?)\s*```", text, flags=_re.DOTALL | _re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()


def _repair_json_unescaped_quotes(text: str) -> str:

    out = []
    in_str = False
    esc = False

    for i, ch in enumerate(text):
        if not in_str:
            if ch == '"':
                in_str = True
            out.append(ch)
            continue

        if esc:
            out.append(ch)
            esc = False
            continue

        if ch == "\\":
            out.append(ch)
            esc = True
            continue

        if ch == '"':
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if nxt in [",", "}", "]"]:
                in_str = False
                out.append(ch)
            else:
                out.append("«")
            continue

        out.append(ch)

    return "".join(out)


def render_prompt_with_placeholders(template_text: str, data: dict) -> str:
    """
    Через LangChain PromptTemplate подставляем значения из JSON в плейсхолдеры {var}.
    """
    if not template_text or not isinstance(data, dict):
        return template_text

    vars_found = sorted(set(_re.findall(r"{(\w+)}", template_text)))
    if not vars_found:
        return template_text

    values = {name: (data.get(name, "") if data.get(name) is not None else "") for name in vars_found}

    try:
        pt = PromptTemplate(template=template_text, input_variables=vars_found)
        rendered = pt.format(**values)
        return rendered
    except Exception as e:
        logger.error(f"[placeholders] Failed to render prompt with placeholders: {e}")
        return template_text


def extract_json_and_think(raw: str):
    if not isinstance(raw, str) or not raw.strip():
        return None, raw

    text = raw.strip()

    unfenced = _strip_json_fences(text)

    block = _extract_first_json_block(unfenced)
    if not block:
        return None, text

    try:
        obj = _json.loads(block)
        think = (text.replace(block, "", 1)).strip()
        return obj, think
    except Exception:
        pass

    normalized = block.replace('""', '"')
    try:
        obj = _json.loads(normalized)
        think = (text.replace(block, "", 1)).strip()
        return obj, think
    except Exception:
        pass

    repaired = _repair_json_unescaped_quotes(normalized)
    try:
        obj = _json.loads(repaired)
        think = (text.replace(block, "", 1)).strip()
        return obj, think
    except Exception:
        return None, text

def wrap_with_initial_data(payload, js, initial_data_flag: bool):
    if not (initial_data_flag and isinstance(js, dict)):
        return payload

    if isinstance(payload, dict):
        out = js.copy()
        out.update(payload)
        return out

    out = js.copy()
    out["answer"] = payload
    return out
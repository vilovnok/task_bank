#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ReAct-агент на GigaChat с тремя инструментами:
1) web_search  — интернет-поиск (DuckDuckGo + mTLS)

МИНИМАЛЬНЫЙ cfg (только справочник):
{
  "model": "GigaChat-2-Max",
  "tools": ["web_search"]
}
"""

import re
import json
import warnings
from pathlib import Path, PureWindowsPath
from typing import Dict, Any, Optional, List, Tuple

import requests
import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage  # (+SystemMessage)
from langchain.tools import tool
from langchain_gigachat import GigaChat
from langgraph.prebuilt import create_react_agent
from langchain_core.callbacks import BaseCallbackHandler

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from ddgs import DDGS


warnings.simplefilter("ignore")

DEFAULT_BASE_URL = "https://gigachat-ift.sberdevices.delta.sbrf.ru/v1/"
DEFAULT_SYSTEM_PROMPT = "Ты эксперт. Отвечай кратко в формате JSON."

certs_dir      = Path.home() / "certs"
ca_bundle_file = str(certs_dir / "ca.pem")
cert_file      = str(certs_dir / "tls.pem")
key_file       = str(certs_dir / "tls.key")


def _create_session(verify_ssl: bool = False) -> requests.Session:
    sess = requests.Session()
    sess.verify = verify_ssl
    sess.cert = (cert_file, key_file)
    return sess

_HTTP_SESSION = _create_session(verify_ssl=False)


class DuckDuckGoSearchAPIWrapperMTLS(DuckDuckGoSearchAPIWrapper):
    def __init__(
        self,
        region: str = "wt-wt",
        safesearch: str = "moderate",
        time: str = "y",
        max_results: int = 5,
        backend: Optional[str] = None,
    ):
        params = {
            "region": region,
            "safesearch": safesearch,
            "time": time,
            "max_results": max_results,
        }
        if backend:
            params["backend"] = backend
        super().__init__(**params)
        self._session = _HTTP_SESSION

    def _get_ddgs(self):
        return DDGS(verify=self._session.verify, timeout=10, proxy=None)

    def _ddgs_text(self, query: str, max_results=None):
        ddgs = self._get_ddgs()
        ddgs.session = self._session
        with ddgs:
            return list(ddgs.text(
                query,
                region=self.region,
                safesearch=self.safesearch,
                timelimit=self.time,
                max_results=max_results or self.max_results,
                backend=self.backend,
            ))


def make_web_search_tool():
    wrapper = DuckDuckGoSearchAPIWrapperMTLS()
    runner = DuckDuckGoSearchRun(api_wrapper=wrapper)

    @tool("web_search")
    def web_search(query: str) -> str:
        """Веб-поиск (DuckDuckGo). Возвращает сводку результатов."""
        return runner.run(query)

    return web_search


class ToolLogger(BaseCallbackHandler):
    def on_tool_start(self, serialized, input_str, **kwargs):
        name = (serialized or {}).get("name") or "<tool?>"
        print(f"[TOOL START] {name}  args={input_str}")

    def on_tool_end(self, output, **kwargs):
        out = str(output)
        if len(out) > 600:
            out = out[:600] + "…"
        print(f"[TOOL END  ] {out}")

    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"[LLM START] prompts: {len(prompts)}")

    def on_llm_end(self, response, **kwargs):
        print("[LLM END]")


def build_agent(conf: Dict[str, Any]):
    """
    Минимальный контракт cfg:
      - model: str (обязательно)
      - tools: list[str] из {"web_search","docs_search","topic_rule"} (опц.)
      - reference_rel_path: str (если используешь "topic_rule")
      - docs_rel_path: str и rag_folder: str (если используешь "docs_search")
      - base_url: str (опц.)
      - recursion_limit: int (опц., по умолчанию 10)
    """
    llm = GigaChat(
        credentials=None,
        model=conf["model"],
        base_url=conf.get("base_url", DEFAULT_BASE_URL),
        ca_bundle_file=ca_bundle_file,
        cert_file=cert_file,
        key_file=key_file,
        timeout=200.0,
        temperature=0,
        profanity_check=False,
        max_tokens=100000
    )

    requested_tools = conf.get("tools")
    tools = []

    if (requested_tools is None) or ("web_search" in requested_tools):
        tools.append(make_web_search_tool())

    return create_react_agent(llm, tools=tools)


def _clean_json(raw: str) -> str:
    cleaned = re.sub(r"```json\s*", "", raw, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "")
    cleaned = re.sub(r"^\s*json\s*\n", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()




def _is_deepseek_model(conf: Dict[str, Any]) -> bool:
    m = str(conf.get("model", "")).lower()
    return ("deepseek" in m) or ("reasoning" in m) or ("qwen" in m)

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)

def _split_think(text: str) -> Tuple[str, str]:
    if not text:
        return "", ""
    parts = _THINK_RE.findall(text)
    think = "\n\n".join([p.strip() for p in parts]) if parts else ""
    cleaned = _THINK_RE.sub("", text).strip()
    return cleaned, think

class TraceLogger(BaseCallbackHandler):
    def __init__(self):
        self.events: List[Dict[str, str]] = []

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = (serialized or {}).get("name") or "<tool?>"
        self.events.append({"type": "start", "tool": name, "input": str(input_str)})

    def on_tool_end(self, output, **kwargs):
        out = str(output)
        if len(out) > 4000:
            out = out[:4000] + "…"
        self.events.append({"type": "end", "output": out})

    def export_trace(self) -> List[Dict[str, str]]:
        trace: List[Dict[str, str]] = []
        current: Optional[Dict[str, str]] = None
        for ev in self.events:
            if ev["type"] == "start":
                current = {"tool": ev["tool"], "input": ev["input"], "output": ""}
            elif ev["type"] == "end" and current is not None:
                current["output"] = ev["output"]
                trace.append(current)
                current = None
        return trace


def _build_tools_from_conf(conf: Dict[str, Any]):
    tools = []
    requested_tools = conf.get("tools")

    if (requested_tools is None) or ("web_search" in requested_tools):
        tools.append(make_web_search_tool())

    return tools


def _build_gigachat_llm_for_tools(conf: Dict[str, Any]) -> GigaChat:
    return GigaChat(
        credentials=None,
        model="GigaChat-2-Max",
        base_url=conf.get("base_url", DEFAULT_BASE_URL),
        ca_bundle_file=ca_bundle_file,
        cert_file=cert_file,
        key_file=key_file,
        timeout=200.0,
        temperature=0,
        profanity_check=False,
        max_tokens=1000000
    )


def _run_react_tools_with_gigachat(conf: Dict[str, Any], system_instruction: str, user_input: str) -> Dict[str, Any]:
    tools = _build_tools_from_conf(conf)
    llm_tools = _build_gigachat_llm_for_tools(conf)
    agent = create_react_agent(llm_tools, tools=tools)

    tracer = TraceLogger()
    res = agent.invoke(
        {"messages": [("system", system_instruction), ("user", user_input)]},
        {"recursion_limit": int(conf.get("recursion_limit", 10)), "callbacks": [tracer]}
    )

    msgs: List[Dict[str, str]] = []
    for m in res.get("messages", []):
        if isinstance(m, HumanMessage):
            msgs.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            msgs.append({"role": "assistant", "content": m.content})
        else:
            msgs.append({"role": "tool", "content": getattr(m, "content", str(m))})

    assistant_last = next((m["content"] for m in reversed(msgs) if m["role"] == "assistant"), "")
    steps_str = str(res.get("agent_scratchpad") or res.get("log") or res)
    tool_trace = tracer.export_trace()

    return {
        "messages": msgs,
        "assistant_last": assistant_last,
        "steps_str": steps_str,
        "tool_trace": tool_trace,
    }


def _mk_handoff_prompt(system_instruction: str, user_input: str, tool_trace: List[Dict[str, str]], assistant_last: str, steps_str: str) -> Tuple[str, str]:
    sys = (
        "Ты — финализатор. На основе контекста ниже верни СТРОГИЙ JSON.\n"
        "Не используй инструменты, не пиши лишний текст."
    )
    parts: List[str] = []
    parts.append("== Исходная задача (system) ==")
    parts.append(system_instruction.strip())
    parts.append("\n== Ввод пользователя ==")
    parts.append(user_input.strip())
    parts.append("\n== Вызовы инструментов (GigaChat ReAct) ==")
    for i, ev in enumerate(tool_trace, 1):
        parts.append(f"[{i}] TOOL: {ev['tool']}\n    INPUT : {ev['input']}\n    OUTPUT: {ev['output']}\n")
    parts.append("\n== Последний ответ GigaChat ==")
    parts.append(assistant_last.strip())
    if steps_str:
        short = steps_str if len(steps_str) <= 2000 else steps_str[:2000] + "…"
        parts.append("\n== Технический лог (усечённо) ==")
        parts.append(short)
    parts.append(
        "\n== Формат финального ответа (СТРОГО JSON) ==\n"
        "Верни только JSON."
    )
    return sys, "\n".join(parts)


def _deepseek_finalize(conf: Dict[str, Any], system_instruction: str, user_input: str, handoff: Dict[str, Any]) -> Dict[str, Any]:
    ds_system, ds_user = _mk_handoff_prompt(
        system_instruction=system_instruction,
        user_input=user_input,
        tool_trace=handoff["tool_trace"],
        assistant_last=handoff["assistant_last"],
        steps_str=handoff["steps_str"],
    )
    ds_llm = GigaChat(
        credentials=None,
        model=conf["model"],
        base_url=conf.get("base_url", DEFAULT_BASE_URL),
        ca_bundle_file=ca_bundle_file,
        cert_file=cert_file,
        key_file=key_file,
        timeout=200.0,
        temperature=0,
        profanity_check=False,
        max_tokens=4000
    )
    resp: AIMessage = ds_llm.invoke([SystemMessage(content=ds_system), HumanMessage(content=ds_user)])
    raw = (resp.content or "").strip()


    cleaned, think_text = _split_think(raw)

    cleaned2 = _clean_json(cleaned)
    m = re.search(r"\{[\s\S]*\}", cleaned2)
    json_str = m.group(0) if m else cleaned2

    try:
        parsed: Any = json.loads(json_str)
        if not isinstance(parsed, dict):
            parsed = {"result": parsed}
    except json.JSONDecodeError:
        parsed = {"model_answer": cleaned2}

    if conf.get("think") and think_text:
        if isinstance(parsed, dict) and "think" not in parsed:
            parsed["think"] = think_text

    return {
        "final_payload": parsed,
        "raw_deepseek_response": raw,
    }




def run_react_agent(
    conf: Dict[str, Any],
    system_instruction: str,
    user_input: str
) -> Dict[str, Any]:

    if _is_deepseek_model(conf):
        handoff = _run_react_tools_with_gigachat(conf, system_instruction, user_input)
        ds = _deepseek_finalize(conf, system_instruction, user_input, handoff)
        payload = ds["final_payload"]
        if conf.get("minimal_payload", False):
            return payload
        out = dict(payload)
        out.update({
            "user_input": user_input,
            "system_prompt": system_instruction,
            "messages": handoff["messages"],
            "agent_steps": handoff["steps_str"],
            "tool_trace": handoff["tool_trace"],
            "raw_agent_response": ds["raw_deepseek_response"],
        })
        return out


    agent = build_agent(conf)
    recursion_limit = int(conf.get("recursion_limit", 10))
    callbacks = [ToolLogger()]

    agent_ret = agent.invoke(
        {"messages": [("system", system_instruction), ("user", user_input)]},
        {"recursion_limit": recursion_limit, "callbacks": callbacks}
    )

    msgs = []
    for m in agent_ret.get("messages", []):
        if isinstance(m, HumanMessage):
            msgs.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            msgs.append({"role": "assistant", "content": m.content})
        else:
            msgs.append({"role": "tool", "content": getattr(m, "content", str(m))})

    steps_str = str(agent_ret.get("agent_scratchpad") or agent_ret.get("log") or agent_ret)
    last_content = msgs[-1]["content"] if msgs else ""

    cleaned = _clean_json(last_content)
    m = re.search(r"\{[\s\S]*\}", cleaned)
    json_str = m.group(0) if m else cleaned

    try:
        parsed = json.loads(json_str)
        if not isinstance(parsed, dict):
            parsed = {"result": parsed}
    except json.JSONDecodeError:
        parsed = {"model_answer": last_content}

    if conf.get("minimal_payload", False):
        return parsed

    payload = parsed
    payload.update({
        "user_input": user_input,
        "system_prompt": system_instruction,
        "messages": msgs,
        "agent_steps": steps_str,
        "raw_agent_response": last_content,
    })
    return payload


if __name__ == "__main__":

    cfg = {
        "model": "Qwen",  # либо GigaChat-2-Max; при DeepSeek включается оркестратор
        "tools": ["web_search"],
        "minimal_payload": False,
        "think": False

        # опционально:
        # "recursion_limit": 10,
        # "minimal_payload": False,
        # "base_url": DEFAULT_BASE_URL,
        # "think": True,  # при DeepSeek добавить ключ "think" в JSON
        # для RAG (если добавишь "docs_search" в tools):
        # "docs_rel_path": r"ivr_csc\docs",
        # "rag_folder": "my_rag_cache"
    }

    system_instruction = r"""
Выполняй указания пользователя. Всю информацию ищи в интернете.
""".strip()

    user_input = """
Какой курс доллара на сегодня?
""".strip()

    res = run_react_agent(cfg, system_instruction, user_input)
    print(json.dumps(res, ensure_ascii=False, indent=2))
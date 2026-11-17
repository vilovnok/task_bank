#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Refactored: ThreadPoolExecutor for parallel API calls.
Thread-safe logging, robust error handling, KeyboardInterrupt safety.

Usage:
    python evaluate_agent_classification.py -w 20  # 20 parallel workers
"""

import argparse
import csv
import json
import os
import requests
import sys
import time
import threading
import concurrent.futures
from pathlib import Path
from datetime import datetime, timezone

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data" / "tagme_balanced_small_dataset.csv"
SYSTEM_PROMPT_PATH = BASE_DIR / "data" / "task_mank_system_prompt_2_5.md"

sys.path.insert(0, str(BASE_DIR.parent))
sys.path.insert(0, str(BASE_DIR))
from services_t import ENDPOINTS, get_token

SERVICE = "vsellm"
MODEL = "anthropic/claude-opus-5"
TEMPERATURE = 0.1
MAX_TOKENS = 1024
URL = ENDPOINTS[SERVICE]

MAX_RETRIES = 3
RETRY_DELAY = 120

def load_system_prompt() -> str:
    """Load system prompt from markdown file."""
    with open(SYSTEM_PROMPT_PATH, encoding="utf-8") as f:
        return f.read()


def extract_target_messages(dialog_json: dict) -> list[dict]:
    """Extract and sort target messages from dialog JSON."""
    messages = dialog_json.get("answer", [])
    messages.sort(key=lambda m: int(m.get("Key", 0)), reverse=True)
    return messages


def parse_model_response(content: str) -> dict | None:
    """Parse model response JSON; return None on error."""
    try:
        data = json.loads(content)
        return data if "agent" in data else None
    except (json.JSONDecodeError, TypeError):
        return None


def get_content_from_response(data: dict) -> str:
    """Extract content from response - same logic as services_t.py chat()."""
    choices = data.get("choices") or []
    c0 = choices[0] if choices else {}
    if "messages" in c0:
        content = (c0["messages"][0] or {}).get("content", "")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content)
    else:
        content = (c0.get("message") or {}).get("content", "")
    return content


def log_request_response(log_path, row_index, chat_session_id,
                         payload, response, error, status):
    """Append a JSONL line to the request log (thread-safe via caller)."""
    entry = {
        "row_index": row_index,
        "chat_session_id": chat_session_id,
        "payload": payload,
        "response": response,
        "error": error,
        "status": status,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
TIMESTAMP_STR = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
REQUEST_LOG_PATH = RESULTS_DIR / f"requests_log_{TIMESTAMP_STR}.jsonl"
log_lock = threading.Lock()

def process_row(row, row_index, system_prompt, headers, url):
    """Process a single CSV row in a thread."""
    i = row_index
    input_raw = row.get("INPUT:input", "").strip()
    expected_agent = row.get("GOLDEN:agent", "").strip()
    try:
        dialog = json.loads(input_raw)
        messages = extract_target_messages(dialog)
        user_msg = json.dumps({"dialog": messages}, ensure_ascii=False, indent=2)
    except Exception as e:
        error_str = f"Invalid JSON: {e}"
        with log_lock:
            log_request_response(REQUEST_LOG_PATH, i, "unknown", None, None,
                                 error_str, "json_error")
        return {
            "row_index": i, "chat_session_id": "unknown",
            "dialog_snippet": "", "expected_agent": expected_agent,
            "predicted_agent": "", "score": 0,
            "full_response": "", "status": "json_error",
        }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }

    chat_session_id = "unknown"
    dialog_snippet = ""
    try:
        ml = dialog.get("answer", [])
        if ml:
            chat_session_id = ml[0].get("ID_chat-sessii", "unknown")
            first = ml[0]
            dialog_snippet = f"{first.get('Source', '')}: {first.get('Text', '')}"[:200]
    except (AttributeError, IndexError):
        pass
    return _api_call_and_evaluate(i, payload, chat_session_id, dialog_snippet,
                                  expected_agent, headers, url, REQUEST_LOG_PATH)

def _api_call_and_evaluate(i, payload, chat_session_id, dialog_snippet,
                           expected_agent, headers, url, log_path):
    answer = None
    response_json = None
    error_msg = None
    status = "no_response"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload,
                                 verify=False, timeout=(480, 480))
            resp.raise_for_status()
            response_json = resp.json()
            answer = get_content_from_response(response_json)
            break
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                status = "all_retries_failed"
    if answer is None:
        predicted_agent = ""
        score = 0
        if status == "no_response":
            status = "all_retries_failed"
    else:
        answer_data = parse_model_response(answer)
        if answer_data is not None:
            predicted_agent = answer_data.get("agent", "")
            score = 1 if predicted_agent == expected_agent else 0
            status = "success"
        else:
            predicted_agent = ""
            score = 0
            status = "parse_error"
    with log_lock:
        log_request_response(log_path, i, chat_session_id, payload,
                             response_json, error_msg, status)
    return {"row_index": i, "chat_session_id": chat_session_id,
            "dialog_snippet": dialog_snippet, "expected_agent": expected_agent,
            "predicted_agent": predicted_agent, "score": score,
            "full_response": answer or "", "status": status}

def main():
    """Main loop: read CSV, process in parallel, write results."""
    # Load system prompt and token (once, not per-thread)
    system_prompt = load_system_prompt()
    headers = get_token(SERVICE)

    # Read CSV
    if not CSV_PATH.exists():
        print(f"Error: CSV not found at {CSV_PATH}")
        sys.exit(1)

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Parse CLI arguments
    parser = argparse.ArgumentParser(
        description="Evaluate agent classification results from CSV"
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=10,
        help="Number of parallel workers (default: 10)"
    )
    args = parser.parse_args()
    worker_count = args.workers

    if worker_count < 1:
        parser.error("--workers must be >= 1")

    print(f"Workers: {worker_count}")
    print("=" * 80)
    print(f"{'Row':>5}  {'Chat Session ID':>24}  {'Expected':>8}  {'Predicted':>8}  {'Score':>5}  Status")
    print("=" * 80)

    # Process rows in parallel
    results = []
    total = len(rows)

    def process_and_collect(idx, row):
        return process_row(row, idx, system_prompt, headers, URL)

    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(process_and_collect, idx, row): idx
            for idx, row in enumerate(rows)
        }
        done = 0
        for future in concurrent.futures.as_completed(futures):
            done += 1
            idx = futures[future]
            try:
                result = future.result()
                if result is None:
                    result = {
                        "row_index": idx, "chat_session_id": "unknown",
                        "dialog_snippet": "", "expected_agent": "",
                        "predicted_agent": "", "score": 0,
                        "full_response": "", "status": "json_error",
                    }
                results.append(result)
                print(
                    f"{idx:>5}  {result['chat_session_id']:>24}  "
                    f"{result['expected_agent']:>8}  {result['predicted_agent']:>8}  "
                    f"{result['score']:>5}  {result.get('status', '?')}"
                )
            except Exception as e:
                print(f"  ERROR processing row {idx}: {e}")
                results.append({
                    "row_index": idx, "chat_session_id": "unknown",
                    "dialog_snippet": "", "expected_agent": "",
                    "predicted_agent": "", "score": 0,
                    "full_response": "", "status": "thread_error",
                })

    # Sort results by row_index for clean output
    results.sort(key=lambda r: r["row_index"])

    # Write results CSV
    results_path = RESULTS_DIR / f"results_{TIMESTAMP_STR}.csv"
    fieldnames = ["row_index", "chat_session_id", "dialog_snippet",
                  "expected_agent", "predicted_agent", "score",
                  "full_response", "status"]
    with open(results_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Print summary
    total_correct = sum(r["score"] for r in results)
    status_counts = {}
    for r in results:
        st = r.get("status", "unknown")
        status_counts[st] = status_counts.get(st, 0) + 1

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total processed: {len(results)}/{total}")
    print(f"Correct:         {total_correct}")
    print(f"Accuracy:        {total_correct / len(results) * 100:.2f}%" if results else "N/A")
    print("Status breakdown:")
    for st, cnt in sorted(status_counts.items()):
        print(f"  {st}: {cnt}")
    print(f"Results saved to: {results_path}")
    print(f"Request log:      {REQUEST_LOG_PATH}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving partial results...")
        sys.exit(130)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple script: read system prompt + CSV, call vsellm, print model answers.
Uses the same request structure as services_t.py chat().
"""

import csv
import json
import requests
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data" / "tagme_balanced_small_dataset.csv"
SYSTEM_PROMPT_PATH = BASE_DIR / "data" / "task_mank_system_prompt_2_5.md"

sys.path.insert(0, str(BASE_DIR.parent))  # task_bank/ for services_t.py
sys.path.insert(0, str(BASE_DIR))          # local_work/ for data
from services_t import ENDPOINTS, get_token

SERVICE = "vsellm"
MODEL = "anthropic/claude-opus-5"
TEMPERATURE = 0.1
MAX_TOKENS = 1024
URL = ENDPOINTS[SERVICE]

MAX_RETRIES = 3
RETRY_DELAY = 120


def load_system_prompt() -> str:
    with open(SYSTEM_PROMPT_PATH, encoding="utf-8") as f:
        return f.read()


def extract_target_messages(dialog_json: dict) -> list[dict]:
    messages = dialog_json.get("answer", [])
    messages.sort(key=lambda m: int(m.get("Ключ для разметки", 0)), reverse=True)
    return messages


def parse_model_response(content: str) -> dict | None:
    """Parse model response JSON and return structured data, or None on error."""
    try:
        data = json.loads(content)
        if "agent" not in data:
            return None
        return data
    except (json.JSONDecodeError, TypeError):
        return None


def get_content_from_response(data: dict) -> str:
    """Extract content from response — same logic as services_t.py chat()."""
    choices = data.get("choices") or []
    c0 = choices[0] if choices else {}
    if "messages" in c0:
        content = (c0["messages"][0] or {}).get("content", "")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content)
    else:
        content = (c0.get("message") or {}).get("content", "")
    return content


def main():
    system_prompt = load_system_prompt()

    if not CSV_PATH.exists():
        print(f"CSV not found: {CSV_PATH}")
        sys.exit(1)

    token = get_token(SERVICE)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    with open(CSV_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Rows: {len(rows)} | Model: {MODEL} | Service: {SERVICE}\n")

    # --- Results writing setup ---
    RESULTS_DIR = BASE_DIR / "results"
    RESULTS_DIR.mkdir(exist_ok=True)
    RESULTS_PATH = RESULTS_DIR / "evaluation_results.csv"
    results_headers = [
        "row_index", "chat_session_id", "dialog_snippet",
        "expected_agent", "predicted_agent", "score",
        "full_response", "status",
    ]
    results_rows: list[dict] = []

    for i, row in enumerate(rows, 1):
        gold = row.get("GOLDEN:agent", "").strip()
        input_raw = row.get("INPUT:input", "").strip()

        try:
            dialog = json.loads(input_raw)
            messages = extract_target_messages(dialog)
            user_msg = json.dumps({"dialog": messages}, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[{i}/{len(rows)}] SKIP: {e}\n")
            continue

        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
        }

        answer = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    URL,
                    headers=headers,
                    json=payload,
                    verify=False,
                    timeout=(60, 180),
                )
                resp.raise_for_status()
                answer = get_content_from_response(resp.json())
                print(f"[{i}/{len(rows)}] Attempt {attempt}/{MAX_RETRIES} succeeded")
                break
            except Exception as e:
                if attempt < MAX_RETRIES:
                    print(f"[{i}/{len(rows)}] Attempt {attempt}/{MAX_RETRIES} failed: {e}")
                    print(f"  Retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"[{i}/{len(rows)}] All {MAX_RETRIES} attempts failed: {e}")

        # --- Evaluate the model response ---
        chat_session_id = "unknown"
        dialog_snippet = ""
        try:
            messages_list = dialog.get("answer", [])
            if messages_list:
                chat_session_id = messages_list[0].get("ID чат-сессии", "unknown")
                first_msg = messages_list[0]
                dialog_snippet = f"{first_msg.get('Источник сообщения', '')}: {first_msg.get('Текст сообщения', '')}"[:200]
        except (AttributeError, IndexError):
            pass

        if answer is None:
            predicted_agent = ""
            expected_agent = row.get("OUTPUT:agent", "").strip()
            score = 0
            status = "all_retries_failed"
        else:
            answer_data = parse_model_response(answer)

            if answer_data is not None:
                predicted_agent = answer_data.get("agent", "")
                expected_agent = row.get("OUTPUT:agent", "").strip()
                score = 1 if predicted_agent == expected_agent else 0
                status = "success"
            else:
                predicted_agent = ""
                expected_agent = row.get("OUTPUT:agent", "").strip()
                score = 0
                status = "parse_error"

        # Write result row
        results_rows.append({
            "row_index": i,
            "chat_session_id": chat_session_id,
            "dialog_snippet": dialog_snippet,
            "expected_agent": expected_agent,
            "predicted_agent": predicted_agent,
            "score": score,
            "full_response": answer,
            "status": status,
        })

        # Print progress
        print(f"[{i}/{len(rows)}] exp={expected_agent} | pred={predicted_agent} | score={score} | status={status}")

    # --- Write results to CSV ---
    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results_headers)
        writer.writeheader()
        writer.writerows(results_rows)

    # --- Print summary statistics ---
    total = len(results_rows)
    successful = sum(1 for r in results_rows if r["status"] == "success")
    matches = sum(1 for r in results_rows if r["score"] == 1)
    mismatches = sum(1 for r in results_rows if r["score"] == 0)
    parse_errors = sum(1 for r in results_rows if r["status"] == "parse_error")
    retries_failed = sum(1 for r in results_rows if r["status"] == "all_retries_failed")

    print(f"\n{'=' * 50}")
    print(f"EVALUATION SUMMARY")
    print(f"{'=' * 50}")
    print(f"Total rows:        {total}")
    print(f"Successful:        {successful}")
    print(f"Matches (score=1): {matches}")
    print(f"Mismatches (0):    {mismatches}")
    print(f"Parse errors:      {parse_errors}")
    print(f"Retries failed:    {retries_failed}")
    if successful > 0:
        accuracy = matches / successful
        print(f"Accuracy:          {accuracy:.2%} ({matches}/{successful})")
    print(f"\nResults saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
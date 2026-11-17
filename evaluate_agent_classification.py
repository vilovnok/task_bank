
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parallel dataset evaluation using VSELLM.

Usage:
    python evaluate.py
    python evaluate.py -w 10
    python evaluate.py -w 20 --input data/dataset.csv
"""

import argparse
import csv
import json
import logging
import requests
import sys
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import local
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = BASE_DIR / "data" / "tagme_balanced_small_dataset_v2.csv"
SYSTEM_PROMPT_PATH = BASE_DIR / "data" / "task_mank_system_prompt_2_6.md"

sys.path.insert(0, str(BASE_DIR.parent))
sys.path.insert(0, str(BASE_DIR))

from services_t import ENDPOINTS, get_token


# ============================================================
# Configuration
# ============================================================

SERVICE = "vsellm"
MODEL = "anthropic/claude-opus-5"

TEMPERATURE = 0.1
# MAX_TOKENS = 1024

URL = ENDPOINTS[SERVICE]

MAX_RETRIES = 3
RETRY_DELAY = 120

REQUEST_TIMEOUT = (60, 180)

RESULTS_DIR = BASE_DIR / "results"

_thread_local = local()


# ============================================================
# Logging
# ============================================================

def setup_logging():
    RESULTS_DIR.mkdir(exist_ok=True)

    log_path = RESULTS_DIR / (
        f"evaluation_{datetime.now():%Y%m%d_%H%M%S}.log"
    )

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)-8s | "
            "%(threadName)s | %(message)s"
        ),
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info("Logging initialized: %s", log_path)

    return log_path


# ============================================================
# HTTP session
# ============================================================

def get_session():
    """
    Each worker gets its own HTTP session.
    Sessions are reused between requests.
    """
    if not hasattr(_thread_local, "session"):
        session = requests.Session()

        token = get_token(SERVICE)

        session.headers.update({
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        })

        _thread_local.session = session

    return _thread_local.session


# ============================================================
# Data processing
# ============================================================

def load_system_prompt() -> str:
    with open(SYSTEM_PROMPT_PATH, encoding="utf-8") as f:
        return f.read()


def extract_target_messages(dialog_json: dict) -> list[dict]:
    messages = dialog_json.get("answer", [])

    return sorted(
        messages,
        key=lambda m: int(m.get("Ключ для разметки", 0)),
        reverse=True,
    )


def parse_model_response(content: str) -> dict | None:
    try:
        data = json.loads(content)

        if "agent" not in data:
            return None

        return data

    except (json.JSONDecodeError, TypeError):
        return None


def get_content_from_response(data: dict) -> str:
    choices = data.get("choices") or []
    c0 = choices[0] if choices else {}

    if "messages" in c0:
        content = (c0["messages"][0] or {}).get("content", "")

        if isinstance(content, list):
            content = " ".join(
                p.get("text", "")
                for p in content
                if isinstance(p, dict)
            )

    else:
        content = (c0.get("message") or {}).get("content", "")

    return content


# ============================================================
# Single row processing
# ============================================================

def process_row(task: tuple, total: int, system_prompt: str) -> dict:
    """
    Process one dataset row.

    This function runs independently in each worker.
    """

    row_index, row = task

    start_time = time.perf_counter()

    expected_agent = row.get("GOLDEN:agent", "").strip()
    input_raw = row.get("INPUT:input", "").strip()

    chat_session_id = "unknown"
    dialog_snippet = ""

    answer = None
    status = "unknown"
    predicted_agent = ""
    score = 0

    # --------------------------------------------------------
    # Parse input
    # --------------------------------------------------------

    try:
        dialog = json.loads(input_raw)

        messages = extract_target_messages(dialog)

        user_msg = json.dumps(
            {"dialog": messages},
            ensure_ascii=False,
            indent=2,
        )

        messages_list = dialog.get("answer", [])

        if messages_list:
            chat_session_id = messages_list[0].get(
                "ID чат-сессии", "unknown"
            )

            first_msg = messages_list[0]

            dialog_snippet = (
                f"{first_msg.get('Источник сообщения', '')}: "
                f"{first_msg.get('Текст сообщения', '')}"
            )[:200]

    except Exception as e:
        logging.exception(
            "[%s/%s] Input parsing failed: %s",
            row_index,
            total,
            e,
        )

        return {
            "row_index": row_index,
            "chat_session_id": chat_session_id,
            "dialog_snippet": dialog_snippet,
            "expected_agent": expected_agent,
            "predicted_agent": "",
            "score": 0,
            "full_response": "",
            "status": "input_error",
            "attempts": 0,
            "duration_sec": round(
                time.perf_counter() - start_time, 2
            ),
        }

    # --------------------------------------------------------
    # Prepare request
    # --------------------------------------------------------

    # payload = {
    #     "model": MODEL,
    #     "messages": [
    #         {
    #             "role": "system",
    #             "content": system_prompt,
    #         },
    #         {
    #             "role": "user",
    #             "content": user_msg,
    #         },
    #     ],
    #     "temperature": TEMPERATURE,
    #     # "max_tokens": MAX_TOKENS,
    # }

    payload = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ],
    "temperature": TEMPERATURE,
    # "max_tokens": 8192,

    "thinking": {
        "type": "adaptive",
        "display": "summarized"
    },

    "output_config": {
        "effort": "medium"
    }
}

    session = get_session()

    # --------------------------------------------------------
    # Send request with retries
    # --------------------------------------------------------

    attempts = 0

    for attempt in range(1, MAX_RETRIES + 1):
        attempts = attempt

        request_start = time.perf_counter()

        logging.info(
            "[%s/%s] Sending request | attempt=%s/%s | session=%s",
            row_index,
            total,
            attempt,
            MAX_RETRIES,
            chat_session_id,
        )

        try:
            response = session.post(
                URL,
                json=payload,
                verify=False,
                timeout=REQUEST_TIMEOUT,
            )

            response.raise_for_status()

            response_data = response.json()

            answer = get_content_from_response(response_data)

            request_duration = time.perf_counter() - request_start

            logging.info(
                "[%s/%s] Request succeeded | HTTP=%s | "
                "duration=%.2fs | attempt=%s",
                row_index,
                total,
                response.status_code,
                request_duration,
                attempt,
            )

            # Log model response.
            logging.debug(
                "[%s/%s] Model response: %s",
                row_index,
                total,
                answer,
            )

            break

        except Exception as e:
            request_duration = time.perf_counter() - request_start

            logging.warning(
                "[%s/%s] Request failed | attempt=%s/%s | "
                "duration=%.2fs | error=%s",
                row_index,
                total,
                attempt,
                MAX_RETRIES,
                request_duration,
                e,
            )

            if attempt < MAX_RETRIES:
                logging.info(
                    "[%s/%s] Retrying in %s seconds",
                    row_index,
                    total,
                    RETRY_DELAY,
                )

                time.sleep(RETRY_DELAY)

            else:
                logging.error(
                    "[%s/%s] All retries failed",
                    row_index,
                    total,
                )

    # --------------------------------------------------------
    # Evaluate model response
    # --------------------------------------------------------

    if answer is None:
        status = "all_retries_failed"

    else:
        answer_data = parse_model_response(answer)

        if answer_data is not None:
            predicted_agent = str(
                answer_data.get("agent", "")
            ).strip()

            score = int(predicted_agent == expected_agent)

            status = "success"

        else:
            status = "parse_error"

    duration = time.perf_counter() - start_time

    result = {
        "row_index": row_index,
        "chat_session_id": chat_session_id,
        "dialog_snippet": dialog_snippet,
        "expected_agent": expected_agent,
        "predicted_agent": predicted_agent,
        "score": score,
        "full_response": answer or "",
        "status": status,
        "attempts": attempts,
        "duration_sec": round(duration, 2),
    }

    logging.info(
        "[%s/%s] Completed | exp=%s | pred=%s | "
        "score=%s | status=%s | duration=%.2fs",
        row_index,
        total,
        expected_agent,
        predicted_agent,
        score,
        status,
        duration,
    )

    return result


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Parallel VSELLM dataset evaluation"
    )

    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=5,
        help="Number of parallel workers (default: 5)",
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=CSV_PATH,
        help="Input dataset CSV path",
    )

    args = parser.parse_args()

    if args.workers < 1:
        parser.error("Workers must be greater than 0")

    if not args.input.exists():
        print(f"CSV not found: {args.input}")
        sys.exit(1)

    log_path = setup_logging()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    system_prompt = load_system_prompt()

    with open(
        args.input,
        encoding="utf-8-sig",
        newline="",
    ) as f:
        rows = list(csv.DictReader(f))

    total = len(rows)

    logging.info("=" * 60)
    logging.info("STARTING EVALUATION")
    logging.info("Rows: %s", total)
    logging.info("Workers: %s", args.workers)
    logging.info("Model: %s", MODEL)
    logging.info("Service: %s", SERVICE)
    logging.info("=" * 60)

    # --------------------------------------------------------
    # Results setup
    # --------------------------------------------------------

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_path = (
        RESULTS_DIR / f"evaluation_results_{timestamp}.csv"
    )

    results_headers = [
        "row_index",
        "chat_session_id",
        "dialog_snippet",
        "expected_agent",
        "predicted_agent",
        "score",
        "full_response",
        "status",
        "attempts",
        "duration_sec",
    ]

    results_rows = []

    # --------------------------------------------------------
    # Parallel processing
    # --------------------------------------------------------

    tasks = [
        (i, row)
        for i, row in enumerate(rows, start=1)
    ]

    completed = 0

    with ThreadPoolExecutor(
        max_workers=args.workers,
        thread_name_prefix="worker",
    ) as executor:

        futures = {
            executor.submit(
                process_row,
                task,
                total,
                system_prompt,
            ): task[0]
            for task in tasks
        }

        for future in as_completed(futures):

            row_index = futures[future]

            try:
                result = future.result()

            except Exception as e:
                logging.exception(
                    "[%s/%s] Unexpected worker error",
                    row_index,
                    total,
                )

                result = {
                    "row_index": row_index,
                    "chat_session_id": "unknown",
                    "dialog_snippet": "",
                    "expected_agent": rows[row_index - 1].get(
                        "GOLDEN:agent", ""
                    ).strip(),
                    "predicted_agent": "",
                    "score": 0,
                    "full_response": str(e),
                    "status": "worker_error",
                    "attempts": 0,
                    "duration_sec": 0,
                }

            results_rows.append(result)

            completed += 1

            print(
                f"\rProgress: {completed}/{total} "
                f"({completed / total:.1%})",
                end="",
                flush=True,
            )

    print()

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results_rows.sort(key=lambda r: r["row_index"])

    with open(
        results_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=results_headers,
        )

        writer.writeheader()
        writer.writerows(results_rows)

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    successful = sum(
        r["status"] == "success"
        for r in results_rows
    )

    matches = sum(
        r["score"] == 1
        for r in results_rows
    )

    mismatches = sum(
        r["status"] == "success" and r["score"] == 0
        for r in results_rows
    )

    parse_errors = sum(
        r["status"] == "parse_error"
        for r in results_rows
    )

    retries_failed = sum(
        r["status"] == "all_retries_failed"
        for r in results_rows
    )

    input_errors = sum(
        r["status"] == "input_error"
        for r in results_rows
    )

    worker_errors = sum(
        r["status"] == "worker_error"
        for r in results_rows
    )

    logging.info("=" * 60)
    logging.info("EVALUATION SUMMARY")
    logging.info("=" * 60)
    logging.info("Total rows:        %s", total)
    logging.info("Successful:        %s", successful)
    logging.info("Matches:           %s", matches)
    logging.info("Mismatches:        %s", mismatches)
    logging.info("Parse errors:      %s", parse_errors)
    logging.info("Retries failed:    %s", retries_failed)
    logging.info("Input errors:      %s", input_errors)
    logging.info("Worker errors:     %s", worker_errors)

    if successful:
        accuracy = matches / successful

        logging.info(
            "Accuracy:          %.2f%% (%s/%s)",
            accuracy * 100,
            matches,
            successful,
        )

    logging.info("Results: %s", results_path)
    logging.info("Log: %s", log_path)
    logging.info("=" * 60)


if __name__ == "__main__":
    main()

# python evaluate_agent_classification.py -w 10
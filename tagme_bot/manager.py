import asyncio
import json
import os
import subprocess
import logging
import http.client
import atexit
import argparse
import sys
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import urllib3

from crowd_sdk.tagme import TagmeClientAdvanced
from utils.config import get_stand

http.client.HTTPConnection.debuglevel = 0

logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("crowd_sdk").setLevel(logging.WARNING)
logging.getLogger("crowd_sdk.core.utils_agent.http_client").setLevel(logging.WARNING)
logging.getLogger("aiohttp").setLevel(logging.WARNING)

load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


ANNOTATOR_SCRIPT = os.getenv("ANNOTATOR_SCRIPT", "tagme_bot_script.py")

MAX_PROJECT_PROCESSES = int(os.getenv("MAX_PROJECT_PROCESSES", "100"))

PAGES = int(os.getenv("TAGME_PAGES", "10"))
PAGE_SIZE = int(os.getenv("TAGME_PAGE_SIZE", "50"))

SCAN_EVERY_SEC = int(os.getenv("TAGME_SCAN_EVERY_SEC", "30"))

SKIP_PROJECT_NAME = (os.getenv("SKIP_PROJECT_NAME", "") or "").strip()


PASS_STAND_TO_BOT = (os.getenv("PASS_STAND_TO_BOT", "0") == "1")

ALREADY_FINISHED_EXIT_CODE = 42
BLOCK_MINUTES_ON_ALREADY_FINISHED = int(os.getenv("BLOCK_MINUTES_ON_ALREADY_FINISHED", "5"))



def _seen_key_project(entry: Dict[str, Any]) -> str:
    return str(entry.get("project_name") or "")


def load_seen(seen_file: str) -> List[Dict[str, Any]]:
    if not os.path.exists(seen_file):
        return []
    try:
        with open(seen_file, "r", encoding="utf-8") as f:
            seen = json.load(f)
        uniq: Dict[str, Dict[str, Any]] = {}
        for e in seen:
            pname = _seen_key_project(e)
            if not pname:
                continue
            uniq[pname] = {
                "project_name": pname,
                "task_id": e.get("task_id"),
                "org_id": e.get("org_id"),
            }
        return list(uniq.values())
    except json.JSONDecodeError:
        print(f"[manager] Warning: {seen_file} испорчен, сбрасываю список.")
        return []


def save_seen(seen_file: str, seen: List[Dict[str, Any]]) -> None:
    with open(seen_file, "w", encoding="utf-8") as f:
        json.dump(seen, f, ensure_ascii=False, indent=2)


def extract_project_name(task) -> Optional[str]:
    pname = getattr(task, "project_name", None)
    if pname and str(pname).strip():
        return str(pname).strip()
    return None


def extract_markup_task_id(task) -> Optional[str]:
    tid = getattr(task, "id", None)
    return str(tid) if tid else None


def chunked(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i:i + n] for i in range(0, len(lst), n)]


async def fetch_my_markup_tasks(client: TagmeClientAdvanced) -> List[Any]:
    all_tasks: List[Any] = []
    for page in range(1, PAGES + 1):
        tasks = await client.get_tasks_for_markup(page=page, size=PAGE_SIZE)
        if not tasks:
            break
        all_tasks.extend(tasks)
        if len(tasks) < PAGE_SIZE:
            break
    return all_tasks


def pick_one_task_per_project(tasks: List[Any]) -> Dict[str, str]:
    chosen: Dict[str, str] = {}
    for t in tasks:
        pname = extract_project_name(t)
        tid = extract_markup_task_id(t)
        if not pname or not tid:
            continue
        if pname not in chosen:
            chosen[pname] = tid
    return chosen


def is_check_project(pname: str) -> bool:
    return bool(pname) and "check" in pname.lower()


def has_valid_price(task) -> bool:
    price = getattr(task, "price", None)
    try:
        return price is not None and float(price) > 0
    except (TypeError, ValueError):
        return False


def filter_tasks_by_price(tasks: List[Any], stand_name: str = "") -> List[Any]:
    """Оставить задачи с ненулевой/заполненной ценой.

    check-проекты — исключение: берём всегда, даже с нулевой/пустой ценой.
    Пропускаемые по цене задачи логируются.
    """
    kept: List[Any] = []
    for t in tasks:
        pname = extract_project_name(t) or ""
        if has_valid_price(t) or is_check_project(pname):
            kept.append(t)
            continue
        print(
            f"[manager:{stand_name}] SKIP по цене: project='{pname}' "
            f"task_id={extract_markup_task_id(t)} price={getattr(t, 'price', None)}"
        )
    return kept


async def resolve_org_for_tasks(
    client: TagmeClientAdvanced,
    task_ids: List[str],
    org_ids: List[str],
    batch_size: int = 50,
) -> Dict[str, str]:
    unresolved = set(task_ids)
    resolved: Dict[str, str] = {}

    for org in org_ids:
        if not unresolved:
            break

        for batch in chunked(sorted(unresolved), batch_size):
            try:
                tasks = await client.get_tasks(
                    task_ids=batch,
                    organization_id=org,
                    page=1,
                    size=len(batch),
                )
            except Exception as e:
                print(f"[manager] resolve_org: get_tasks failed org={org}: {type(e).__name__}: {e}")
                continue

            for td in tasks or []:
                uid = getattr(td, "uid", None) or getattr(td, "id", None)
                if not uid:
                    continue
                uid = str(uid)
                if uid in unresolved:
                    resolved[uid] = org
                    unresolved.remove(uid)

    return resolved


def proc_key(project_name: str, task_id: str) -> str:
    return f"{project_name}::{task_id}"


async def run_manager(stand_name: str):
    stand = get_stand(stand_name)

    cfg_path = os.path.expanduser(stand.cfg_path)
    seen_file = stand.seen_file
    organizations = stand.organizations


    skip_names = set(getattr(stand, "skip_project_names", []) or [])

    def should_skip_project(pname: str) -> bool:
        if not pname:
            return False
        if SKIP_PROJECT_NAME and pname == SKIP_PROJECT_NAME:
            return True
        if pname in skip_names:
            return True
        return False


    @atexit.register
    def _cleanup_seen_file():
        try:
            os.remove(seen_file)
        except FileNotFoundError:
            pass

    seen = load_seen(seen_file)


    launched: Dict[str, subprocess.Popen] = {}

    client = TagmeClientAdvanced(cfg_path)


    run_cmd_base = [sys.executable, ANNOTATOR_SCRIPT]

    tick = 0
    while True:
        tick += 1


        for key, proc in list(launched.items()):
            if proc.poll() is not None:
                print(f"[manager:{stand.name}] Процесс '{key}' завершён (returncode={proc.returncode})")
                del launched[key]

                try:
                    pname, tid = key.split("::", 1)
                except Exception:
                    pname, tid = None, None

                if pname:
                    if proc.returncode == ALREADY_FINISHED_EXIT_CODE:
                        # другой воркер уже забрал/завершил этот assignment.
                        # не убираем task из seen, а временно блокируем её,
                        # чтобы менеджер не сразу перезапустил скрипт на ту же task.
                        blocked_until = time.time() + BLOCK_MINUTES_ON_ALREADY_FINISHED * 60
                        found = False
                        for e in seen:
                            if str(e.get("project_name") or "") == str(pname):
                                e["blocked_until"] = blocked_until
                                found = True
                        if found:
                            save_seen(seen_file, seen)
                            print(
                                f"[manager:{stand.name}] blocked project='{pname}' (task_id={tid}) "
                                f"на {BLOCK_MINUTES_ON_ALREADY_FINISHED} мин из-за already_finished"
                            )
                    else:
                        # если процесс завершился — убираем проект из seen.json
                        before = len(seen)
                        seen = [e for e in seen if str(e.get("project_name") or "") != str(pname)]
                        after = len(seen)
                        if after != before:
                            save_seen(seen_file, seen)
                            print(f"[manager:{stand.name}] removed from seen project='{pname}' (task_id={tid})")


        if (tick == 1) or (tick % SCAN_EVERY_SEC == 0):
            try:
                tasks = await fetch_my_markup_tasks(client)
                tasks = filter_tasks_by_price(tasks, stand.name)
                project_to_task = pick_one_task_per_project(tasks)

                if project_to_task:
                    print(f"[manager:{stand.name}] Scan: проектов с задачами = {len(project_to_task)}")
                else:
                    print(f"[manager:{stand.name}] Scan: задач нет")

                seen_by_name: Dict[str, Dict[str, Any]] = {
                    e["project_name"]: e for e in seen if e.get("project_name")
                }

                updated = False

                for pname, tid in project_to_task.items():
                    if should_skip_project(pname):
                        continue

                    if pname not in seen_by_name:
                        seen_by_name[pname] = {"project_name": pname, "task_id": tid, "org_id": None}
                        updated = True
                    else:
                        old_tid = str(seen_by_name[pname].get("task_id") or "")
                        new_tid = str(tid)
                        if old_tid != new_tid:
                            seen_by_name[pname]["task_id"] = new_tid
                            seen_by_name[pname]["org_id"] = None
                            seen_by_name[pname].pop("blocked_until", None)
                            updated = True
                            print(f"[manager:{stand.name}] task_id updated for project='{pname}': {old_tid} -> {new_tid}")

                            old_key = proc_key(pname, old_tid)
                            if old_key in launched:
                                print(f"[manager:{stand.name}] detach old process key (no kill): {old_key}")
                                launched.pop(old_key, None)

                seen = list(seen_by_name.values())

                need_resolve = [
                    e for e in seen
                    if (not e.get("org_id")) and e.get("task_id") and (not should_skip_project(str(e.get("project_name") or "")))
                ]
                if need_resolve:
                    task_ids = [str(e["task_id"]) for e in need_resolve]
                    resolved_map = await resolve_org_for_tasks(client, task_ids, organizations, batch_size=50)

                    resolved_cnt = 0
                    for e in need_resolve:
                        tid = str(e["task_id"])
                        org = resolved_map.get(tid)
                        if org:
                            e["org_id"] = org
                            resolved_cnt += 1

                    if resolved_cnt:
                        updated = True
                        print(f"[manager:{stand.name}] Resolved org_id: {resolved_cnt}/{len(need_resolve)}")

                if updated:
                    save_seen(seen_file, seen)

            except Exception as e:
                print(f"[manager:{stand.name}] Scan error: {type(e).__name__}: {e}")

        running_count = len(launched)

        for entry in seen:
            if running_count >= MAX_PROJECT_PROCESSES:
                break

            pname = entry.get("project_name")
            tid = entry.get("task_id")
            org_id = entry.get("org_id")

            if not pname or not tid or not org_id:
                continue


            if should_skip_project(str(pname)):
                continue

            blocked_until = entry.get("blocked_until")
            if blocked_until and time.time() < float(blocked_until):
                continue

            key = proc_key(pname, str(tid))
            if key in launched:
                continue

            cmd = run_cmd_base + ["-t", str(tid), "-o", str(org_id), "--stand", stand.name]
            if PASS_STAND_TO_BOT:
                cmd += ["--stand", stand.name]

            print(f"[manager:{stand.name}] RUN: {key}")
            print("  CMD:", " ".join(cmd))

            launched[key] = subprocess.Popen(cmd)
            running_count += 1
            await asyncio.sleep(0.5)

        await asyncio.sleep(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stand", required=True, choices=["cloud", "sigma", "sberdev"])
    args = parser.parse_args()
    asyncio.run(run_manager(args.stand))


if __name__ == "__main__":
    main()
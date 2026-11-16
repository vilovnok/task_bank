import argparse
import asyncio
import json
import logging
import os
import traceback
from copy import deepcopy
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from crowd_sdk.tagme import TagmeClientAdvanced

from utils.utils import (
    get_task_update_date,
    get_task_id,
    get_task_state_str,
    is_task_active,
    detect_task_kind,
    should_skip_task,
)
from utils.excel_builder import ExcelReportBuilder
from utils.fir_uploader import upload_report_to_fir


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
log = logging.getLogger("qa_manager")


BASE_DIR = Path(__file__).resolve().parent

CONFIG_PATH = BASE_DIR / "configs" / "projects_config.json"
STATE_PATH = BASE_DIR / "state" / "tracking_state.json"

DONE_STATE_KEYWORDS = ("DONE", "FINISHED", "COMPLETED")


def now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def today_str() -> str:
    return date.today().isoformat()


def expand_path(path: str) -> str:
    return os.path.expanduser(path)


def resolve_project_path(path: str) -> str:
    p = Path(os.path.expanduser(path))

    if p.is_absolute():
        return str(p)

    return str(BASE_DIR / p)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return deepcopy(default)

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path: Path, data: Any) -> None:
    """
    Атомарная запись JSON с поддержкой pandas типов.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(path.suffix + ".tmp")

    class DateTimeEncoder(json.JSONEncoder):
        """Кастомный encoder для pandas и datetime типов."""
        def default(self, obj):
            # pandas Timestamp
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            # datetime/date
            if isinstance(obj, (datetime, date)):
                return obj.isoformat()
            # pandas NA/NaT
            if pd.isna(obj):
                return None
            return super().default(obj)

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, cls=DateTimeEncoder)

    tmp_path.replace(path)


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def dt_to_iso(value: Any) -> Optional[str]:
    if value is None:
        return None

    if hasattr(value, "isoformat"):
        return value.isoformat()

    return str(value)


def dt_to_date(value: Any) -> Optional[date]:
    if value is None:
        return None

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
        except Exception:
            return None

    return None


def get_stand_cfg(config: Dict[str, Any], stand_name: str) -> Dict[str, Any]:
    stands = config.get("stands", {})

    if stand_name not in stands:
        raise KeyError(f"Unknown stand_name={stand_name}. Available: {list(stands.keys())}")

    stand_cfg = dict(stands[stand_name])
    stand_cfg["cfg_path"] = expand_path(stand_cfg["cfg_path"])

    return stand_cfg


def validate_project_cfg(project_cfg: Dict[str, Any]) -> None:
    required = [
        "project_id",
        "project_name",
        "stand_name",
        "organization_id",
        "tracking_start_date",
        "paths",
    ]

    for key in required:
        if key not in project_cfg:
            raise ValueError(f"Project config missing required key: {key}")

    paths = project_cfg["paths"]

    required_paths = [
        "quality_script",
        "short_rules_file",
    ]

    # Опциональные пути (имеют fallback на хардкоженные значения)
    optional_paths = [
        "comments_script",
        "summary_script",
        "comment_prompt_file",
        "summary_prompt_system_file",
        "summary_prompt_user_file",
        "tagme_config",
    ]

    for key in required_paths:
        if key not in paths:
            raise ValueError(f"Project paths missing required key: {key}")

    # Логируем опциональные пути если они не указаны
    for key in optional_paths:
        if key not in paths:
            log.debug(f"Optional path '{key}' not specified in config, will use default if available")


async def create_clients_by_stand(config: Dict[str, Any]) -> Dict[str, TagmeClientAdvanced]:
    used_stands = sorted({
        p["stand_name"]
        for p in config.get("projects", [])
        if p.get("enabled", True)
    })

    clients = {}

    for stand_name in used_stands:
        stand_cfg = get_stand_cfg(config, stand_name)
        cfg_path = stand_cfg["cfg_path"]

        log.info("Create TagMe client | stand=%s | cfg_path=%s", stand_name, cfg_path)

        clients[stand_name] = TagmeClientAdvanced(cfg_path)

    return clients


async def close_clients(clients: Dict[str, TagmeClientAdvanced]) -> None:
    for stand_name, client in clients.items():
        try:
            await client.close()
            log.info("Client closed | stand=%s", stand_name)
        except Exception:
            log.exception("Failed to close client | stand=%s", stand_name)


def is_done_state(state: str) -> bool:
    state_upper = str(state or "").upper()
    return any(keyword in state_upper for keyword in DONE_STATE_KEYWORDS)


def normalize_tagme_task(task: Any) -> Dict[str, Any]:
    task_id = get_task_id(task)
    task_name = str(getattr(task, "name", "") or "").strip()

    if isinstance(task, dict):
        task_name = str(task.get("task_name") or task.get("name") or "").strip()

    state = get_task_state_str(task)
    update_dt = get_task_update_date(task)
    kind = detect_task_kind(task)

    active = is_task_active(state)
    done = is_done_state(state)

    update_date_only = None
    update_date_obj = dt_to_date(update_dt)
    if update_date_obj:
        update_date_only = update_date_obj.isoformat()

    return {
        "task_id": task_id,
        "task_name": task_name,
        "state": state,
        "task_kind": kind,
        "update_date": dt_to_iso(update_dt),
        "update_date_only": update_date_only,
        "is_active": active,
        "is_done": done,
        "raw_skip_by_utils": bool(should_skip_task(task)),
    }


async def load_project_tasks_from_tagme(
    client: TagmeClientAdvanced,
    project_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    project_id = project_cfg["project_id"]
    organization_id = project_cfg["organization_id"]

    log.info(
        "Load tasks from TagMe | project=%s | project_id=%s | org_id=%s",
        project_cfg["project_name"],
        project_id,
        organization_id,
    )

    tasks = await client.get_project_tasks(
        project_id=project_id,
        organization_id=organization_id,
    )

    normalized = []

    for task in tasks or []:
        try:
            item = normalize_tagme_task(task)
            normalized.append(item)
        except Exception:
            log.warning(
                "Failed to normalize task | project=%s\n%s",
                project_cfg.get("project_name"),
                traceback.format_exc(),
            )

    return normalized


def is_task_regular(project_cfg: Dict[str, Any], task: Dict[str, Any]) -> bool:
    task_selection = project_cfg.get("task_selection", {})
    skip_kinds = {
        str(x).strip().lower()
        for x in task_selection.get("skip_task_kinds", [])
    }

    task_kind = str(task.get("task_kind") or "").strip().lower()

    if task_kind in skip_kinds:
        return False

    if task.get("raw_skip_by_utils"):
        return False

    return True


def task_is_after_tracking_start(project_cfg: Dict[str, Any], task: Dict[str, Any]) -> bool:
    start_date = parse_date(project_cfg["tracking_start_date"])

    update_date_only = task.get("update_date_only")
    if not update_date_only:
        return False

    task_date = parse_date(update_date_only)

    return task_date >= start_date


def should_take_task_by_state(project_cfg: Dict[str, Any], task: Dict[str, Any]) -> bool:
    task_selection = project_cfg.get("task_selection", {})

    include_active = bool(task_selection.get("include_active_tasks", True))
    include_done = bool(task_selection.get("include_done_tasks", True))

    if task.get("is_active") and include_active:
        return True

    if task.get("is_done") and include_done:
        return True

    return False


def select_tasks_for_project(
    project_cfg: Dict[str, Any],
    tasks: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    selected = []
    skipped = []

    for task in tasks:
        reason = None

        if not task.get("task_id"):
            reason = "empty_task_id"

        elif not is_task_regular(project_cfg, task):
            reason = f"skip_task_kind_or_utils_skip: {task.get('task_kind')}"

        elif not task_is_after_tracking_start(project_cfg, task):
            reason = f"before_tracking_start: update_date={task.get('update_date_only')}"

        elif not should_take_task_by_state(project_cfg, task):
            reason = f"state_not_selected: state={task.get('state')}"

        if reason:
            skipped.append({
                "task_id": task.get("task_id"),
                "task_name": task.get("task_name"),
                "state": task.get("state"),
                "task_kind": task.get("task_kind"),
                "update_date": task.get("update_date"),
                "reason": reason,
            })
            continue

        selected.append(task)

    return selected, skipped


def ensure_project_state(state: Dict[str, Any], project_cfg: Dict[str, Any]) -> Dict[str, Any]:
    project_id = project_cfg["project_id"]

    state.setdefault("projects", {})

    if project_id not in state["projects"]:
        state["projects"][project_id] = {
            "project_id": project_id,
            "project_name": project_cfg.get("project_name"),
            "stand_name": project_cfg.get("stand_name"),
            "organization_id": project_cfg.get("organization_id"),
            "tracking_start_date": project_cfg.get("tracking_start_date"),
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "daily_runs": {},
            "tasks": {},
        }

    project_state = state["projects"][project_id]

    project_state["project_name"] = project_cfg.get("project_name")
    project_state["stand_name"] = project_cfg.get("stand_name")
    project_state["organization_id"] = project_cfg.get("organization_id")
    project_state["tracking_start_date"] = project_cfg.get("tracking_start_date")
    project_state["updated_at"] = now_iso()

    project_state.setdefault("daily_runs", {})
    project_state.setdefault("tasks", {})

    return project_state


def ensure_task_state(project_state: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
    task_id = task["task_id"]

    if task_id not in project_state["tasks"]:
        project_state["tasks"][task_id] = {
            "task_id": task_id,
            "task_name": task.get("task_name"),
            "first_seen_at": now_iso(),
            "last_seen_at": now_iso(),
            "last_known_state": task.get("state"),
            "last_known_kind": task.get("task_kind"),
            "last_update_date": task.get("update_date"),
            "fully_analyzed": False,
            "slices": {},
            "pipeline_runs": [],
        }

    task_state = project_state["tasks"][task_id]

    task_state["task_name"] = task.get("task_name")
    task_state["last_seen_at"] = now_iso()
    task_state["last_known_state"] = task.get("state")
    task_state["last_known_kind"] = task.get("task_kind")
    task_state["last_update_date"] = task.get("update_date")

    task_state.setdefault("slices", {})
    task_state.setdefault("pipeline_runs", [])

    return task_state


def project_has_successful_daily_run_today(project_state: Dict[str, Any], run_date: str) -> bool:
    daily_run = project_state.get("daily_runs", {}).get(run_date)

    if not daily_run:
        return False

    return daily_run.get("status") == "completed"


def task_slice_completed_today(task_state: Dict[str, Any], run_date: str) -> bool:
    slice_info = task_state.get("slices", {}).get(run_date)

    if not slice_info:
        return False

    return slice_info.get("status") == "completed"


def should_process_task_today(task: Dict[str, Any], task_state: Dict[str, Any], run_date: str) -> bool:
    if task_slice_completed_today(task_state, run_date):
        return False

    if task_state.get("fully_analyzed") and task.get("is_done"):
        return False

    return True


async def run_quality_stage(
    client: TagmeClientAdvanced,
    project_cfg: Dict[str, Any],
    task: Dict[str, Any],
    run_date: str,
    excel_builder: ExcelReportBuilder
) -> Dict[str, Any]:
    """
    Запускает stage для расчета качества разметки.

    Args:
        client: TagMe клиент
        project_cfg: Конфигурация проекта
        task: Данные задачи
        run_date: Дата запуска
        excel_builder: Builder для единого Excel отчета

    Returns:
        Результат выполнения stage
    """
    started_at = now_iso()

    try:
        # Динамический импорт скрипта по имени из конфига
        quality_script_path = project_cfg["paths"]["quality_script"]

        # Извлекаем имя модуля (без .py и директории)
        script_name = Path(quality_script_path).stem

        # Импортируем модуль
        import importlib
        quality_module = importlib.import_module(f"scripts.{script_name}")

        # Определяем директорию для данных
        data_dir = BASE_DIR / "data" / project_cfg["project_id"]

        log.info(
            "RUN quality stage | project=%s | task=%s | task_id=%s | script=%s",
            project_cfg["project_name"],
            task["task_name"],
            task["task_id"],
            script_name,
        )

        # Запускаем расчет качества
        result = await quality_module.run(
            client=client,
            task_id=task["task_id"],
            task_name=task["task_name"],
            run_date=run_date,
            data_dir=data_dir,
            organization_id=project_cfg["organization_id"],
            project_id=project_cfg["project_id"],
            state_dir=BASE_DIR / "state",
            excel_builder=excel_builder,  # Передаем builder
        )

        # Проверяем статус выполнения
        if result.get("status") == "success":
            log.info(
                "Quality stage COMPLETED | project=%s | task=%s | assignments=%s | problems=%s",
                project_cfg["project_name"],
                task["task_name"],
                result.get("assignments_count", 0),
                result.get("bad_assignments_count", 0),
            )

            return {
                "stage": "quality",
                "status": "completed",
                "started_at": started_at,
                "finished_at": now_iso(),
                "result": result,
            }
        else:
            log.warning(
                "Quality stage FAILED | project=%s | task=%s | error=%s",
                project_cfg["project_name"],
                task["task_name"],
                result.get("error", "Unknown error"),
            )

            return {
                "stage": "quality",
                "status": "failed",
                "started_at": started_at,
                "finished_at": now_iso(),
                "error": result.get("error", "Unknown error"),
                "result": result,
            }

    except Exception as e:
        log.error(
            "Quality stage ERROR | project=%s | task=%s | exception=%s",
            project_cfg["project_name"],
            task["task_name"],
            str(e),
            exc_info=True,
        )

        return {
            "stage": "quality",
            "status": "error",
            "started_at": started_at,
            "finished_at": now_iso(),
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


async def run_comments_stage(
    project_cfg: Dict[str, Any],
    task: Dict[str, Any],
    run_date: str,
    quality_result: Dict[str, Any],
    excel_builder: ExcelReportBuilder
) -> Dict[str, Any]:
    """
    Запускает stage для генерации комментариев к плохим assignments.

    Args:
        project_cfg: Конфигурация проекта
        task: Данные задачи
        run_date: Дата запуска
        quality_result: Результат stage "quality"
        excel_builder: Builder для единого Excel отчета

    Returns:
        Результат выполнения stage
    """
    started_at = now_iso()

    try:
        # Импортируем скрипт генерации комментариев
        from scripts import generate_comments

        # Определяем пути
        data_dir = BASE_DIR / "data" / project_cfg["project_id"]
        rules_file = resolve_project_path(project_cfg["paths"]["short_rules_file"])

        # Путь к конфигу TagMe для LLM прокси
        proxy_cfg_path = project_cfg.get("paths", {}).get("tagme_config")
        if proxy_cfg_path:
            proxy_cfg_path = str(Path(proxy_cfg_path).expanduser())

        # Путь к промпту для генерации комментариев (опционально)
        comment_prompt_file = project_cfg.get("paths", {}).get("comment_prompt_file")
        if comment_prompt_file:
            comment_prompt_file = resolve_project_path(comment_prompt_file)

        log.info(
            "RUN comments stage | project=%s | task=%s | task_id=%s",
            project_cfg["project_name"],
            task["task_name"],
            task["task_id"],
        )

        # Запускаем генерацию комментариев
        result = await generate_comments.run(
            task_id=task["task_id"],
            task_name=task["task_name"],
            run_date=run_date,
            data_dir=data_dir,
            quality_result=quality_result,
            rules_file=rules_file,
            proxy_cfg_path=proxy_cfg_path,
            comment_prompt_file=comment_prompt_file,  # Путь к промпту
            excel_builder=excel_builder,  # Передаем builder
        )

        # Проверяем статус выполнения
        if result.get("status") == "success":
            log.info(
                "Comments stage COMPLETED | project=%s | task=%s | comments=%s",
                project_cfg["project_name"],
                task["task_name"],
                result.get("comments_generated", 0),
            )
        elif result.get("status") == "skipped":
            log.info(
                "Comments stage SKIPPED | project=%s | task=%s | reason=%s",
                project_cfg["project_name"],
                task["task_name"],
                result.get("reason", "unknown"),
            )
        else:
            log.warning(
                "Comments stage FAILED | project=%s | task=%s | error=%s",
                project_cfg["project_name"],
                task["task_name"],
                result.get("error", "Unknown error"),
            )

        return {
            "stage": "comments",
            "status": result.get("status", "error"),
            "started_at": started_at,
            "finished_at": now_iso(),
            "result": result,
        }

    except Exception as e:
        log.error(
            "Comments stage ERROR | project=%s | task=%s | exception=%s",
            project_cfg["project_name"],
            task["task_name"],
            str(e),
            exc_info=True,
        )

        return {
            "stage": "comments",
            "status": "error",
            "started_at": started_at,
            "finished_at": now_iso(),
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


async def run_summary_stage(
    project_cfg: Dict[str, Any],
    task: Dict[str, Any],
    run_date: str,
    comments_result: Dict[str, Any],
    quality_result: Dict[str, Any],
    excel_builder: ExcelReportBuilder
) -> Dict[str, Any]:
    """
    Запускает stage для генерации персональных сводок по разметчикам.

    Args:
        project_cfg: Конфигурация проекта
        task: Данные задачи
        run_date: Дата запуска
        comments_result: Результат stage "comments"
        quality_result: Результат stage "quality" (для доступа к processed_assignment_ids)
        excel_builder: Builder для единого Excel отчета

    Returns:
        Результат выполнения stage
    """
    started_at = now_iso()

    try:
        # Импортируем скрипт генерации сводок
        from scripts import generate_summary

        # Определяем пути
        data_dir = BASE_DIR / "data" / project_cfg["project_id"]
        rules_file = resolve_project_path(project_cfg["paths"]["short_rules_file"])

        # Путь к конфигу TagMe для LLM прокси
        proxy_cfg_path = project_cfg.get("paths", {}).get("tagme_config")
        if proxy_cfg_path:
            proxy_cfg_path = str(Path(proxy_cfg_path).expanduser())

        # Пути к промптам для генерации сумм (опционально)
        summary_prompt_system_file = project_cfg.get("paths", {}).get("summary_prompt_system_file")
        if summary_prompt_system_file:
            summary_prompt_system_file = resolve_project_path(summary_prompt_system_file)

        summary_prompt_user_file = project_cfg.get("paths", {}).get("summary_prompt_user_file")
        if summary_prompt_user_file:
            summary_prompt_user_file = resolve_project_path(summary_prompt_user_file)

        log.info(
            "RUN summary stage | project=%s | task=%s | task_id=%s",
            project_cfg["project_name"],
            task["task_name"],
            task["task_id"],
        )

        # Запускаем генерацию сводок
        result = await generate_summary.run(
            task_id=task["task_id"],
            task_name=task["task_name"],
            run_date=run_date,
            data_dir=data_dir,
            comments_result=comments_result,
            rules_file=rules_file,
            proxy_cfg_path=proxy_cfg_path,
            summary_prompt_system_file=summary_prompt_system_file,  # Путь к system промпту
            summary_prompt_user_file=summary_prompt_user_file,  # Путь к user промпту
            project_id=project_cfg["project_id"],
            state_dir=BASE_DIR / "state",
            quality_result=quality_result,  # Передаем для доступа к processed_assignment_ids
            excel_builder=excel_builder,  # Передаем builder
        )

        # Проверяем статус выполнения
        if result.get("status") == "success":
            log.info(
                "Summary stage COMPLETED | project=%s | task=%s | summaries=%s | report=%s",
                project_cfg["project_name"],
                task["task_name"],
                result.get("markers_with_summaries", 0),
                result.get("excel_report_path", "N/A"),
            )
        elif result.get("status") == "skipped":
            log.info(
                "Summary stage SKIPPED | project=%s | task=%s | reason=%s",
                project_cfg["project_name"],
                task["task_name"],
                result.get("reason", "unknown"),
            )
        else:
            log.warning(
                "Summary stage FAILED | project=%s | task=%s | error=%s",
                project_cfg["project_name"],
                task["task_name"],
                result.get("error", "Unknown error"),
            )

        return {
            "stage": "summary",
            "status": result.get("status", "error"),
            "started_at": started_at,
            "finished_at": now_iso(),
            "result": result,
        }

    except Exception as e:
        log.error(
            "Summary stage ERROR | project=%s | task=%s | exception=%s",
            project_cfg["project_name"],
            task["task_name"],
            str(e),
            exc_info=True,
        )

        return {
            "stage": "summary",
            "status": "error",
            "started_at": started_at,
            "finished_at": now_iso(),
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


async def run_task_pipeline(
    client: TagmeClientAdvanced,
    project_cfg: Dict[str, Any],
    task: Dict[str, Any],
    run_date: str
) -> Dict[str, Any]:
    """
    Запускает полный pipeline обработки задачи.

    Args:
        client: TagMe клиент
        project_cfg: Конфигурация проекта
        task: Данные задачи
        run_date: Дата запуска

    Returns:
        Результат выполнения pipeline
    """
    started_at = now_iso()

    # Создаем единый Excel отчет для этой задачи
    data_dir = BASE_DIR / "data" / project_cfg["project_id"]
    output_dir = data_dir / "reports" / run_date
    output_dir.mkdir(parents=True, exist_ok=True)

    # Формируем имя файла: qa_report_<task_id_короткий>_<timestamp>.xlsx
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"qa_report_{task['task_id'][:8]}_{timestamp}.xlsx"
    report_path = output_dir / report_filename

    log.info(f"Создаем единый отчет: {report_path}")
    excel_builder = ExcelReportBuilder(report_path)

    # Запускаем stages последовательно, передавая excel_builder
    quality_stage = await run_quality_stage(client, project_cfg, task, run_date, excel_builder)

    # Передаем результат quality в comments
    quality_result = quality_stage.get("result", {})
    comments_stage = await run_comments_stage(project_cfg, task, run_date, quality_result, excel_builder)

    # Передаем результаты comments и quality в summary
    comments_result = comments_stage.get("result", {})
    summary_stage = await run_summary_stage(
        project_cfg, task, run_date, comments_result, quality_result, excel_builder
    )

    stages = [quality_stage, comments_stage, summary_stage]

    status = "completed"
    for stage in stages:
        stage_status = stage.get("status")
        # "skipped" не считается ошибкой
        if stage_status not in ["completed", "success", "skipped"]:
            status = "failed"
            break

    log.info(f"Pipeline завершен. Итоговый отчет: {report_path}")

    # Загружаем отчёт на FIR, если настроено
    fir_uploaded = False
    fir_config = project_cfg.get("fir_upload", {})
    if fir_config.get("enabled", False) and status == "completed":
        fir_base_path = fir_config.get("base_path")
        if fir_base_path:
            log.info(f"Загружаем отчёт на FIR: {fir_base_path}")
            try:
                fir_uploaded = upload_report_to_fir(
                    report_path=report_path,
                    fir_base_path=fir_base_path,
                    run_date=run_date
                )
                if fir_uploaded:
                    log.info("✓ Отчёт успешно загружен на FIR")
                else:
                    log.warning("✗ Не удалось загрузить отчёт на FIR")
            except Exception as e:
                log.error(f"Ошибка при загрузке на FIR: {e}", exc_info=True)
        else:
            log.warning("FIR upload включен, но не указан base_path")

    return {
        "run_date": run_date,
        "status": status,
        "started_at": started_at,
        "finished_at": now_iso(),
        "project_id": project_cfg["project_id"],
        "project_name": project_cfg["project_name"],
        "stand_name": project_cfg["stand_name"],
        "organization_id": project_cfg["organization_id"],
        "task_id": task["task_id"],
        "task_name": task.get("task_name"),
        "task_state": task.get("state"),
        "task_kind": task.get("task_kind"),
        "task_update_date": task.get("update_date"),
        "feedback_scope": project_cfg.get("feedback_scope", "problem_only"),
        "excel_report_path": str(report_path),  # Добавляем путь к отчету
        "fir_uploaded": fir_uploaded,  # Флаг успешной загрузки на FIR
        "stages": stages,
    }


async def process_project(
    client: TagmeClientAdvanced,
    project_cfg: Dict[str, Any],
    state: Dict[str, Any],
    run_date: str,
) -> None:
    validate_project_cfg(project_cfg)

    project_name = project_cfg["project_name"]
    project_id = project_cfg["project_id"]

    if not project_cfg.get("enabled", True):
        log.info("SKIP disabled project: %s", project_name)
        return

    if not project_cfg.get("daily_feedback_enabled", False):
        log.info("SKIP daily feedback disabled: %s", project_name)
        return

    project_state = ensure_project_state(state, project_cfg)

    if project_has_successful_daily_run_today(project_state, run_date):
        log.info("SKIP project already completed today: %s | date=%s", project_name, run_date)
        return

    log.info("=" * 80)
    log.info("PROCESS PROJECT: %s | id=%s | date=%s", project_name, project_id, run_date)

    daily_run = {
        "run_date": run_date,
        "status": "running",
        "started_at": now_iso(),
        "finished_at": None,
        "all_tasks_count": 0,
        "selected_tasks_count": 0,
        "processed_tasks_count": 0,
        "skipped_tasks_count": 0,
        "selected_task_ids": [],
        "processed_task_ids": [],
        "skipped_tasks": [],
        "errors": [],
    }

    project_state["daily_runs"][run_date] = daily_run

    try:
        tasks = await load_project_tasks_from_tagme(client, project_cfg)

        daily_run["all_tasks_count"] = len(tasks)

        selected_tasks, skipped_tasks = select_tasks_for_project(project_cfg, tasks)

        daily_run["selected_tasks_count"] = len(selected_tasks)
        daily_run["selected_task_ids"] = [t["task_id"] for t in selected_tasks]
        daily_run["skipped_tasks_count"] = len(skipped_tasks)
        daily_run["skipped_tasks"] = skipped_tasks

        log.info(
            "TASKS: project=%s | all=%d | selected=%d | skipped=%d",
            project_name,
            len(tasks),
            len(selected_tasks),
            len(skipped_tasks),
        )

        if not selected_tasks:
            daily_run["status"] = "completed"
            daily_run["finished_at"] = now_iso()
            return

        for task in selected_tasks:
            task_id = task["task_id"]
            task_name = task.get("task_name")

            task_state = ensure_task_state(project_state, task)

            if not should_process_task_today(task, task_state, run_date):
                log.info("SKIP already processed task today/full: %s | %s", task_id, task_name)
                continue

            log.info(
                "PROCESS TASK: %s | %s | state=%s | kind=%s | update=%s",
                task_id,
                task_name,
                task.get("state"),
                task.get("task_kind"),
                task.get("update_date"),
            )

            task_state["slices"][run_date] = {
                "run_date": run_date,
                "status": "running",
                "started_at": now_iso(),
                "finished_at": None,
                "task_state_at_run": task.get("state"),
                "task_kind_at_run": task.get("task_kind"),
                "task_update_date_at_run": task.get("update_date"),
            }

            try:
                pipeline_run = await run_task_pipeline(client, project_cfg, task, run_date)

                task_state["pipeline_runs"].append(pipeline_run)

                task_state["slices"][run_date].update({
                    "status": pipeline_run["status"],
                    "finished_at": now_iso(),
                    "pipeline_status": pipeline_run["status"],
                })

                if pipeline_run["status"] == "completed":
                    daily_run["processed_tasks_count"] += 1
                    daily_run["processed_task_ids"].append(task_id)

                    if task.get("is_done"):
                        task_state["fully_analyzed"] = True
                else:
                    daily_run["errors"].append({
                        "task_id": task_id,
                        "task_name": task_name,
                        "error": "pipeline failed",
                    })

            except Exception as e:
                log.exception("Task pipeline failed: %s | %s", task_id, task_name)

                task_state["slices"][run_date].update({
                    "status": "failed",
                    "finished_at": now_iso(),
                    "error": str(e),
                })

                daily_run["errors"].append({
                    "task_id": task_id,
                    "task_name": task_name,
                    "error": str(e),
                })

        if daily_run["errors"]:
            daily_run["status"] = "failed"
        else:
            daily_run["status"] = "completed"

        daily_run["finished_at"] = now_iso()

    except Exception as e:
        log.exception("Project processing failed: %s", project_name)

        daily_run["status"] = "failed"
        daily_run["finished_at"] = now_iso()
        daily_run["errors"].append({
            "project_id": project_id,
            "project_name": project_name,
            "error": str(e),
        })

    log.info(
        "PROJECT DONE: %s | status=%s | all=%d | selected=%d | processed=%d | skipped=%d",
        project_name,
        daily_run["status"],
        daily_run["all_tasks_count"],
        daily_run["selected_tasks_count"],
        daily_run["processed_tasks_count"],
        daily_run["skipped_tasks_count"],
    )


async def run_manager_once(config_path: Path, state_path: Path) -> None:
    config = load_json(config_path, default={})

    state = load_json(
        state_path,
        default={
            "version": 1,
            "updated_at": None,
            "projects": {},
        },
    )

    state.setdefault("version", 1)
    state.setdefault("projects", {})

    run_date = today_str()

    projects = config.get("projects", [])

    log.info("Manager iteration started | projects=%d | date=%s", len(projects), run_date)

    clients = await create_clients_by_stand(config)

    try:
        for project_cfg in projects:
            if not project_cfg.get("enabled", True):
                log.info("SKIP disabled project: %s", project_cfg.get("project_name"))
                continue

            stand_name = project_cfg["stand_name"]
            client = clients[stand_name]

            await process_project(client, project_cfg, state, run_date)

            state["updated_at"] = now_iso()
            atomic_write_json(state_path, state)

    finally:
        await close_clients(clients)

    state["updated_at"] = now_iso()
    atomic_write_json(state_path, state)

    log.info("Manager iteration finished | state saved to %s", state_path)


async def run_manager_loop(config_path: Path, state_path: Path, interval_seconds: int) -> None:
    log.info("Manager loop started | interval=%s sec", interval_seconds)

    while True:
        try:
            await run_manager_once(config_path, state_path)
        except Exception:
            log.exception("Manager iteration failed")

        await asyncio.sleep(interval_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--state", default=str(STATE_PATH))
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval", type=int, default=None)

    args = parser.parse_args()

    config_path = Path(args.config)
    state_path = Path(args.state)

    config = load_json(config_path, default={})
    manager_cfg = config.get("manager", {})

    interval_seconds = args.interval
    if interval_seconds is None:
        interval_seconds = int(manager_cfg.get("check_interval_seconds", 300))

    run_once_by_default = bool(manager_cfg.get("run_once_by_default", True))

    if args.loop:
        asyncio.run(run_manager_loop(config_path, state_path, interval_seconds))
        return

    if args.once or run_once_by_default:
        asyncio.run(run_manager_once(config_path, state_path))
        return

    asyncio.run(run_manager_loop(config_path, state_path, interval_seconds))


if __name__ == "__main__":
    main()
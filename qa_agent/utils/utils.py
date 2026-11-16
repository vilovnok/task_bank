from datetime import datetime
from typing import Any, Optional


def get_task_id(task: Any) -> str:
    for attr in ["task_id", "id", "uid", "uuid"]:
        value = getattr(task, attr, None)
        if value:
            return str(value)

    if isinstance(task, dict):
        for key in ["task_id", "id", "uid", "uuid"]:
            value = task.get(key)
            if value:
                return str(value)

    return ""


def get_task_state_str(task: Any) -> str:
    for attr in ["state", "status"]:
        value = getattr(task, attr, None)
        if value:
            return str(value)

    if isinstance(task, dict):
        for key in ["state", "status"]:
            value = task.get(key)
            if value:
                return str(value)

    return ""


def get_task_update_date(task: Any) -> Optional[datetime]:
    possible_attrs = [
        "updated_at",
        "update_date",
        "updated",
        "modified_at",
        "last_update_date",
        "last_modified",
    ]

    for attr in possible_attrs:
        value = getattr(task, attr, None)
        dt = _to_datetime(value)
        if dt:
            return dt

    if isinstance(task, dict):
        for key in possible_attrs:
            value = task.get(key)
            dt = _to_datetime(value)
            if dt:
                return dt

    return None


def _to_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None

    if isinstance(value, datetime):
        return value

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None

        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            return None

    return None


def is_task_active(state: str) -> bool:
    state_upper = str(state or "").upper()

    active_markers = [
        "ACTIVE",
        "RUNNING",
        "IN_PROGRESS",
        "PROCESSING",
        "STARTED",
    ]

    return any(marker in state_upper for marker in active_markers)


def detect_task_kind(task: Any) -> str:
    """
    Пытаемся определить тип задачи:
    - regular
    - training
    - exam
    - education
    - qualification

    Сейчас эвристика простая: смотрим имя/тип задачи.
    Потом можно будет уточнить под реальные поля TagMe.
    """
    name = str(getattr(task, "name", "") or "").lower()

    if isinstance(task, dict):
        name = str(task.get("task_name") or task.get("name") or "").lower()
        raw_kind = str(task.get("kind") or task.get("type") or "").lower()
    else:
        raw_kind = str(getattr(task, "kind", "") or getattr(task, "type", "") or "").lower()

    text = f"{name} {raw_kind}"

    if "exam" in text or "экзам" in text:
        return "exam"

    if "training" in text or "обуч" in text:
        return "training"

    if "education" in text:
        return "education"

    if "qualification" in text or "квалификац" in text:
        return "qualification"

    return "regular"


def should_skip_task(task: Any) -> bool:
    kind = detect_task_kind(task)
    return kind in {"training", "exam", "education", "qualification"}
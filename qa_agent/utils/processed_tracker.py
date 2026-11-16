"""
Утилита для отслеживания обработанных assignments.

Предотвращает повторную обработку одних и тех же assignments.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set

log = logging.getLogger(__name__)


class ProcessedAssignmentsTracker:
    """Отслеживание обработанных assignments по проектам."""

    def __init__(self, state_dir: Path):
        """
        Инициализация трекера.

        Args:
            state_dir: Директория для хранения состояния
        """
        self.state_dir = state_dir / "processed_assignments"
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _get_project_file(self, project_id: str) -> Path:
        """
        Получает путь к файлу проекта.

        Args:
            project_id: ID проекта

        Returns:
            Путь к файлу состояния
        """
        return self.state_dir / f"{project_id}.json"

    def load_processed_ids(self, project_id: str, task_id: str) -> Set[str]:
        """
        Загружает список обработанных assignment_id для задачи.

        Args:
            project_id: ID проекта
            task_id: ID задачи

        Returns:
            Множество обработанных assignment_id
        """
        file_path = self._get_project_file(project_id)

        if not file_path.exists():
            log.info(f"Файл отслеживания не найден: {file_path}, начинаем с чистого листа")
            return set()

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            task_data = data.get("tasks", {}).get(task_id, {})
            processed_list = task_data.get("processed_assignments", [])

            log.info(f"Загружено {len(processed_list)} обработанных assignment_id для task={task_id}")
            return set(processed_list)

        except Exception as e:
            log.error(f"Ошибка при загрузке processed_assignments: {e}")
            return set()

    def save_processed_ids(
        self,
        project_id: str,
        task_id: str,
        task_name: str,
        new_assignment_ids: List[str]
    ) -> None:
        """
        Сохраняет новые обработанные assignment_id.

        Args:
            project_id: ID проекта
            task_id: ID задачи
            task_name: Название задачи
            new_assignment_ids: Список новых обработанных assignment_id
        """
        file_path = self._get_project_file(project_id)

        # Загружаем существующие данные
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                log.warning(f"Не удалось загрузить существующий файл: {e}, создаем новый")
                data = {}
        else:
            data = {}

        # Инициализируем структуру если нужно
        if "project_id" not in data:
            data["project_id"] = project_id

        if "tasks" not in data:
            data["tasks"] = {}

        if task_id not in data["tasks"]:
            data["tasks"][task_id] = {
                "task_name": task_name,
                "processed_assignments": [],
                "total_processed": 0
            }

        # Объединяем старые и новые ID (убираем дубликаты)
        existing_ids = set(data["tasks"][task_id].get("processed_assignments", []))
        all_ids = existing_ids | set(new_assignment_ids)

        # Обновляем данные
        data["tasks"][task_id]["processed_assignments"] = sorted(list(all_ids))
        data["tasks"][task_id]["total_processed"] = len(all_ids)
        data["tasks"][task_id]["last_processed"] = datetime.now().isoformat()
        data["tasks"][task_id]["task_name"] = task_name  # Обновляем название на случай изменения
        data["last_updated"] = datetime.now().isoformat()

        # Сохраняем
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            log.info(
                f"Сохранено {len(new_assignment_ids)} новых assignment_id для task={task_id} "
                f"(всего {len(all_ids)})"
            )

        except Exception as e:
            log.error(f"Ошибка при сохранении processed_assignments: {e}")

    def filter_new_assignments(
        self,
        project_id: str,
        task_id: str,
        all_assignment_ids: List[str]
    ) -> List[str]:
        """
        Фильтрует список assignments, оставляя только новые (необработанные).

        Args:
            project_id: ID проекта
            task_id: ID задачи
            all_assignment_ids: Полный список assignment_id

        Returns:
            Список только новых assignment_id
        """
        processed_ids = self.load_processed_ids(project_id, task_id)

        new_ids = [aid for aid in all_assignment_ids if aid not in processed_ids]

        log.info(
            f"Фильтрация assignments: всего={len(all_assignment_ids)}, "
            f"обработано ранее={len(processed_ids)}, новых={len(new_ids)}"
        )

        return new_ids

    def get_stats(self, project_id: str) -> Dict[str, Any]:
        """
        Получает статистику по проекту.

        Args:
            project_id: ID проекта

        Returns:
            Словарь со статистикой
        """
        file_path = self._get_project_file(project_id)

        if not file_path.exists():
            return {
                "project_id": project_id,
                "total_tasks": 0,
                "total_assignments": 0,
                "tasks": {}
            }

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            tasks_data = data.get("tasks", {})
            total_assignments = sum(
                task.get("total_processed", 0)
                for task in tasks_data.values()
            )

            return {
                "project_id": project_id,
                "total_tasks": len(tasks_data),
                "total_assignments": total_assignments,
                "last_updated": data.get("last_updated"),
                "tasks": {
                    task_id: {
                        "task_name": task_data.get("task_name"),
                        "total_processed": task_data.get("total_processed", 0),
                        "last_processed": task_data.get("last_processed")
                    }
                    for task_id, task_data in tasks_data.items()
                }
            }

        except Exception as e:
            log.error(f"Ошибка при получении статистики: {e}")
            return {
                "project_id": project_id,
                "error": str(e)
            }

"""
Скрипт для скачивания результатов разметки и расчета метрик качества.

Функциональность:
- Скачивание assignments по task_id через TagMe SDK
- Расчет метрик качества (согласие разметчиков, точность и т.д.)
- Сохранение результатов в JSON, CSV и Excel форматах
- Идентификация проблемных заданий для комментирования
"""
import asyncio
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from crowd_sdk.tagme import TagmeClientAdvanced

# Добавляем родительскую директорию в sys.path для импорта utils
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from utils.quality_metrics import (
    calculate_overlap_agreement,
    calculate_task_quality_summary,
    identify_problem_assignments,
    calculate_marker_quality,
)


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
log = logging.getLogger("calc_quality")


class QualityCalculator:
    """Класс для скачивания результатов и расчета метрик качества."""

    def __init__(self, client: TagmeClientAdvanced, data_dir: Path, organization_id: Optional[str] = None):
        """
        Инициализация калькулятора качества.

        Args:
            client: Инициализированный TagMe клиент
            data_dir: Директория для сохранения данных
            organization_id: ID организации в TagMe (для доступа к задачам)
        """
        self.client = client
        self.data_dir = data_dir
        self.organization_id = organization_id
        self.data_dir.mkdir(parents=True, exist_ok=True)

    async def download_assignments(self, task_id: str) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
        """
        Скачивает assignments для задачи.

        Args:
            task_id: ID задачи в TagMe

        Returns:
            Кортеж (список assignments как dict, DataFrame с assignments)
        """
        log.info(f"Скачивание assignments для задачи {task_id}...")

        try:
            # Получаем assignments как DataFrame с указанием organization_id
            df = await self.client.get_task_assignments_df(
                task_id,
                organization_id=self.organization_id
            )

            if df is None or df.empty:
                log.warning(f"Задача {task_id}: assignments не найдены")
                return [], pd.DataFrame()

            log.info(f"Загружено {len(df)} assignments")

            # Конвертируем DataFrame в список словарей для анализа
            assignments_list = df.to_dict('records')

            return assignments_list, df

        except Exception as e:
            log.error(f"Ошибка при скачивании assignments для задачи {task_id}: {e}")
            raise

    def detect_output_fields(self, assignments: List[Dict[str, Any]]) -> List[str]:
        """
        Определяет поля с ответами (output fields) из assignments.

        Args:
            assignments: Список assignments

        Returns:
            Список названий полей с ответами
        """
        output_fields = set()

        for assignment in assignments:
            output_data = assignment.get("output_data", {})
            if isinstance(output_data, dict):
                output_fields.update(output_data.keys())

        # Фильтруем служебные поля
        excluded_fields = {"__time_spent_ms", "__assignment_id", "__metadata"}
        output_fields = [f for f in sorted(output_fields) if f not in excluded_fields]

        log.info(f"Обнаружено {len(output_fields)} полей для анализа: {output_fields}")
        return output_fields

    def group_assignments_by_task(self, assignments: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Группирует assignments по заданиям (task/input_data).

        Args:
            assignments: Список assignments

        Returns:
            Словарь {task_id: [assignments для этого задания]}
        """
        grouped = defaultdict(list)

        for assignment in assignments:
            # Пытаемся найти уникальный идентификатор задания
            input_data = assignment.get("input_data", {})

            # Используем несколько стратегий для определения task_id
            task_id = None

            if "task_id" in input_data:
                task_id = str(input_data["task_id"])
            elif "id" in input_data:
                task_id = str(input_data["id"])
            else:
                # Если нет явного ID, используем хеш input_data
                task_id = str(hash(json.dumps(input_data, sort_keys=True)))

            grouped[task_id].append(assignment)

        log.info(f"Assignments сгруппированы по {len(grouped)} уникальным заданиям")
        return grouped

    async def calculate_quality_metrics(self,
                                       task_id: str,
                                       assignments: List[Dict[str, Any]],
                                       output_fields: List[str]) -> Dict[str, Any]:
        """
        Вычисляет метрики качества для задачи.

        Args:
            task_id: ID задачи
            assignments: Список assignments
            output_fields: Список полей для анализа

        Returns:
            Словарь с метриками качества
        """
        log.info(f"Расчет метрик качества для задачи {task_id}...")

        if not assignments:
            return {
                "task_id": task_id,
                "error": "No assignments available"
            }

        # Группируем по заданиям
        assignments_by_task = self.group_assignments_by_task(assignments)

        # Общие метрики по задаче
        summary = calculate_task_quality_summary(assignments, output_fields)

        # Проблемные задания
        problem_tasks = identify_problem_assignments(
            assignments_by_task,
            output_fields,
            agreement_threshold=0.5
        )

        # Метрики по разметчикам
        marker_quality = calculate_marker_quality(assignments, output_fields)

        metrics = {
            "task_id": task_id,
            "calculated_at": datetime.now().isoformat(),
            "summary": summary,
            "problem_tasks": problem_tasks,
            "marker_quality": marker_quality,
            "output_fields": output_fields
        }

        log.info(f"Метрики рассчитаны: "
                f"{summary['total_assignments']} assignments, "
                f"{summary['unique_tasks']} заданий, "
                f"{len(problem_tasks)} проблемных заданий")

        return metrics

    def save_results(self,
                    task_id: str,
                    task_name: str,
                    run_date: str,
                    assignments_df: pd.DataFrame,
                    assignments_list: List[Dict[str, Any]],
                    quality_metrics: Dict[str, Any]) -> Dict[str, Path]:
        """
        Сохраняет результаты в различных форматах.

        Args:
            task_id: ID задачи
            task_name: Название задачи
            run_date: Дата запуска (YYYY-MM-DD)
            assignments_df: DataFrame с assignments
            assignments_list: Список assignments как словари
            quality_metrics: Вычисленные метрики качества

        Returns:
            Словарь с путями к сохраненным файлам
        """
        # Создаем директорию для задачи
        task_dir = self.data_dir / task_id / run_date
        task_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # 1. Сохраняем сырые assignments в JSON
        assignments_json_path = task_dir / "assignments.json"
        with open(assignments_json_path, "w", encoding="utf-8") as f:
            json.dump(assignments_list, f, ensure_ascii=False, indent=2, default=str)
        saved_files["assignments_json"] = assignments_json_path
        log.info(f"Сохранен JSON: {assignments_json_path}")

        # 2. Сохраняем assignments в CSV
        if not assignments_df.empty:
            assignments_csv_path = task_dir / "assignments.csv"
            assignments_df.to_csv(assignments_csv_path, index=False, encoding="utf-8")
            saved_files["assignments_csv"] = assignments_csv_path
            log.info(f"Сохранен CSV: {assignments_csv_path}")

            # 3. Сохраняем assignments в Excel
            assignments_excel_path = task_dir / "assignments.xlsx"
            try:
                with pd.ExcelWriter(assignments_excel_path, engine='openpyxl') as writer:
                    assignments_df.to_excel(writer, sheet_name='Assignments', index=False)
                saved_files["assignments_excel"] = assignments_excel_path
                log.info(f"Сохранен Excel: {assignments_excel_path}")
            except Exception as e:
                log.warning(f"Не удалось сохранить Excel: {e}")

        # 4. Сохраняем метрики качества в JSON
        metrics_json_path = task_dir / "quality_metrics.json"
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(quality_metrics, f, ensure_ascii=False, indent=2, default=str)
        saved_files["quality_metrics"] = metrics_json_path
        log.info(f"Сохранены метрики: {metrics_json_path}")

        # 5. Создаем Excel отчет с метриками
        self._create_quality_excel_report(
            task_dir,
            task_name,
            assignments_df,
            quality_metrics,
            saved_files
        )

        # 6. Сохраняем проблемные задания отдельно
        problem_tasks = quality_metrics.get("problem_tasks", [])
        if problem_tasks:
            problem_tasks_path = task_dir / "problem_tasks.json"
            with open(problem_tasks_path, "w", encoding="utf-8") as f:
                json.dump(problem_tasks, f, ensure_ascii=False, indent=2, default=str)
            saved_files["problem_tasks"] = problem_tasks_path
            log.info(f"Сохранены проблемные задания: {problem_tasks_path}")

        return saved_files

    def _create_quality_excel_report(self,
                                    task_dir: Path,
                                    task_name: str,
                                    assignments_df: pd.DataFrame,
                                    quality_metrics: Dict[str, Any],
                                    saved_files: Dict[str, Path]) -> None:
        """
        Создает Excel отчет с несколькими листами: метрики, проблемные задания, статистика по разметчикам.

        Args:
            task_dir: Директория для сохранения
            task_name: Название задачи
            assignments_df: DataFrame с assignments
            quality_metrics: Метрики качества
            saved_files: Словарь для добавления пути к отчету
        """
        report_path = task_dir / "quality_report.xlsx"

        try:
            with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
                # Лист 1: Общие метрики
                summary = quality_metrics.get("summary", {})
                summary_df = pd.DataFrame([{
                    "Параметр": "Название задачи",
                    "Значение": task_name
                }, {
                    "Параметр": "Всего assignments",
                    "Значение": summary.get("total_assignments", 0)
                }, {
                    "Параметр": "Уникальных заданий",
                    "Значение": summary.get("unique_tasks", 0)
                }, {
                    "Параметр": "Среднее перекрытие",
                    "Значение": f"{summary.get('average_overlap', 0):.2f}"
                }, {
                    "Параметр": "Среднее согласие",
                    "Значение": f"{summary.get('average_agreement', 0):.2%}"
                }, {
                    "Параметр": "Проблемных заданий",
                    "Значение": summary.get("problem_tasks_count", 0)
                }, {
                    "Параметр": "Доля проблемных заданий",
                    "Значение": f"{summary.get('problem_tasks_rate', 0):.2%}"
                }])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)

                # Лист 2: Статистика по разметчикам
                marker_quality = quality_metrics.get("marker_quality", {})
                if marker_quality:
                    marker_df = pd.DataFrame([
                        {
                            "Marker ID": marker_id,
                            "Всего assignments": stats["total_assignments"],
                            "Согласий": stats["agreements"],
                            "Разногласий": stats["disagreements"],
                            "Точность": f"{stats['accuracy']:.2%}"
                        }
                        for marker_id, stats in marker_quality.items()
                    ])
                    marker_df = marker_df.sort_values("Точность", ascending=False)
                    marker_df.to_excel(writer, sheet_name='Marker Quality', index=False)

                # Лист 3: Проблемные задания
                problem_tasks = quality_metrics.get("problem_tasks", [])
                if problem_tasks:
                    problem_df = pd.DataFrame([
                        {
                            "Task ID": task["task_id"],
                            "Перекрытие": task["overlap"],
                            "Согласие": f"{task['average_agreement']:.2%}",
                            "Полей с разногласиями": len([
                                f for f, m in task.get("field_metrics", {}).items()
                                if m.get("unique_values", 1) > 1
                            ])
                        }
                        for task in problem_tasks[:100]  # Топ 100 проблемных
                    ])
                    problem_df.to_excel(writer, sheet_name='Problem Tasks', index=False)

                # Лист 4: Raw assignments (если не слишком большой)
                if not assignments_df.empty and len(assignments_df) <= 10000:
                    assignments_df.to_excel(writer, sheet_name='Raw Assignments', index=False)

            saved_files["quality_report"] = report_path
            log.info(f"Создан отчет Excel: {report_path}")

        except Exception as e:
            log.warning(f"Не удалось создать Excel отчет: {e}")

    async def process_task(self,
                          task_id: str,
                          task_name: str,
                          run_date: str) -> Dict[str, Any]:
        """
        Полный цикл обработки задачи: скачивание, анализ, сохранение.

        Args:
            task_id: ID задачи в TagMe
            task_name: Название задачи
            run_date: Дата запуска (YYYY-MM-DD)

        Returns:
            Словарь с результатами обработки
        """
        log.info(f"Начало обработки задачи {task_id} ({task_name})")

        try:
            # 1. Скачиваем assignments
            assignments_list, assignments_df = await self.download_assignments(task_id)

            if not assignments_list:
                return {
                    "task_id": task_id,
                    "task_name": task_name,
                    "status": "no_data",
                    "error": "No assignments found"
                }

            # 2. Определяем поля для анализа
            output_fields = self.detect_output_fields(assignments_list)

            if not output_fields:
                log.warning(f"Задача {task_id}: не найдено полей для анализа")
                output_fields = []

            # 3. Вычисляем метрики качества
            quality_metrics = await self.calculate_quality_metrics(
                task_id,
                assignments_list,
                output_fields
            )

            # 4. Сохраняем результаты
            saved_files = self.save_results(
                task_id,
                task_name,
                run_date,
                assignments_df,
                assignments_list,
                quality_metrics
            )

            log.info(f"Задача {task_id} успешно обработана")

            return {
                "task_id": task_id,
                "task_name": task_name,
                "status": "success",
                "assignments_count": len(assignments_list),
                "quality_metrics": quality_metrics["summary"],
                "problem_tasks_count": len(quality_metrics.get("problem_tasks", [])),
                "saved_files": {k: str(v) for k, v in saved_files.items()}
            }

        except Exception as e:
            log.error(f"Ошибка при обработке задачи {task_id}: {e}", exc_info=True)
            return {
                "task_id": task_id,
                "task_name": task_name,
                "status": "error",
                "error": str(e)
            }


async def run(client: TagmeClientAdvanced,
             task_id: str,
             task_name: str,
             run_date: str,
             data_dir: Path,
             organization_id: Optional[str] = None,
             **kwargs) -> Dict[str, Any]:
    """
    Точка входа для запуска скрипта из qa_manager.

    Args:
        client: Инициализированный TagMe клиент
        task_id: ID задачи
        task_name: Название задачи
        run_date: Дата запуска (YYYY-MM-DD)
        data_dir: Директория для сохранения данных
        organization_id: ID организации (для доступа к задачам)
        **kwargs: Дополнительные параметры

    Returns:
        Словарь с результатами выполнения
    """
    calculator = QualityCalculator(client, data_dir, organization_id)
    result = await calculator.process_task(task_id, task_name, run_date)
    return result


# Для тестирования скрипта напрямую
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Скачивание и анализ качества разметки")
    parser.add_argument("--task-id", required=True, help="ID задачи в TagMe")
    parser.add_argument("--task-name", default="Test Task", help="Название задачи")
    parser.add_argument("--config", default="~/.crowd.cfg", help="Путь к конфигу TagMe")
    parser.add_argument("--data-dir", default="./data", help="Директория для данных")
    parser.add_argument("--run-date", default=None, help="Дата запуска (YYYY-MM-DD)")

    args = parser.parse_args()

    if args.run_date is None:
        from datetime import date
        args.run_date = date.today().isoformat()

    async def main():
        config_path = Path(args.config).expanduser()
        client = TagmeClientAdvanced(str(config_path))

        data_dir = Path(args.data_dir).resolve()

        result = await run(
            client=client,
            task_id=args.task_id,
            task_name=args.task_name,
            run_date=args.run_date,
            data_dir=data_dir
        )

        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))

    asyncio.run(main())

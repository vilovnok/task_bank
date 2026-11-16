"""
Скрипт расчета качества для проекта "Выделение споттерных слов на аудио"

Особенности:
- Обрабатывает задания с временными метками (аудиосегменты)
- Вычисляет правильный ответ через consensus (если нет golden)
- Учитывает толерантность по времени (TIME_TOLERANCE_SEC)
- Сохраняет результаты временно для отладки (потом будут на ФИР)
- НЕ отправляет комментарии в TagMe
"""
import asyncio
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from crowd_sdk.tagme import TagmeClientAdvanced

# Добавляем родительскую директорию в sys.path для импорта utils
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from utils.processed_tracker import ProcessedAssignmentsTracker
from utils.excel_builder import ExcelReportBuilder


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
log = logging.getLogger("calc_quality_spotter")


# =========================
# ПАРАМЕТРЫ ПРОЕКТА
# =========================
MIN_MARKERS = 5  # Минимальное перекрытие для расчета consensus
CONSISTENCY_THRESHOLD = 0.8  # Порог согласованности
TIME_TOLERANCE_SEC = 0.15  # Толерантность по времени (секунды)

# Колонки проекта
OUTPUT_COL = "OUTPUT:marks"  # Колонка с ответами разметчика
GOLDEN_COL = "GOLDEN:marks"  # Колонка с golden-ответами (если есть)
ITEM_ID_COL = "item_id"  # ID задания
MARKER_ID_COL = "marker_id"  # ID разметчика


# =========================
# ФУНКЦИИ ДЛЯ РАБОТЫ С МЕТКАМИ
# =========================

def parse_marks(marks):
    """Парсит метки из разных форматов в список словарей."""
    if pd.isna(marks):
        return []
    if isinstance(marks, str):
        try:
            return json.loads(marks)
        except:
            return []
    if isinstance(marks, list):
        return marks
    return []


def normalize_marks(marks):
    """
    Нормализует метки: сортирует по type, entityId, position.

    Формат метки:
    {
        "type": "keyword",
        "entityId": "word_1",
        "position": {"start": 1.23, "finish": 2.45}
    }
    """
    normalized = []

    for seg in marks or []:
        if not isinstance(seg, dict):
            continue

        position = seg.get("position") or {}
        normalized.append({
            "type": seg.get("type"),
            "entityId": seg.get("entityId"),
            "position": {
                "start": float(position.get("start", 0)),
                "finish": float(position.get("finish", 0)),
            }
        })

    # Сортируем для сравнения
    normalized.sort(
        key=lambda x: (
            str(x.get("type")),
            str(x.get("entityId")),
            x["position"]["start"],
            x["position"]["finish"],
        )
    )
    return normalized


def marks_match(marks1, marks2, time_tolerance_sec=TIME_TOLERANCE_SEC):
    """
    Проверяет, совпадают ли две разметки с учетом толерантности по времени.

    Args:
        marks1: Первая разметка
        marks2: Вторая разметка
        time_tolerance_sec: Максимальная разница во времени (секунды)

    Returns:
        True если разметки совпадают, False иначе
    """
    m1 = normalize_marks(marks1)
    m2 = normalize_marks(marks2)

    # Обе пустые - совпадают
    if not m1 and not m2:
        return True

    # Разное количество сегментов - не совпадают
    if len(m1) != len(m2):
        return False

    # Сравниваем каждый сегмент
    for seg1, seg2 in zip(m1, m2):
        # Type и entityId должны совпадать точно
        if seg1.get("type") != seg2.get("type"):
            return False

        if seg1.get("entityId") != seg2.get("entityId"):
            return False

        # Временные метки - с толерантностью
        start_diff = abs(seg1["position"]["start"] - seg2["position"]["start"])
        finish_diff = abs(seg1["position"]["finish"] - seg2["position"]["finish"])

        if start_diff >= time_tolerance_sec or finish_diff >= time_tolerance_sec:
            return False

    return True


def get_consensus_reference(all_marks, time_tolerance_sec=TIME_TOLERANCE_SEC):
    """
    Находит consensus (мажоритарный ответ) среди всех разметок.

    Выбирает разметку, которая совпадает с наибольшим количеством других.
    Если кластер согласных < 2, возвращает None (нет консенсуса).

    Args:
        all_marks: Список всех разметок для задания
        time_tolerance_sec: Толерантность по времени

    Returns:
        Consensus разметка или None если консенсуса нет
    """
    if not all_marks:
        return None

    total = len(all_marks)
    support_counts = []

    # Для каждой разметки считаем, сколько других с ней совпадают
    for i, marks_i in enumerate(all_marks):
        support = 0
        for j, marks_j in enumerate(all_marks):
            if i == j:
                continue
            if marks_match(marks_i, marks_j, time_tolerance_sec=time_tolerance_sec):
                support += 1
        support_counts.append(support)

    # Берем разметку с максимальной поддержкой
    best_idx = int(np.argmax(support_counts))
    best_marks = all_marks[best_idx]

    # Считаем размер кластера согласных
    cluster_indices = []
    for i, marks_i in enumerate(all_marks):
        if marks_match(best_marks, marks_i, time_tolerance_sec=time_tolerance_sec):
            cluster_indices.append(i)

    # Если кластер < 2, значит нет консенсуса
    if len(cluster_indices) < 2:
        return None

    return best_marks


# =========================
# ОСНОВНОЙ КЛАСС
# =========================

class SpotterQualityCalculator:
    """Калькулятор качества для проекта Spotter."""

    def __init__(self, client: TagmeClientAdvanced, organization_id: Optional[str] = None):
        """
        Инициализация калькулятора.

        Args:
            client: TagMe клиент
            organization_id: ID организации
        """
        self.client = client
        self.organization_id = organization_id

    async def download_assignments(self, task_id: str, project_id: str, tracker: Optional[ProcessedAssignmentsTracker] = None) -> pd.DataFrame:
        """
        Скачивает assignments для задачи и фильтрует уже обработанные.

        Args:
            task_id: ID задачи
            project_id: ID проекта (для трекинга)
            tracker: Трекер обработанных assignments (опционально)

        Returns:
            DataFrame с новыми assignments
        """
        log.info(f"Скачивание assignments для задачи {task_id}...")

        try:
            df = await self.client.get_task_assignments_df(
                task_id,
                organization_id=self.organization_id
            )

            if df is None or df.empty:
                log.warning(f"Задача {task_id}: assignments не найдены")
                return pd.DataFrame()

            log.info(f"Загружено {len(df)} assignments")

            # Фильтруем уже обработанные assignments
            if tracker and "assignment_id" in df.columns:
                all_assignment_ids = df["assignment_id"].tolist()
                new_assignment_ids = tracker.filter_new_assignments(project_id, task_id, all_assignment_ids)

                if not new_assignment_ids:
                    log.info("Все assignments уже были обработаны ранее")
                    return pd.DataFrame()

                # Оставляем только новые
                df = df[df["assignment_id"].isin(new_assignment_ids)].copy()
                log.info(f"После фильтрации осталось {len(df)} новых assignments")
            else:
                log.warning("Колонка 'assignment_id' не найдена или трекер не передан, обрабатываем все")

            return df

        except Exception as e:
            log.error(f"Ошибка при скачивании assignments: {e}")
            raise

    def calculate_quality_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитывает метрики качества для датафрейма.

        Добавляет колонки:
        - parsed_marks: распарсенные метки разметчика
        - parsed_golden_marks: распарсенные golden метки
        - right_answer: правильный ответ (из golden или consensus)
        - score: 1 если ответ правильный, 0 если нет
        - mean_consistency_by_item: средняя согласованность по заданию

        Args:
            df: DataFrame с assignments

        Returns:
            DataFrame с добавленными метриками
        """
        log.info(f"Расчет метрик качества для {len(df)} assignments...")

        df = df.copy()

        # Проверяем наличие необходимых колонок
        required_cols = [OUTPUT_COL, ITEM_ID_COL, MARKER_ID_COL]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            log.error(f"Отсутствуют необходимые колонки: {missing_cols}")
            return df

        # Парсим метки
        df["parsed_marks"] = df[OUTPUT_COL].apply(parse_marks)

        if GOLDEN_COL in df.columns:
            df["parsed_golden_marks"] = df[GOLDEN_COL].apply(parse_marks)
        else:
            df["parsed_golden_marks"] = [[] for _ in range(len(df))]

        # Фильтруем задания с минимальным перекрытием
        marker_counts = df.groupby(ITEM_ID_COL)[MARKER_ID_COL].nunique()
        valid_items = marker_counts[marker_counts >= MIN_MARKERS].index

        initial_count = len(df)
        df = df[df[ITEM_ID_COL].isin(valid_items)].copy()
        filtered_count = initial_count - len(df)

        if filtered_count > 0:
            log.info(f"Отфильтровано {filtered_count} assignments с перекрытием < {MIN_MARKERS}")

        # Инициализируем колонки
        df["right_answer"] = None
        df["score"] = np.nan

        items_to_drop = set()

        # Для каждого задания находим правильный ответ
        for item_id, group in df.groupby(ITEM_ID_COL):

            # Проверяем, есть ли golden ответ
            has_golden = group["parsed_golden_marks"].apply(lambda x: len(x) > 0).any()

            if has_golden:
                # Берем golden как правильный ответ
                reference = next(
                    marks for marks in group["parsed_golden_marks"]
                    if len(marks) > 0
                )
                log.debug(f"Задание {item_id}: используется golden ответ")
            else:
                # Находим consensus
                reference = get_consensus_reference(
                    group["parsed_marks"].tolist(),
                    time_tolerance_sec=TIME_TOLERANCE_SEC
                )

                if reference is None:
                    log.debug(f"Задание {item_id}: consensus не найден, пропускаем")
                    items_to_drop.add(item_id)
                    continue

                log.debug(f"Задание {item_id}: используется consensus")

            # Считаем согласованность
            matches = []
            for _, row in group.iterrows():
                match = marks_match(
                    row["parsed_marks"],
                    reference,
                    time_tolerance_sec=TIME_TOLERANCE_SEC
                )
                matches.append(int(match))

            consistency = np.mean(matches)

            # Фильтруем задания с низкой согласованностью
            if consistency < CONSISTENCY_THRESHOLD:
                log.debug(f"Задание {item_id}: низкая согласованность {consistency:.2f}, пропускаем")
                items_to_drop.add(item_id)
                continue

            # Сохраняем правильный ответ
            df.loc[group.index, "right_answer"] = pd.Series(
                [reference] * len(group),
                index=group.index
            )

        # Удаляем задания без консенсуса
        before_drop = len(df)
        df = df[~df[ITEM_ID_COL].isin(items_to_drop)].copy()
        dropped_count = before_drop - len(df)

        if dropped_count > 0:
            log.info(f"Отфильтровано {dropped_count} assignments без консенсуса или с низкой согласованностью")

        # Вычисляем score для каждого assignment
        for idx, row in df.iterrows():
            # Проверяем что right_answer не None и не пустой
            right_answer = row["right_answer"]
            if right_answer is None or (isinstance(right_answer, list) and len(right_answer) == 0):
                continue

            score = marks_match(
                row["parsed_marks"],
                right_answer,
                time_tolerance_sec=TIME_TOLERANCE_SEC
            )
            df.at[idx, "score"] = int(score)

        # Средняя согласованность по заданию
        mean_consistency = (
            df.groupby(ITEM_ID_COL)["score"]
            .mean()
            .rename("mean_consistency_by_item")
        )
        df = df.merge(mean_consistency, on=ITEM_ID_COL, how="left")

        log.info(f"Метрики рассчитаны для {len(df)} assignments")

        return df

    def build_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Строит статистику по качеству разметки.

        Args:
            df: DataFrame с метриками

        Returns:
            Словарь со статистикой
        """
        if df.empty or "score" not in df.columns:
            return {
                "total_assignments": 0,
                "total_items": 0,
                "average_quality": 0.0,
                "markers_stats": []
            }

        # Общая статистика
        total_assignments = len(df)
        total_items = df[ITEM_ID_COL].nunique()
        average_quality = df["score"].mean()

        # Статистика по разметчикам
        markers_stats = []
        for marker_id, group in df.groupby(MARKER_ID_COL):
            stats = {
                "marker_id": marker_id,
                "total_assignments": len(group),
                "correct_assignments": int(group["score"].sum()),
                "accuracy": group["score"].mean(),
                "mean_consistency": group["mean_consistency_by_item"].mean() if "mean_consistency_by_item" in group.columns else 0.0
            }
            markers_stats.append(stats)

        # Сортируем по точности
        markers_stats.sort(key=lambda x: x["accuracy"])

        return {
            "total_assignments": total_assignments,
            "total_items": total_items,
            "average_quality": average_quality,
            "markers_stats": markers_stats
        }

    def prepare_excel_data(self, df: pd.DataFrame, stats: Dict[str, Any],
                          task_id: str, task_name: str) -> Dict[str, Any]:
        """
        Подготавливает данные для Excel отчета (вместо сохранения множества файлов).

        Args:
            df: DataFrame с данными и метриками
            stats: Статистика
            task_id: ID задачи
            task_name: Название задачи

        Returns:
            Словарь с данными для разных листов Excel
        """
        log.info("Подготовка данных для Excel отчета...")

        # 1. Данные для листа "Качество"
        pool_stats = {
            "total_items": stats["total_items"],
            "average_quality": stats["average_quality"],
            "bad_items_count": int((df.groupby(ITEM_ID_COL)["score"].mean() == 0).sum()) if "score" in df.columns else 0,
            "total_assignments": stats["total_assignments"]
        }

        # Статистика по разметчикам (сортировка по возрастанию accuracy)
        marker_stats_df = pd.DataFrame(stats["markers_stats"])
        if not marker_stats_df.empty:
            marker_stats_df = marker_stats_df.rename(columns={
                "marker_id": "ID разметчика",
                "total_assignments": "Всего заданий",
                "correct_assignments": "Правильных",
                "accuracy": "Качество (score)",
                "mean_consistency": "Средняя согласованность"
            })
            # Сортируем по возрастанию (от худших к лучшим)
            marker_stats_df = marker_stats_df.sort_values("Качество (score)", ascending=True)
            # Форматируем числа
            marker_stats_df["Качество (score)"] = marker_stats_df["Качество (score)"].apply(lambda x: round(x, 3))
            marker_stats_df["Средняя согласованность"] = marker_stats_df["Средняя согласованность"].apply(lambda x: round(x, 3))

        # 2. Данные для листа "Проблемные задания"
        # Группируем по item_id и считаем метрики
        problems_df = pd.DataFrame()
        if "score" in df.columns and "mean_consistency_by_item" in df.columns:
            item_groups = df.groupby(ITEM_ID_COL).agg({
                "score": ["mean", "count"],
                "mean_consistency_by_item": "first",
                MARKER_ID_COL: "nunique"
            })
            item_groups.columns = ["Качество (score)", "Кол-во разметок", "Согласованность", "Кол-во разметчиков"]
            item_groups = item_groups.reset_index()
            item_groups = item_groups.rename(columns={ITEM_ID_COL: "ID задания"})

            # Сортируем от наиболее проблемных к наименее
            # (сначала по низкому качеству, потом по низкой согласованности)
            item_groups = item_groups.sort_values(
                ["Качество (score)", "Согласованность"],
                ascending=[True, True]
            )

            # Форматируем числа
            item_groups["Качество (score)"] = item_groups["Качество (score)"].apply(lambda x: round(x, 3))
            item_groups["Согласованность"] = item_groups["Согласованность"].apply(lambda x: round(x, 3))

            problems_df = item_groups

        # 3. Плохие assignments (для stage 2 - генерации комментариев)
        bad_assignments_data = None
        if "score" in df.columns:
            bad_assignments_df = df[df["score"] == 0].copy()
            if not bad_assignments_df.empty:
                # Сохраняем полные данные для передачи в следующие стадии
                # Включаем assignment_id, ответы разметчика и правильный ответ
                columns_to_save = [
                    "assignment_id", ITEM_ID_COL, MARKER_ID_COL, OUTPUT_COL,
                    "parsed_marks", "right_answer", "score"
                ]
                bad_assignments_data = bad_assignments_df[columns_to_save].to_dict('records')
                log.info(f"Найдено {len(bad_assignments_data)} плохих assignments для генерации комментариев")
            else:
                log.info("Плохих assignments не найдено (все правильно!)")

        return {
            "pool_stats": pool_stats,
            "marker_stats_df": marker_stats_df.to_dict('records') if not marker_stats_df.empty else [],
            "problems_df": problems_df.to_dict('records') if not problems_df.empty else [],
            "bad_assignments_data": bad_assignments_data,
            "full_df": df.to_dict('records') if not df.empty else []  # Конвертируем в list для JSON
        }

    async def process_task(self, task_id: str, task_name: str,
                          output_dir: Path, project_id: str = None,
                          tracker: Optional[ProcessedAssignmentsTracker] = None,
                          excel_builder: Optional[ExcelReportBuilder] = None) -> Dict[str, Any]:
        """
        Полный цикл обработки задачи.

        Args:
            task_id: ID задачи
            task_name: Название задачи
            output_dir: Директория для временных файлов
            project_id: ID проекта (для трекинга)
            tracker: Трекер обработанных assignments
            excel_builder: Builder для единого Excel отчета (если нужно сразу создавать листы)

        Returns:
            Результат обработки с данными для Excel
        """
        log.info(f"Обработка задачи {task_id} ({task_name})")

        try:
            # 1. Скачиваем данные (с фильтрацией обработанных)
            df = await self.download_assignments(task_id, project_id, tracker)

            if df.empty:
                return {
                    "status": "no_data",
                    "task_id": task_id,
                    "task_name": task_name,
                    "error": "No assignments found"
                }

            # 2. Рассчитываем метрики
            df = self.calculate_quality_metrics(df)

            if df.empty:
                return {
                    "status": "filtered_out",
                    "task_id": task_id,
                    "task_name": task_name,
                    "error": "All assignments filtered out (low overlap or consistency)"
                }

            # 3. Строим статистику
            stats = self.build_statistics(df)

            # 4. Подготавливаем данные для Excel отчета
            excel_data = self.prepare_excel_data(df, stats, task_id, task_name)

            # 5. Если передан excel_builder, сразу добавляем листы "Качество" и "Проблемные задания"
            if excel_builder:
                log.info("Добавляем листы 'Качество' и 'Проблемные задания' в Excel отчет...")

                # Конвертируем обратно в DataFrame для Excel builder
                marker_stats_df = pd.DataFrame(excel_data["marker_stats_df"]) if excel_data["marker_stats_df"] else pd.DataFrame()
                problems_df = pd.DataFrame(excel_data["problems_df"]) if excel_data["problems_df"] else pd.DataFrame()

                excel_builder.add_quality_sheet(
                    pool_stats=excel_data["pool_stats"],
                    marker_stats=marker_stats_df
                )
                excel_builder.add_problems_sheet(problems_df)
                log.info("Листы успешно добавлены")

            # Подсчитываем количество плохих assignments
            bad_assignments_count = 0
            if excel_data["bad_assignments_data"]:
                bad_assignments_count = len(excel_data["bad_assignments_data"])

            # Собираем список assignment_id для трекинга
            processed_assignment_ids = []
            if "assignment_id" in df.columns:
                processed_assignment_ids = df["assignment_id"].tolist()

            log.info(f"Задача {task_id} обработана успешно")

            return {
                "status": "success",
                "task_id": task_id,
                "task_name": task_name,
                "assignments_count": len(df),
                "items_count": stats["total_items"],
                "average_quality": stats["average_quality"],
                "bad_assignments_count": bad_assignments_count,
                "processed_assignment_ids": processed_assignment_ids,  # Для трекинга
                "excel_data": excel_data  # Данные для Excel (для следующих стадий)
            }

        except Exception as e:
            log.error(f"Ошибка при обработке задачи {task_id}: {e}", exc_info=True)
            return {
                "status": "error",
                "task_id": task_id,
                "task_name": task_name,
                "error": str(e)
            }


# =========================
# ТОЧКА ВХОДА ДЛЯ qa_manager
# =========================

async def run(client: TagmeClientAdvanced,
             task_id: str,
             task_name: str,
             run_date: str,
             data_dir: Path,
             organization_id: Optional[str] = None,
             project_id: Optional[str] = None,
             state_dir: Optional[Path] = None,
             excel_builder: Optional[ExcelReportBuilder] = None,
             **kwargs) -> Dict[str, Any]:
    """
    Точка входа для запуска из qa_manager.

    Args:
        client: TagMe клиент
        task_id: ID задачи
        task_name: Название задачи
        run_date: Дата запуска
        data_dir: Директория для временных файлов
        organization_id: ID организации
        project_id: ID проекта (для трекинга)
        state_dir: Директория для состояния (для трекинга)
        excel_builder: Builder для единого Excel отчета
        **kwargs: Дополнительные параметры

    Returns:
        Результат выполнения (включает excel_data для следующих стадий)
    """
    # Временная директория для отладки
    temp_dir = data_dir / "temp" / run_date

    # Создаем трекер если передан project_id и state_dir
    tracker = None
    if project_id and state_dir:
        tracker = ProcessedAssignmentsTracker(state_dir)
        log.info(f"Трекер обработанных assignments активирован для project_id={project_id}")

    calculator = SpotterQualityCalculator(client, organization_id)
    result = await calculator.process_task(
        task_id, task_name, temp_dir, project_id, tracker, excel_builder
    )

    return result


# =========================
# ДЛЯ ПРЯМОГО ЗАПУСКА
# =========================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Расчет качества для Spotter проекта")
    parser.add_argument("--task-id", required=True, help="ID задачи")
    parser.add_argument("--task-name", default="Test Task", help="Название задачи")
    parser.add_argument("--config", default="~/.crowd.cfg", help="Путь к конфигу TagMe")
    parser.add_argument("--org-id", required=True, help="Organization ID")
    parser.add_argument("--output-dir", default="./temp_results", help="Директория для результатов")

    args = parser.parse_args()

    async def main():
        config_path = Path(args.config).expanduser()
        client = TagmeClientAdvanced(str(config_path))

        output_dir = Path(args.output_dir).resolve()

        result = await run(
            client=client,
            task_id=args.task_id,
            task_name=args.task_name,
            run_date=datetime.now().strftime("%Y-%m-%d"),
            data_dir=output_dir,
            organization_id=args.org_id
        )

        print(json.dumps(result, indent=2, ensure_ascii=False))

    asyncio.run(main())

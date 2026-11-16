"""
Модуль для формирования единого Excel отчета по задаче.

Создает один Excel файл с четырьмя листами:
1. "Качество" - общая статистика по пулу и разметчикам
2. "Проблемные задания" - задания с низкой согласованностью
3. "Комментарии" - LLM комментарии по плохим заданиям
4. "Обратная связь" - персональные сводки по разметчикам
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils.dataframe import dataframe_to_rows

log = logging.getLogger(__name__)


def _safe_cell_value(value):
    """
    Преобразует значение для безопасной записи в ячейку Excel.

    - list, dict -> JSON строка
    - pd.NA, None -> None
    - остальное -> как есть
    """
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    elif pd.isna(value):
        return None
    return value


class ExcelReportBuilder:
    """
    Построитель единого Excel отчета для задачи.

    Формирует файл с четырьмя листами последовательно:
    - Stage 1 (quality): добавляет листы "Качество" и "Проблемные задания"
    - Stage 2 (comments): добавляет лист "Комментарии"
    - Stage 3 (summary): добавляет лист "Обратная связь"
    """

    SHEET_QUALITY = "Качество"
    SHEET_PROBLEMS = "Проблемные задания"
    SHEET_COMMENTS = "Комментарии"
    SHEET_FEEDBACK = "Обратная связь"

    def __init__(self, output_path: Path):
        """
        Args:
            output_path: Путь к выходному Excel файлу
        """
        self.output_path = output_path
        self.workbook = None

    def init_workbook(self):
        """Инициализировать новый workbook или загрузить существующий."""
        if self.output_path.exists():
            log.info(f"Загружаем существующий файл: {self.output_path}")
            self.workbook = load_workbook(self.output_path)
        else:
            log.info(f"Создаем новый файл: {self.output_path}")
            self.workbook = Workbook()
            # Удаляем дефолтный лист
            if "Sheet" in self.workbook.sheetnames:
                del self.workbook["Sheet"]

    def add_quality_sheet(self, pool_stats: Dict[str, Any], marker_stats: pd.DataFrame):
        """
        Добавить лист "Качество" с общей статистикой.

        Args:
            pool_stats: Статистика по пулу (dict с ключами: total_items, average_quality, etc)
            marker_stats: DataFrame со статистикой по разметчикам (marker_id, score, assignments_count)
                         Должен быть отсортирован по возрастанию score
        """
        log.info(f"Добавляем лист '{self.SHEET_QUALITY}'")

        if self.workbook is None:
            self.init_workbook()

        # Удаляем лист если уже существует
        if self.SHEET_QUALITY in self.workbook.sheetnames:
            del self.workbook[self.SHEET_QUALITY]

        ws = self.workbook.create_sheet(self.SHEET_QUALITY, 0)

        # Секция 1: Статистика пула
        ws['A1'] = "ОБЩАЯ СТАТИСТИКА ПУЛА"
        ws['A1'].font = Font(bold=True, size=14)

        row = 3
        ws[f'A{row}'] = "Показатель"
        ws[f'B{row}'] = "Значение"
        ws[f'A{row}'].font = Font(bold=True)
        ws[f'B{row}'].font = Font(bold=True)

        row += 1
        ws[f'A{row}'] = "Всего заданий (items)"
        ws[f'B{row}'] = pool_stats.get("total_items", 0)

        row += 1
        ws[f'A{row}'] = "Среднее качество (score)"
        ws[f'B{row}'] = round(pool_stats.get("average_quality", 0), 3)

        row += 1
        ws[f'A{row}'] = "Проблемных заданий (score=0)"
        ws[f'B{row}'] = pool_stats.get("bad_items_count", 0)

        row += 1
        ws[f'A{row}'] = "Всего assignments"
        ws[f'B{row}'] = pool_stats.get("total_assignments", 0)

        # Секция 2: Статистика по разметчикам
        row += 3
        ws[f'A{row}'] = "КАЧЕСТВО РАЗМЕТЧИКОВ (от худших к лучшим)"
        ws[f'A{row}'].font = Font(bold=True, size=14)

        row += 2

        # Записываем DataFrame с разметчиками
        if not marker_stats.empty:
            # Данные уже отсортированы в calc_quality_spotter.py по возрастанию score
            # (от худших к лучшим), просто выводим как есть

            # Заголовки
            for col_idx, col_name in enumerate(marker_stats.columns, start=1):
                cell = ws.cell(row=row, column=col_idx, value=col_name)
                cell.font = Font(bold=True)
                cell.fill = PatternFill(start_color="D3D3D3", end_color="D3D3D3", fill_type="solid")

            # Данные
            for _, row_data in marker_stats.iterrows():
                row += 1
                for col_idx, value in enumerate(row_data, start=1):
                    ws.cell(row=row, column=col_idx, value=_safe_cell_value(value))

        # Автоширина колонок
        for column in ws.columns:
            max_length = 0
            column = [cell for cell in column]
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(cell.value)
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            ws.column_dimensions[column[0].column_letter].width = adjusted_width

        self._save()
        log.info(f"Лист '{self.SHEET_QUALITY}' добавлен")

    def add_problems_sheet(self, problems_df: pd.DataFrame):
        """
        Добавить лист "Проблемные задания".

        Args:
            problems_df: DataFrame с проблемными заданиями (item_id, consistency, marker_count, etc)
                        Должен быть отсортирован от наиболее проблемных к наименее
        """
        log.info(f"Добавляем лист '{self.SHEET_PROBLEMS}'")

        if self.workbook is None:
            self.init_workbook()

        # Удаляем лист если уже существует
        if self.SHEET_PROBLEMS in self.workbook.sheetnames:
            del self.workbook[self.SHEET_PROBLEMS]

        ws = self.workbook.create_sheet(self.SHEET_PROBLEMS, 1)

        if problems_df.empty:
            ws['A1'] = "Проблемных заданий не найдено"
            ws['A1'].font = Font(bold=True)
            self._save()
            return

        # Данные уже отсортированы в calc_quality_spotter.py от наиболее проблемных к наименее,
        # просто выводим как есть

        # Заголовки
        row = 1
        for col_idx, col_name in enumerate(problems_df.columns, start=1):
            cell = ws.cell(row=row, column=col_idx, value=col_name)
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color="FFD700", end_color="FFD700", fill_type="solid")

        # Данные
        for _, row_data in problems_df.iterrows():
            row += 1
            for col_idx, value in enumerate(row_data, start=1):
                ws.cell(row=row, column=col_idx, value=_safe_cell_value(value))

        # Автоширина колонок
        for column in ws.columns:
            max_length = 0
            column = [cell for cell in column]
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(cell.value)
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            ws.column_dimensions[column[0].column_letter].width = adjusted_width

        self._save()
        log.info(f"Лист '{self.SHEET_PROBLEMS}' добавлен ({len(problems_df)} заданий)")

    def add_comments_sheet(self, comments_df: pd.DataFrame):
        """
        Добавить лист "Комментарии".

        Args:
            comments_df: DataFrame с комментариями (item_id, marker_id, llm_comment, etc)
        """
        log.info(f"Добавляем лист '{self.SHEET_COMMENTS}'")

        if self.workbook is None:
            self.init_workbook()

        # Удаляем лист если уже существует
        if self.SHEET_COMMENTS in self.workbook.sheetnames:
            del self.workbook[self.SHEET_COMMENTS]

        ws = self.workbook.create_sheet(self.SHEET_COMMENTS, 2)

        if comments_df.empty:
            ws['A1'] = "Комментарии не были сгенерированы"
            ws['A1'].font = Font(bold=True)
            self._save()
            return

        # Заголовки
        row = 1
        for col_idx, col_name in enumerate(comments_df.columns, start=1):
            cell = ws.cell(row=row, column=col_idx, value=col_name)
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color="ADD8E6", end_color="ADD8E6", fill_type="solid")

        # Данные
        for _, row_data in comments_df.iterrows():
            row += 1
            for col_idx, value in enumerate(row_data, start=1):
                cell = ws.cell(row=row, column=col_idx, value=_safe_cell_value(value))
                # Увеличиваем высоту строки для комментариев
                if 'comment' in str(comments_df.columns[col_idx-1]).lower():
                    cell.alignment = Alignment(wrap_text=True, vertical='top')

        # Автоширина колонок (с особым вниманием к комментариям)
        for column in ws.columns:
            max_length = 0
            column = [cell for cell in column]
            col_name = str(column[0].value).lower()

            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(cell.value)
                except:
                    pass

            # Для колонок с комментариями устанавливаем фиксированную ширину
            if 'comment' in col_name:
                adjusted_width = 60
            else:
                adjusted_width = min(max_length + 2, 50)

            ws.column_dimensions[column[0].column_letter].width = adjusted_width

        self._save()
        log.info(f"Лист '{self.SHEET_COMMENTS}' добавлен ({len(comments_df)} записей)")

    def add_feedback_sheet(self, feedback_df: pd.DataFrame):
        """
        Добавить лист "Обратная связь".

        Args:
            feedback_df: DataFrame с обратной связью (marker_id, summary, error_count, etc)
        """
        log.info(f"Добавляем лист '{self.SHEET_FEEDBACK}'")

        if self.workbook is None:
            self.init_workbook()

        # Удаляем лист если уже существует
        if self.SHEET_FEEDBACK in self.workbook.sheetnames:
            del self.workbook[self.SHEET_FEEDBACK]

        ws = self.workbook.create_sheet(self.SHEET_FEEDBACK, 3)

        if feedback_df.empty:
            ws['A1'] = "Обратная связь не была сгенерирована"
            ws['A1'].font = Font(bold=True)
            self._save()
            return

        # Заголовки
        row = 1
        for col_idx, col_name in enumerate(feedback_df.columns, start=1):
            cell = ws.cell(row=row, column=col_idx, value=col_name)
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color="90EE90", end_color="90EE90", fill_type="solid")

        # Данные
        for _, row_data in feedback_df.iterrows():
            row += 1
            for col_idx, value in enumerate(row_data, start=1):
                cell = ws.cell(row=row, column=col_idx, value=_safe_cell_value(value))
                # Увеличиваем высоту строки для summaries
                if 'summary' in str(feedback_df.columns[col_idx-1]).lower():
                    cell.alignment = Alignment(wrap_text=True, vertical='top')

        # Автоширина колонок
        for column in ws.columns:
            max_length = 0
            column = [cell for cell in column]
            col_name = str(column[0].value).lower()

            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(cell.value)
                except:
                    pass

            # Для колонок с summary устанавливаем фиксированную ширину
            if 'summary' in col_name:
                adjusted_width = 60
            else:
                adjusted_width = min(max_length + 2, 50)

            ws.column_dimensions[column[0].column_letter].width = adjusted_width

        self._save()
        log.info(f"Лист '{self.SHEET_FEEDBACK}' добавлен ({len(feedback_df)} записей)")

    def _save(self):
        """Сохранить workbook."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.workbook.save(self.output_path)
        log.debug(f"Сохранен файл: {self.output_path}")

    def get_path(self) -> Path:
        """Получить путь к файлу отчета."""
        return self.output_path


# =========================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================

def create_report_for_task(
    task_id: str,
    task_name: str,
    output_dir: Path,
    pool_stats: Optional[Dict[str, Any]] = None,
    marker_stats: Optional[pd.DataFrame] = None,
    problems_df: Optional[pd.DataFrame] = None,
    comments_df: Optional[pd.DataFrame] = None,
    feedback_df: Optional[pd.DataFrame] = None
) -> Path:
    """
    Создать полный отчет для задачи (convenience функция).

    Args:
        task_id: ID задачи
        task_name: Название задачи
        output_dir: Директория для сохранения
        pool_stats: Статистика пула
        marker_stats: Статистика разметчиков
        problems_df: Проблемные задания
        comments_df: Комментарии
        feedback_df: Обратная связь

    Returns:
        Путь к созданному файлу
    """
    # Формируем имя файла
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"qa_report_{task_id[:8]}_{timestamp}.xlsx"
    output_path = output_dir / filename

    builder = ExcelReportBuilder(output_path)

    # Добавляем листы последовательно
    if pool_stats and marker_stats is not None:
        builder.add_quality_sheet(pool_stats, marker_stats)

    if problems_df is not None:
        builder.add_problems_sheet(problems_df)

    if comments_df is not None:
        builder.add_comments_sheet(comments_df)

    if feedback_df is not None:
        builder.add_feedback_sheet(feedback_df)

    log.info(f"Полный отчет создан: {output_path}")
    return output_path

"""
Stage "summary": Генерация персональных сводок для разметчиков

Адаптировано из summary_spotter.py для интеграции в qa_agent.

Особенности:
- Группирует ошибки по разметчикам
- Генерирует персональную обратную связь через LLM
- Сохраняет результаты в отдельный лист Excel
- Использует TagMe прокси для LLM
"""
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Добавляем родительскую директорию в sys.path
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from utils.vsellm_client import call_vsellm
from utils.processed_tracker import ProcessedAssignmentsTracker
from utils.excel_builder import ExcelReportBuilder


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
log = logging.getLogger("generate_summary")


# =========================
# CONFIG
# =========================

# LLM настройки
LLM_MODEL = "GigaChat-2-Max"
LLM_TIMEOUT_SEC = 180
LLM_TEMPERATURE = 0.6

# Колонки
MARKER_ID_COL = "marker_id"
COMMENT_COL = "llm_comment"

# Минимум ошибок для генерации сводки
MIN_ERRORS_FOR_SUMMARY = 5


# =========================
# PROMPTS
# =========================

SYSTEM_PROMPT = """\
Ты — опытный наставник по разметке данных. Твоя главная задача — дать разметчику обратную связь так, чтобы она была максимально понятной, полезной и применимой в работе: разметчик должен не просто увидеть ошибки, а понять логику и запомнить правила, которые помогут размечать лучше.

Обращение к разметчику: всегда на «Вы» (формы «у вас», «вам», «вы»). Никогда не используй «ты/тебе/твоя».

Тон: уважительный, поддерживающий, дружелюбный (как старший коллега, который помогает расти). Без давления и без морализаторства.

Стиль: простой человеческий язык, без канцелярита и "отчетных" формулировок.

Формат ответа:
- ТОЛЬКО связный текст абзацами
- допускаются короткие подзаголовки в одну строку: "🔍 Где возникают сложности", "📌 Запомните правило", "🧠 Полезная привычка"
- НЕ используй маркированные списки и нумерацию

Эмодзи:
- Используй 3–6 эмодзи на весь текст
- Эмодзи должны помогать структуре (👍 🔍 📌 🧠 💪)
- Не превращай текст в "чатик"

Обязательная логика текста:
1) Анализируй частые ошибки
2) Объясняй правила простыми словами
3) Давай короткие примеры
4) Заверши позитивно и мотивирующе

Цель: после ответа у разметчика должна сложиться ясная картина, как улучшить качество разметки.

Особенно для аудио:
учитывай границы, таймкоды, пропуски, лишние срабатывания, споттерные слова
"""

USER_PROMPT_TEMPLATE = """\
ИДЕНТИФИКАТОР РАЗМЕТЧИКА:
{marker_id}

ПРАВИЛА РАЗМЕТКИ:
{rules}

КОММЕНТАРИИ ПО ОШИБКАМ:
{explanations}

Сформируй персональную обратную связь для этого разметчика.
"""


# =========================
# HELPERS
# =========================

def load_prompt_from_file(prompt_file_path: Optional[Path], default_prompt: str, prompt_name: str = "prompt") -> str:
    """
    Загружает промпт из файла, если путь указан.
    Если файл не найден или путь не указан - возвращает дефолтный промпт.

    Args:
        prompt_file_path: Путь к файлу с промптом (опционально, может быть str или Path)
        default_prompt: Дефолтный промпт (fallback)
        prompt_name: Название промпта для логов

    Returns:
        Текст промпта
    """
    # Конвертируем в Path если пришла строка
    if prompt_file_path and not isinstance(prompt_file_path, Path):
        prompt_file_path = Path(prompt_file_path)

    if prompt_file_path and prompt_file_path.exists():
        log.info(f"Загружаем {prompt_name} из файла: {prompt_file_path}")
        try:
            with open(prompt_file_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            if prompt:
                return prompt
            else:
                log.warning(f"Файл {prompt_name} пустой: {prompt_file_path}, используем дефолтный")
        except Exception as e:
            log.warning(f"Ошибка чтения {prompt_name} из {prompt_file_path}: {e}, используем дефолтный")
    else:
        if prompt_file_path:
            log.warning(f"Файл {prompt_name} не найден: {prompt_file_path}, используем дефолтный")
        else:
            log.info(f"Путь к {prompt_name} не указан, используем дефолтный хардкоженный")

    return default_prompt


def collect_explanations(group: pd.DataFrame, max_n: int = 15) -> str:
    """
    Собирает комментарии по ошибкам для разметчика.

    Args:
        group: DataFrame с ошибками одного разметчика
        max_n: Максимум комментариев

    Returns:
        Объединенные комментарии
    """
    texts = []

    for comment in group[COMMENT_COL].tolist():
        if not isinstance(comment, str):
            continue
        if not comment.strip():
            continue
        if comment.startswith("ERROR"):
            continue
        texts.append(comment.strip())

    # Убираем дубликаты, сохраняя порядок
    unique_texts = list(dict.fromkeys(texts))
    unique_texts = unique_texts[:max_n]

    return "\n\n".join(unique_texts) if unique_texts else "Комментариев нет."


# =========================
# LLM CALL
# =========================

def generate_summary_for_marker(
    marker_id: str,
    explanations: str,
    rules: str,
    proxy_cfg_path: str,
    system_prompt: str = None,
    user_prompt_template: str = None
) -> str:
    """
    Генерирует персональную сводку для разметчика через LLM.

    Args:
        marker_id: ID разметчика
        explanations: Комментарии по ошибкам
        rules: Правила разметки
        proxy_cfg_path: Путь к конфигу TagMe
        system_prompt: Системный промпт (если None, использует SYSTEM_PROMPT)
        user_prompt_template: Шаблон пользовательского промпта (если None, использует USER_PROMPT_TEMPLATE)

    Returns:
        Персональная обратная связь
    """
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT
    if user_prompt_template is None:
        user_prompt_template = USER_PROMPT_TEMPLATE

    user_prompt = user_prompt_template.format(
        marker_id=marker_id,
        rules=rules,
        explanations=explanations
    )

    log.debug(f"Генерация сводки для marker_id={marker_id}")

    try:
        result = call_vsellm(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=LLM_TEMPERATURE,
            timeout_sec=LLM_TIMEOUT_SEC,
            proxy_cfg_path=proxy_cfg_path
        )

        summary = result.get("answer", "") or result.get("raw_text", "")
        return summary

    except Exception as e:
        log.error(f"Ошибка при генерации сводки для marker_id={marker_id}: {e}")
        return f"ERROR: {e}"


# =========================
# MAIN PIPELINE
# =========================

def build_marker_summaries(
    df_with_comments: pd.DataFrame,
    rules: str,
    proxy_cfg_path: str,
    system_prompt: str = None,
    user_prompt_template: str = None,
    min_errors: int = MIN_ERRORS_FOR_SUMMARY
) -> pd.DataFrame:
    """
    Строит персональные сводки для разметчиков.

    Args:
        df_with_comments: DataFrame с комментариями
        rules: Правила разметки
        proxy_cfg_path: Путь к конфигу TagMe
        system_prompt: Системный промпт (опционально)
        user_prompt_template: Шаблон пользовательского промпта (опционально)
        min_errors: Минимум ошибок для генерации сводки

    Returns:
        DataFrame с сводками по разметчикам
    """
    log.info("Построение персональных сводок для разметчиков...")

    # Фильтруем: только строки с валидными комментариями
    df_filtered = df_with_comments[
        (df_with_comments[COMMENT_COL].astype(str).str.strip() != "") &
        (~df_with_comments[COMMENT_COL].astype(str).str.startswith("ERROR"))
    ].copy()

    log.info(f"Строк после фильтрации: {len(df_filtered)}")

    if df_filtered.empty:
        log.warning("Нет данных для построения сводок")
        return pd.DataFrame()

    # Статистика
    total_markers = df_filtered[MARKER_ID_COL].nunique()
    marker_sizes = df_filtered.groupby(MARKER_ID_COL).size()
    selected_markers = marker_sizes[marker_sizes >= min_errors]

    log.info(f"Всего разметчиков с ошибками: {total_markers}")
    log.info(f"Разметчиков с >= {min_errors} ошибок: {len(selected_markers)}")

    results = []

    for marker_id, group in df_filtered.groupby(MARKER_ID_COL):
        group_size = len(group)

        if group_size < min_errors:
            log.info(f"Пропуск marker_id={marker_id} — мало ошибок ({group_size} < {min_errors})")
            continue

        log.info(f"Обработка marker_id={marker_id} | ошибок={group_size}")

        # Собираем комментарии
        explanations = collect_explanations(group)

        if not explanations or explanations.strip() == "Комментариев нет.":
            log.info(f"Пропуск marker_id={marker_id} — нет валидных комментариев")
            continue

        # Генерируем сводку
        try:
            feedback = generate_summary_for_marker(
                marker_id=marker_id,
                explanations=explanations,
                rules=rules,
                proxy_cfg_path=proxy_cfg_path,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt_template
            )

            log.info(f"Сводка для marker_id={marker_id} сгенерирована: {len(feedback)} символов")

        except Exception as e:
            feedback = f"ERROR: {e}"
            log.exception(f"Ошибка для marker_id={marker_id}")

        results.append({
            "marker_id": marker_id,
            "num_errors": group_size,
            "personal_feedback": feedback
        })

    result_df = pd.DataFrame(results)
    log.info(f"Всего сводок построено: {len(result_df)}")

    return result_df


# =========================
# STAGE RUNNER
# =========================

async def run(
    task_id: str,
    task_name: str,
    run_date: str,
    data_dir: Path,
    comments_result: Dict[str, Any],
    rules_file: Path,
    proxy_cfg_path: Optional[str] = None,
    summary_prompt_system_file: Optional[Path] = None,
    summary_prompt_user_file: Optional[Path] = None,
    project_id: Optional[str] = None,
    state_dir: Optional[Path] = None,
    excel_builder: Optional[ExcelReportBuilder] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Точка входа для stage "summary" из qa_manager.

    Args:
        task_id: ID задачи
        task_name: Название задачи
        run_date: Дата запуска
        data_dir: Директория с данными
        comments_result: Результат stage "comments"
        rules_file: Путь к файлу с правилами
        proxy_cfg_path: Путь к конфигу TagMe
        summary_prompt_system_file: Путь к файлу с системным промптом для summary
        summary_prompt_user_file: Путь к файлу с пользовательским промптом для summary
        project_id: ID проекта (для трекинга)
        state_dir: Директория состояния (для трекинга)
        excel_builder: Builder для добавления листа "Обратная связь" в единый Excel отчет
        **kwargs: Дополнительные параметры

    Returns:
        Результат выполнения stage
    """
    log.info(f"Stage 'summary' для задачи {task_id} ({task_name})")

    # Используем дефолтный конфиг если не передан
    if not proxy_cfg_path:
        proxy_cfg_path = str(Path.home() / "configs" / "ml_config.cfg")

    try:
        # 1. Проверяем что stage "comments" выполнен успешно
        if comments_result.get("status") != "success":
            log.warning(f"Stage 'comments' не выполнен успешно, пропускаем 'summary'")
            return {
                "status": "skipped",
                "reason": "comments stage failed",
                "comments_status": comments_result.get("status")
            }

        # 2. Получаем данные с комментариями из результата stage "comments"
        comments_data = comments_result.get("comments_data")

        if not comments_data:
            log.error("Данные с комментариями не найдены в результатах comments stage")
            return {
                "status": "error",
                "error": "comments_data not found in comments results"
            }

        # Конвертируем список словарей обратно в DataFrame
        df_with_comments = pd.DataFrame(comments_data)
        log.info(f"Загружено {len(df_with_comments)} строк с комментариями")

        # 3. Загружаем правила разметки
        if not isinstance(rules_file, Path):
            rules_file = Path(rules_file)

        if not rules_file.exists():
            log.error(f"Файл с правилами не найден: {rules_file}")
            return {
                "status": "error",
                "error": f"Rules file not found: {rules_file}"
            }

        log.info(f"Загрузка правил из {rules_file}")
        with open(rules_file, "r", encoding="utf-8") as f:
            rules = f.read()

        # 4. Загружаем промпты для summary (с fallback на хардкоженные)
        system_prompt = load_prompt_from_file(summary_prompt_system_file, SYSTEM_PROMPT, "system prompt")
        user_prompt_template = load_prompt_from_file(summary_prompt_user_file, USER_PROMPT_TEMPLATE, "user prompt template")

        # 5. Генерируем сводки по разметчикам
        log.info("Начало генерации персональных сводок...")
        log.info(f"Используется конфиг: {proxy_cfg_path}")

        summary_df = build_marker_summaries(
            df_with_comments,
            rules,
            proxy_cfg_path,
            system_prompt,
            user_prompt_template
        )

        if summary_df.empty:
            log.warning("Сводки не созданы (нет разметчиков с достаточным количеством ошибок)")
            return {
                "status": "skipped",
                "reason": "no markers with enough errors",
                "min_errors_threshold": MIN_ERRORS_FOR_SUMMARY
            }

        # 6. Добавляем лист "Обратная связь" в Excel отчет
        if excel_builder:
            log.info("Добавляем лист 'Обратная связь' в Excel отчет...")
            # Подготавливаем DataFrame для отображения
            feedback_display_df = summary_df.copy()
            feedback_display_df = feedback_display_df.rename(columns={
                "marker_id": "ID разметчика",
                "num_errors": "Количество ошибок",
                "personal_feedback": "Персональная обратная связь"
            })
            excel_builder.add_feedback_sheet(feedback_display_df)
            log.info("Лист 'Обратная связь' успешно добавлен")

        log.info(f"Stage 'summary' завершен успешно")

        # 7. Сохраняем обработанные assignment_id в трекер
        if project_id and state_dir:
            # Получаем список обработанных assignment_id из результата quality stage
            # (который хранится в comments_result через цепочку)
            quality_result = kwargs.get("quality_result", {})
            processed_ids = quality_result.get("processed_assignment_ids", [])

            if processed_ids:
                tracker = ProcessedAssignmentsTracker(state_dir)
                tracker.save_processed_ids(
                    project_id=project_id,
                    task_id=task_id,
                    task_name=task_name,
                    new_assignment_ids=processed_ids
                )
                log.info(f"Сохранено {len(processed_ids)} обработанных assignment_id в трекер")
            else:
                log.warning("Список обработанных assignment_id пуст, не сохраняем в трекер")

        return {
            "status": "success",
            "task_id": task_id,
            "task_name": task_name,
            "markers_with_summaries": len(summary_df),
            "excel_report_path": str(excel_builder.get_path()) if excel_builder else None
        }

    except Exception as e:
        log.error(f"Ошибка в stage 'summary': {e}", exc_info=True)
        return {
            "status": "error",
            "task_id": task_id,
            "task_name": task_name,
            "error": str(e)
        }


# =========================
# MAIN (для тестирования)
# =========================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Генерация персональных сводок для разметчиков")
    parser.add_argument("--comments-file", required=True, help="Путь к Excel с комментариями")
    parser.add_argument("--rules", required=True, help="Путь к файлу с правилами")
    parser.add_argument("--output", required=True, help="Путь для сохранения результата")

    args = parser.parse_args()

    # Загружаем данные
    df = pd.read_excel(args.comments_file)

    # Загружаем правила
    with open(args.rules, "r", encoding="utf-8") as f:
        rules = f.read()

    # Генерируем сводки
    summary_df = build_marker_summaries(df, rules, "~/configs/ml_config.cfg")

    # Сохраняем
    with pd.ExcelWriter(args.output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Comments', index=False)
        summary_df.to_excel(writer, sheet_name='Personal Summaries', index=False)

    print(f"✅ Готово! Сохранено в {args.output}")
    print(f"Сводок создано: {len(summary_df)}")

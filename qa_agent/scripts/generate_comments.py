"""
Stage "comments": Генерация LLM-комментариев для плохих assignments

Адаптирован из llm_audio_review.py для интеграции в qa_agent.

Особенности:
- Читает плохие assignments из stage "quality"
- Использует правила из rules/short_rules.md
- Генерирует объяснения ошибок через GigaChat
- Сохраняет результаты временно (НЕ отправляет в TagMe!)
"""
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

# Добавляем родительскую директорию в sys.path
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Импортируем VseLLM клиент для работы через TagMe прокси
from utils.vsellm_client import call_vsellm
from utils.excel_builder import ExcelReportBuilder


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
log = logging.getLogger("generate_comments")


# =========================
# CONFIG
# =========================

# LLM настройки через TagMe прокси
LLM_MODEL = "anthropic/claude-opus-4.8"  # Антропик Claude Opus 4.8
LLM_TIMEOUT_SEC = 180
LLM_TEMPERATURE = 0.2

# Параллелизм
MAX_CONCURRENT_REQUESTS = 10  # Максимум одновременных запросов к LLM

# Колонки
MARKS_COL = "OUTPUT:marks"
CORRECT_COL = "right_answer"
ITEM_ID_COL = "item_id"
MARKER_ID_COL = "marker_id"


# =========================
# PROMPT
# =========================

PROMPT_EXPLAIN_ERROR = """\
Ты — эксперт по контролю качества аудио-разметки данных.

Ниже приведены правила разметки.
Также указаны ответ разметчика и правильный ответ для сравнения.

Проанализируй ответ разметчика и правильный ответ, объясни кратко и конкретно, почему ответ разметчика неверен.
Ссылайся на конкретные правила из инструкции.
Твой комментарий будет видеть разметчик, поэтому:
- не обращайся к нему напрямую
- не пиши в 3 лице про разметчика
- просто кратко объясни ошибку по сути

ПРАВИЛА РАЗМЕТКИ:
{rules}

ОТВЕТ РАЗМЕТЧИКА:
{marker_answer}

ПРАВИЛЬНЫЙ ОТВЕТ:
{correct_answer}

Вот примеры комментариев пояснений:
"Пропущено негативное споттерное слово "салютом" на 11й секунде.",
"Границы споттерного слова указаны неверно, конец на 10.12."
"Неверная классификация споттерного слова. Вместо отдельного споттерного слова "салют" должно быть выделено негативное споттерное слово "салют тайм" как единое целое"
и т.д.

Важно чтобы твое объяснение соответствовало правильному ответу (а не противоречило ему).
Важные правила!!:
1. Правильный ответ всегда верен
2. споттерные слова все всегда прописывать строчными буквами ("салют" и т.д.)
3. Важно писать про неверную классификацию слова (когда не сходится с правильным ответом) и про неверные границы.
Не указывай сверх точные границы, максимум границы до сотых: 9.34 ,  9.80 и т.д. или 9 секунда. Но учитыввай что разрешается погрешность в границах + - 0.15 сек!
То есть: Ответ разметчика например: start: 9.143059362963763, finish: 9.580336115105508, правильные границы: start: 9.057846198365882, finish: 9.54838264094067
И в начале и в конце разница между границами разметчика и правильного ответа меньше чем 0.15, значит это не ошибка, про границы не пиши, пишу про другую ошибку. А вот если уже разница больше чем 0.15 - тогда можно упомянуть границы.
Нельзя писать про разницу в 0.06 секунд и т.д. - это слишком маленькая разница, это не ошибка!! Упомяни что просто, что пропущенно спотерное слово или выделено не то.
4. "salut" – это "салют", а не "салут". Про споттерные слова в своем пояснении лучше пиши на кириллице: не "joy", а "джой" и т.д., не смотря на то что в от правильном ответе "joy".
5. Не делай лишних догадок, если правильный ответ: [], достаточно просто ответить: "На записи нет споттерных слов".
6. "афина" не является негативным споттерным словом!
"""


# =========================
# HELPERS
# =========================

def load_prompt_from_file(prompt_file_path: Optional[Path]) -> str:
    """
    Загружает промпт из файла, если путь указан.
    Если файл не найден или путь не указан - возвращает хардкоженный промпт.

    Args:
        prompt_file_path: Путь к файлу с промптом (опционально, может быть str или Path)

    Returns:
        Текст промпта
    """
    # Конвертируем в Path если пришла строка
    if prompt_file_path and not isinstance(prompt_file_path, Path):
        prompt_file_path = Path(prompt_file_path)

    if prompt_file_path and prompt_file_path.exists():
        log.info(f"Загружаем промпт из файла: {prompt_file_path}")
        try:
            with open(prompt_file_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            if prompt:
                return prompt
            else:
                log.warning(f"Файл промпта пустой: {prompt_file_path}, используем дефолтный")
        except Exception as e:
            log.warning(f"Ошибка чтения промпта из {prompt_file_path}: {e}, используем дефолтный")
    else:
        if prompt_file_path:
            log.warning(f"Файл промпта не найден: {prompt_file_path}, используем дефолтный")
        else:
            log.info("Путь к промпту не указан, используем дефолтный хардкоженный промпт")

    # Fallback на хардкоженный промпт
    return PROMPT_EXPLAIN_ERROR


def parse_marks(value: Any) -> Any:
    """Парсит метки из различных форматов."""
    # Проверяем None явно
    if value is None:
        return None

    # Если это список - возвращаем как есть
    if isinstance(value, list):
        return value

    # Если это словарь - возвращаем как есть
    if isinstance(value, dict):
        return value

    # Проверяем NaN только для скалярных значений
    if isinstance(value, (int, float)) and pd.isna(value):
        return None

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            return s

    return value


def marks_to_pretty_text(value: Any) -> str:
    """Конвертирует метки в читаемый текст."""
    parsed = parse_marks(value)

    if parsed is None:
        return "пусто"

    if parsed == []:
        return "[]"

    if isinstance(parsed, (dict, list)):
        return json.dumps(parsed, ensure_ascii=False, indent=2)

    return str(parsed)


# =========================
# LLM CALL через TagMe прокси
# =========================

def explain_one_error_with_llm(
    rules: str,
    marker_answer: Any,
    correct_answer: Any,
    proxy_cfg_path: str,
    prompt_template: str = None,
) -> Tuple[str, Dict[str, int]]:
    """
    Генерирует объяснение ошибки через LLM (GigaChat) используя TagMe прокси.

    Args:
        rules: Правила разметки
        marker_answer: Ответ разметчика
        correct_answer: Правильный ответ
        proxy_cfg_path: Путь к конфигу TagMe для получения токена
        prompt_template: Шаблон промпта (если None, использует PROMPT_EXPLAIN_ERROR)

    Returns:
        Кортеж (объяснение, usage статистика)
    """
    if prompt_template is None:
        prompt_template = PROMPT_EXPLAIN_ERROR

    marker_text = marks_to_pretty_text(marker_answer)
    correct_text = marks_to_pretty_text(correct_answer)

    prompt = prompt_template.format(
        rules=rules,
        marker_answer=marker_text,
        correct_answer=correct_text,
    )

    log.debug("Prompt to LLM:\n%s", prompt[:500])

    try:
        # Вызываем LLM через TagMe прокси
        result = call_vsellm(
            model=LLM_MODEL,
            user_text=prompt,
            temperature=LLM_TEMPERATURE,
            timeout_sec=LLM_TIMEOUT_SEC,
            proxy_cfg_path=proxy_cfg_path,
        )

        explanation = result.get("answer", "") or result.get("raw_text", "")
        usage = result.get("usage", {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        })

        return explanation, usage

    except Exception as e:
        log.error(f"Ошибка при вызове LLM: {e}")
        raise


# =========================
# MAIN PIPELINE с параллелизмом
# =========================

async def process_single_assignment(
    idx: int,
    row: pd.Series,
    rules: str,
    proxy_cfg_path: str,
    prompt_template: str,
    semaphore: asyncio.Semaphore,
    progress: Dict[str, int]
) -> Dict[str, Any]:
    """
    Обрабатывает один assignment асинхронно.

    Args:
        idx: Индекс строки в DataFrame
        row: Строка DataFrame
        rules: Правила разметки
        proxy_cfg_path: Путь к конфигу
        prompt_template: Шаблон промпта
        semaphore: Семафор для ограничения параллелизма
        progress: Словарь для отслеживания прогресса

    Returns:
        Словарь с результатами обработки
    """
    async with semaphore:
        progress["current"] += 1
        current = progress["current"]
        total = progress["total"]

        log.info(
            f"Обработка {current}/{total} | item_id={row.get(ITEM_ID_COL)} | marker_id={row.get(MARKER_ID_COL)}"
        )

        try:
            # call_vsellm - синхронная функция, запускаем в executor
            loop = asyncio.get_event_loop()
            explanation, usage = await loop.run_in_executor(
                None,
                explain_one_error_with_llm,
                rules,
                row.get("parsed_marks"),
                row.get(CORRECT_COL),
                proxy_cfg_path,
                prompt_template
            )

            log.info(f"Готово {current}/{total} | комментарий: {explanation[:100]}...")

            return {
                "idx": idx,
                "llm_comment": explanation,
                "llm_model": LLM_MODEL,
                "llm_prompt_tokens": usage.get("prompt_tokens", 0),
                "llm_completion_tokens": usage.get("completion_tokens", 0),
                "llm_total_tokens": usage.get("total_tokens", 0),
                "error": None
            }

        except Exception as e:
            log.exception(f"Ошибка при обработке idx={idx}")
            return {
                "idx": idx,
                "llm_comment": f"ERROR: {e}",
                "llm_model": LLM_MODEL,
                "llm_prompt_tokens": 0,
                "llm_completion_tokens": 0,
                "llm_total_tokens": 0,
                "error": str(e)
            }


async def generate_comments_for_bad_assignments_async(
    df: pd.DataFrame,
    rules: str,
    proxy_cfg_path: str,
    prompt_template: str = None,
    max_concurrent: int = MAX_CONCURRENT_REQUESTS
) -> pd.DataFrame:
    """
    Генерирует комментарии для всех плохих assignments параллельно.

    Args:
        df: DataFrame с плохими assignments
        rules: Правила разметки
        proxy_cfg_path: Путь к конфигу TagMe
        prompt_template: Шаблон промпта (опционально)
        max_concurrent: Максимум одновременных запросов

    Returns:
        DataFrame с добавленными комментариями
    """
    df = df.copy()

    # Добавляем колонки для результатов
    for col in [
        "llm_comment",
        "llm_model",
        "llm_prompt_tokens",
        "llm_completion_tokens",
        "llm_total_tokens"
    ]:
        if col not in df.columns:
            df[col] = 0 if "tokens" in col else ""

    total_rows = len(df)
    log.info(f"Генерация комментариев для {total_rows} плохих assignments (параллельно, max_concurrent={max_concurrent})...")

    # Семафор для ограничения параллелизма
    semaphore = asyncio.Semaphore(max_concurrent)

    # Прогресс
    progress = {"current": 0, "total": total_rows}

    # Создаем задачи для всех строк
    tasks = [
        process_single_assignment(
            idx=idx,
            row=row,
            rules=rules,
            proxy_cfg_path=proxy_cfg_path,
            prompt_template=prompt_template,
            semaphore=semaphore,
            progress=progress
        )
        for idx, row in df.iterrows()
    ]

    # Запускаем все задачи параллельно
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Записываем результаты обратно в DataFrame
    for result in results:
        if isinstance(result, Exception):
            log.error(f"Неожиданная ошибка: {result}")
            continue

        if isinstance(result, dict) and "idx" in result:
            idx = result["idx"]
            df.at[idx, "llm_comment"] = result["llm_comment"]
            df.at[idx, "llm_model"] = result["llm_model"]
            df.at[idx, "llm_prompt_tokens"] = result["llm_prompt_tokens"]
            df.at[idx, "llm_completion_tokens"] = result["llm_completion_tokens"]
            df.at[idx, "llm_total_tokens"] = result["llm_total_tokens"]

    return df


def generate_comments_for_bad_assignments(
    df: pd.DataFrame,
    rules: str,
    proxy_cfg_path: str,
    prompt_template: str = None,
) -> pd.DataFrame:
    """
    Обертка для запуска асинхронной генерации комментариев.

    Args:
        df: DataFrame с плохими assignments
        rules: Правила разметки
        proxy_cfg_path: Путь к конфигу TagMe
        prompt_template: Шаблон промпта (опционально)

    Returns:
        DataFrame с добавленными комментариями
    """
    # Запускаем асинхронную версию
    return asyncio.run(
        generate_comments_for_bad_assignments_async(df, rules, proxy_cfg_path, prompt_template)
    )


# =========================
# STAGE RUNNER
# =========================

async def run(
    task_id: str,
    task_name: str,
    run_date: str,
    data_dir: Path,
    quality_result: Dict[str, Any],
    rules_file: Path,
    proxy_cfg_path: Optional[str] = None,
    comment_prompt_file: Optional[Path] = None,
    excel_builder: Optional[ExcelReportBuilder] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Точка входа для stage "comments" из qa_manager.

    Args:
        task_id: ID задачи
        task_name: Название задачи
        run_date: Дата запуска
        data_dir: Директория с данными
        quality_result: Результат stage "quality"
        rules_file: Путь к файлу с правилами
        proxy_cfg_path: Путь к конфигу TagMe для получения токена прокси
        comment_prompt_file: Путь к файлу с промптом для генерации комментариев
        excel_builder: Builder для добавления листа "Комментарии" в единый Excel отчет
        **kwargs: Дополнительные параметры

    Returns:
        Результат выполнения stage (с данными для Excel)
    """
    log.info(f"Stage 'comments' для задачи {task_id} ({task_name})")

    # Используем дефолтный конфиг если не передан
    if not proxy_cfg_path:
        proxy_cfg_path = str(Path.home() / "configs" / "ml_config.cfg")

    try:
        # 1. Проверяем что stage "quality" выполнен успешно
        if quality_result.get("status") != "success":
            log.warning(f"Stage 'quality' не выполнен успешно, пропускаем 'comments'")
            return {
                "status": "skipped",
                "reason": "quality stage failed",
                "quality_status": quality_result.get("status")
            }

        # 2. Проверяем что есть плохие assignments
        bad_count = quality_result.get("bad_assignments_count", 0)
        if bad_count == 0:
            log.info("Плохих assignments нет, пропускаем генерацию комментариев")
            return {
                "status": "skipped",
                "reason": "no bad assignments",
                "bad_assignments_count": 0
            }

        # 3. Получаем данные с плохими assignments из excel_data quality_result
        excel_data = quality_result.get("excel_data", {})
        bad_assignments_data = excel_data.get("bad_assignments_data")

        if not bad_assignments_data:
            log.error("Данные с плохими assignments не найдены в результатах quality stage")
            return {
                "status": "error",
                "error": "bad_assignments_data not found in quality results"
            }

        df = pd.DataFrame(bad_assignments_data)
        log.info(f"Загружено {len(df)} плохих assignments из результата quality stage")

        # 4. Загружаем правила разметки
        # Конвертируем в Path если пришла строка
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

        # 5. Загружаем промпт для комментариев (с fallback на хардкоженный)
        prompt_template = load_prompt_from_file(comment_prompt_file)

        # 6. Генерируем комментарии
        log.info(f"Начало генерации комментариев через LLM (прокси TagMe)...")
        log.info(f"Используется конфиг: {proxy_cfg_path}")
        df_with_comments = generate_comments_for_bad_assignments(df, rules, proxy_cfg_path, prompt_template)

        # 7. Добавляем лист "Комментарии" в Excel отчет
        if excel_builder:
            log.info("Добавляем лист 'Комментарии' в Excel отчет...")
            # Подготавливаем DataFrame для отображения с дополнительными колонками
            comments_display_df = df_with_comments[[
                "assignment_id", ITEM_ID_COL, MARKER_ID_COL,
                "parsed_marks", "right_answer", "score", "llm_comment"
            ]].copy()
            comments_display_df = comments_display_df.rename(columns={
                "assignment_id": "Assignment ID",
                ITEM_ID_COL: "ID задания",
                MARKER_ID_COL: "ID разметчика",
                "parsed_marks": "Ответ разметчика",
                "right_answer": "Правильный ответ",
                "score": "Оценка",
                "llm_comment": "Комментарий"
            })
            excel_builder.add_comments_sheet(comments_display_df)
            log.info("Лист 'Комментарии' успешно добавлен")

        # Статистика токенов
        total_prompt_tokens = df_with_comments["llm_prompt_tokens"].sum()
        total_completion_tokens = df_with_comments["llm_completion_tokens"].sum()
        total_tokens = df_with_comments["llm_total_tokens"].sum()

        log.info(f"Stage 'comments' завершен успешно")
        log.info(f"Токены: prompt={total_prompt_tokens}, completion={total_completion_tokens}, total={total_tokens}")

        return {
            "status": "success",
            "task_id": task_id,
            "task_name": task_name,
            "bad_assignments_count": len(df),
            "comments_generated": len(df_with_comments[df_with_comments["llm_comment"].str.len() > 0]),
            "total_prompt_tokens": int(total_prompt_tokens),
            "total_completion_tokens": int(total_completion_tokens),
            "total_tokens": int(total_tokens),
            "comments_data": df_with_comments.to_dict('records')  # Для stage 3 (summary), конвертируем в список
        }

    except Exception as e:
        log.error(f"Ошибка в stage 'comments': {e}", exc_info=True)
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

    parser = argparse.ArgumentParser(description="Генерация комментариев для плохих assignments")
    parser.add_argument("--bad-assignments", required=True, help="Путь к JSON/Excel с плохими assignments")
    parser.add_argument("--rules", required=True, help="Путь к файлу с правилами")
    parser.add_argument("--output", required=True, help="Путь для сохранения результата")

    args = parser.parse_args()

    # Загружаем данные
    bad_file = Path(args.bad_assignments)
    if bad_file.suffix == ".json":
        with open(bad_file, "r", encoding="utf-8") as f:
            records = json.load(f)
        df = pd.DataFrame(records)
    else:
        df = pd.read_excel(bad_file)

    # Загружаем правила
    with open(args.rules, "r", encoding="utf-8") as f:
        rules = f.read()

    # Генерируем комментарии
    df_result = generate_comments_for_bad_assignments(df, rules)

    # Сохраняем
    output_path = Path(args.output)
    if output_path.suffix == ".json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(df_result.to_dict('records'), f, ensure_ascii=False, indent=2, default=str)
    else:
        df_result.to_excel(output_path, index=False)

    print(f"✅ Готово! Сохранено в {output_path}")
    print(f"Комментариев: {len(df_result[df_result['llm_comment'].str.len() > 0])}")

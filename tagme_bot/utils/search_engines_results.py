import json
import ast
import re
from pathlib import Path

import pandas as pd

# =========================
# НАСТРОЙКИ
# =========================

INPUT_PATH = "secqa_results.xlsx"  # сюда укажи путь к своему файлу
OUTPUT_PATH = "secqa_results_processed.xlsx"

SEARCH_ENGINES_COL = "OUTPUT:search_engines"

# Как сохранять contexts в ячейке:
# "json"  -> ["context1", "context2", ...]
# "text"  -> context1\n\n---\n\ncontext2
CONTEXTS_FORMAT = "json"


# =========================
# ЧТЕНИЕ ФАЙЛА
# =========================

def read_table(path: str | Path) -> pd.DataFrame:
    """
    Читает файл и возвращает DataFrame.
    Поддерживает: .xlsx, .xls, .csv, .tsv, .txt, .json
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    if suffix in [".tsv", ".txt"]:
        # Сначала пробуем как JSON
        try:
            return read_json_file(path)
        except Exception:
            # Если не JSON, то читаем как TSV
            return pd.read_csv(path, sep="\t")

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix == ".json":
        return read_json_file(path)

    raise ValueError(f"Неизвестный формат файла: {suffix}")


def read_json_file(path: str | Path) -> pd.DataFrame:
    """
    Читает JSON файл и преобразует в DataFrame.
    Поддерживает два формата:
    1. Массив объектов с полем result.search_engines
    2. Обычный массив объектов
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON должен содержать массив объектов")

    # Проверяем, есть ли вложенная структура с result
    if data and isinstance(data[0], dict) and 'result' in data[0]:
        # Извлекаем данные из result и добавляем метаданные
        rows = []
        for item in data:
            row = {}

            # Копируем метаданные верхнего уровня
            for key, value in item.items():
                if key != 'result':
                    row[key] = value

            # Копируем данные из result
            if 'result' in item and isinstance(item['result'], dict):
                for key, value in item['result'].items():
                    # Используем префикс OUTPUT: для совместимости
                    if key == 'search_engines':
                        row['OUTPUT:search_engines'] = value
                    else:
                        row[f'OUTPUT:{key}'] = value

            rows.append(row)

        return pd.DataFrame(rows)
    else:
        # Обычный массив объектов
        return pd.DataFrame(data)


# =========================
# ПАРСИНГ search_engines
# =========================

def parse_search_engines(value):
    """
    Превращает значение из ячейки OUTPUT:search_engines в Python list[dict].
    Поддерживает:
    - нормальный JSON
    - строку с Python-литералом
    - уже готовый list
    - пустые значения
    """
    # Проверка на None/NaN для скалярных значений
    if value is None:
        return []

    # Для pandas Series или других итерируемых объектов
    if isinstance(value, (list, tuple)):
        # Если это уже список, проверяем его содержимое
        if not value:
            return []
        # Если первый элемент - словарь, значит это уже готовый список поисковиков
        if isinstance(value[0], dict):
            return value
        return []

    # Проверка на pandas NaN
    try:
        if pd.isna(value):
            return []
    except (ValueError, TypeError):
        # Если pd.isna() не может обработать значение, продолжаем
        pass

    if isinstance(value, dict):
        return [value]

    if not isinstance(value, str):
        return []

    value = value.strip()

    if not value:
        return []

    # 1. Пробуем как JSON
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
        return []
    except Exception:
        pass

    # 2. Пробуем как Python literal, если вдруг кавычки одинарные
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
        return []
    except Exception:
        return []


def clean_xml_invalid_chars(text):
    """
    Удаляет недопустимые XML символы из текста.
    Это нужно для корректного сохранения в Excel.
    """
    if not isinstance(text, str):
        return text

    # Удаляем недопустимые XML символы
    # Допустимые диапазоны: #x9 | #xA | #xD | [#x20-#xD7FF] | [#xE000-#xFFFD] | [#x10000-#x10FFFF]
    def is_valid_xml_char(char):
        codepoint = ord(char)
        return (
                codepoint == 0x9 or
                codepoint == 0xA or
                codepoint == 0xD or
                (0x20 <= codepoint <= 0xD7FF) or
                (0xE000 <= codepoint <= 0xFFFD) or
                (0x10000 <= codepoint <= 0x10FFFF)
        )

    return ''.join(char for char in text if is_valid_xml_char(char))


def normalize_contexts(contexts):
    """
    Нормализует contexts к списку строк.
    """
    if contexts is None:
        return []

    if isinstance(contexts, list):
        # Очищаем каждый контекст от недопустимых XML символов
        return [clean_xml_invalid_chars(str(x)) for x in contexts if x is not None]

    if isinstance(contexts, str):
        return [clean_xml_invalid_chars(contexts)]

    return [clean_xml_invalid_chars(str(contexts))]


def format_contexts(contexts, mode="json"):
    """
    Готовит contexts для записи в Excel.
    """
    contexts = normalize_contexts(contexts)

    if mode == "json":
        return json.dumps(contexts, ensure_ascii=False)

    if mode == "text":
        return "\n\n---\n\n".join(contexts)

    raise ValueError("CONTEXTS_FORMAT должен быть 'json' или 'text'")


# =========================
# ОСНОВНАЯ ОБРАБОТКА
# =========================

def split_search_engines(
        input_path: str | Path,
        output_path: str | Path,
        search_engines_col: str = SEARCH_ENGINES_COL,
        contexts_format: str = CONTEXTS_FORMAT,
) -> pd.DataFrame:
    df = read_table(input_path)

    if search_engines_col not in df.columns:
        raise ValueError(
            f"В файле нет колонки {search_engines_col!r}. "
            f"Доступные колонки: {list(df.columns)}"
        )

    # Собираем все возможные имена поисковиков из всего файла
    all_engine_names = []

    parsed_rows = []

    for value in df[search_engines_col]:
        engines = parse_search_engines(value)
        parsed_rows.append(engines)

        for engine in engines:
            if not isinstance(engine, dict):
                continue

            name = engine.get("name")
            if name and name not in all_engine_names:
                all_engine_names.append(name)

    # Создаем новые колонки
    for engine_name in all_engine_names:
        df[engine_name] = ""

    # Заполняем новые колонки contexts
    for row_idx, engines in enumerate(parsed_rows):
        per_engine_contexts = {}

        for engine in engines:
            if not isinstance(engine, dict):
                continue

            name = engine.get("name")
            contexts = normalize_contexts(engine.get("contexts", []))

            if not name:
                continue

            # Если один и тот же поисковик встретился несколько раз в строке,
            # объединяем contexts
            per_engine_contexts.setdefault(name, [])
            per_engine_contexts[name].extend(contexts)

        for engine_name, contexts in per_engine_contexts.items():
            df.at[row_idx, engine_name] = format_contexts(
                contexts,
                mode=contexts_format,
            )

    # Сохраняем в Excel
    output_path = Path(output_path)
    df.to_excel(output_path, index=False)

    return df


# =========================
# ЗАПУСК
# =========================

if __name__ == "__main__":
    result_df = split_search_engines(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        search_engines_col=SEARCH_ENGINES_COL,
        contexts_format=CONTEXTS_FORMAT,
    )

    print(f"Готово! Файл сохранен: {OUTPUT_PATH}")
    print("Созданные столбцы поисковиков:")
    engine_columns = [col for col in result_df.columns if col not in ['OUTPUT:search_engines']]
    print(engine_columns)

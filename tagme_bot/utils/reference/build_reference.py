"""Подготовка справочника для RAG-инструмента.

Читает csv/xlsx с колонками `description` (текст) и `emb` (текст для эмбеддинга),
считает эмбеддинги и пишет рядом с входным файлом `<имя входного файла>.json`:
    [{"description": "...", "<model>": [floats]}, ...]
Этот JSON загружается в TagMe вручную как dataset-файл.

Запуск из tagme_bot/:
    python utils/reference/build_reference.py PATH [--service direct] [--model MODEL]
Код возврата 0 только если все эмбеддинги посчитаны.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from utils.vsellm_client import (
    TAGME_DIRECT_URL,
    TAGME_EMBEDDINGS_PROXY_URL,
    TAGME_IFT_PROXY_CFG_PATH,
    TAGME_IFT_URL,
    call_vsellm_embeddings,
)

DEFAULT_MODEL = "Giga-Embeddings-instruct-10B-A1.8B-0826"
DEFAULT_SERVICE = "direct"
TEXT_COL = "description"
EMB_COL = "emb"
PROGRESS_EVERY = 10
RETRIES = 3
REQUEST_TIMEOUT = 180.0


def _route(service: str) -> tuple[str, str | None]:
    """Возвращает (url эмбеддингов, путь к cfg или None для дефолтного)."""
    s = service.strip().lower()
    if s == "direct":
        return TAGME_DIRECT_URL.replace("/chat/completions", "/embeddings"), None
    if s == "ift":
        return TAGME_IFT_URL.replace("/chat/completions", "/embeddings"), TAGME_IFT_PROXY_CFG_PATH
    if s == "vsellm":
        return TAGME_EMBEDDINGS_PROXY_URL, None
    raise ValueError(f"Неизвестный service: {service!r} (direct | ift | vsellm)")


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        return pd.read_excel(path, dtype=str)
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str)
    raise ValueError(f"Поддерживаются только .csv и .xlsx, получено: {suffix!r}")


def _clean_column(df: pd.DataFrame, col: str) -> list[str]:
    if col not in df.columns:
        raise ValueError(f"В файле нет колонки {col!r}. Есть: {list(df.columns)}")
    values = df[col].tolist()
    bad = [i + 2 for i, v in enumerate(values) if not isinstance(v, str) or not v.strip()]
    if bad:
        raise ValueError(f"Пустые значения в колонке {col!r}, строки файла: {bad[:20]}")
    return [v.strip() for v in values]


def _embed_one(text: str, model: str, url: str, cfg_path: str | None) -> tuple[list[float], int]:
    """Эмбеддинг одного текста с ретраями. Возвращает (вектор, токены)."""
    last_err: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            embs, usage, _ = call_vsellm_embeddings(
                model=model,
                inputs=[text],
                request_timeout=REQUEST_TIMEOUT,
                proxy_cfg_path=cfg_path,
                proxy_url=url,
            )
            if len(embs) != 1 or not embs[0]:
                raise RuntimeError(f"Некорректный ответ: получено {len(embs)} векторов")
            tokens = int(usage.get("total_tokens") or usage.get("prompt_tokens") or 0)
            return [float(x) for x in embs[0]], tokens
        except Exception as e:
            last_err = e
            print(f"  попытка {attempt}/{RETRIES} не удалась: {e}", file=sys.stderr)
            if attempt < RETRIES:
                time.sleep(0.5 * attempt)
    raise RuntimeError(f"Не удалось получить эмбеддинг после {RETRIES} попыток") from last_err


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="Путь к csv/xlsx со справочником")
    ap.add_argument("--service", default=DEFAULT_SERVICE, help=f"Стенд эмбеддинга (по умолчанию {DEFAULT_SERVICE})")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Модель эмбеддера (по умолчанию {DEFAULT_MODEL})")
    args = ap.parse_args()

    src = Path(args.path)
    url, cfg_path = _route(args.service)

    df = _read_table(src)
    descriptions = _clean_column(df, TEXT_COL)
    emb_texts = _clean_column(df, EMB_COL)
    total = len(emb_texts)
    print(f"Файл: {src.name} | записей: {total} | service={args.service} | model={args.model}")

    records: list[dict] = []
    total_tokens = 0
    dim: int | None = None
    for i, (desc, text) in enumerate(zip(descriptions, emb_texts), start=1):
        vec, tokens = _embed_one(text, args.model, url, cfg_path)
        if dim is None:
            dim = len(vec)
        elif len(vec) != dim:
            raise RuntimeError(f"Строка {i}: размерность {len(vec)} != {dim}")
        total_tokens += tokens
        records.append({TEXT_COL: desc, args.model: vec})
        if i % PROGRESS_EVERY == 0:
            print(f"Прогресс: {i}/{total}")

    out = src.resolve().with_suffix(".json")
    out.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    print(f"Готово: {total}/{total} | размерность={dim} | токенов суммарно={total_tokens}")
    print(f"Результат: {out}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        sys.exit(1)

"""Recall поиска rag_tool по выгрузке результатов: попал ли GOLDEN в `reference` из final_user_prompt."""
import argparse
import json
import sys
from typing import Any, List

import pandas as pd



def _read(path: str) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path, dtype=str)
    return pd.read_csv(path, dtype=str)


def _ref_values(prompt: Any, key: str) -> List[str]:
    """Значения поля `key` записей reference в порядке релевантности."""
    refs = json.loads(prompt)["reference"]
    res = []
    for r in refs:
        rec = json.loads(r) if isinstance(r, str) else r
        res.append(str(rec[key]).strip())
    return res


def _ks(top_n: int) -> List[int]:
    """k для recall: 1, 3, 5, 10, 20, ... в пределах top_n; последним всегда top_n."""
    ks = [k for k in (1, 3, 5) if k < top_n]
    ks += list(range(10, top_n, 10))
    return ks + [top_n]


def _pct(a: int, b: int) -> str:
    return f"{a}/{b} = {a / b:.2f}" if b else "-"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", help="csv или xlsx с результатами разметки")
    ap.add_argument("--ref-field", required=True, help="поле записи reference, по которому ищем эталон")
    ap.add_argument("--gold-col", required=True, help="имя колонки эталона без префикса GOLDEN:")
    ap.add_argument("--prompt-col", default="OUTPUT:final_user_prompt", help="колонка с final_user_prompt")
    ap.add_argument("--out-col", help="колонка с ответом бота (по умолчанию OUTPUT:<gold_col>)")
    args = ap.parse_args()

    try:
        df = _read(args.path)
        gold_col = f"GOLDEN:{args.gold_col}"
        for c in (gold_col, args.prompt_col):
            if c not in df.columns:
                raise ValueError(f"нет колонки {c!r}")
        ref_key = args.ref_field
        out_col = args.out_col or f"OUTPUT:{args.gold_col}"
        has_out = out_col in df.columns
    except Exception as e:
        print(f"ОШИБКА: {type(e).__name__}: {e}")
        return 1

    ranks, correct, skipped, ref_lens = [], [], [], []
    for i, row in df.iterrows():
        gold, prompt = row[gold_col], row[args.prompt_col]
        if pd.isna(gold) or pd.isna(prompt) or not str(gold).strip():
            skipped.append(i)
            continue
        try:
            vals = _ref_values(prompt, ref_key)
        except Exception:
            skipped.append(i)
            continue
        gold = str(gold).strip()
        ref_lens.append(len(vals))
        ranks.append(vals.index(gold) + 1 if gold in vals else None)
        if has_out:
            out = row[out_col]
            correct.append(not pd.isna(out) and str(out).strip() == gold)

    n = len(ranks)
    if not n:
        print("ОШИБКА: нет ни одной валидной строки")
        return 1
    hit = [r for r in ranks if r]

    top_n = max(ref_lens)
    rows = []
    for k in _ks(top_n):
        label = f"recall@{k} (top_n)" if k == top_n else f"recall@{k}"
        rows.append((label, _pct(sum(r <= k for r in hit), n)))
    if has_out:
        rows.append(("accuracy бота", _pct(sum(correct), n)))
        c_in = [c for c, r in zip(correct, ranks) if r]
        c_out = [c for c, r in zip(correct, ranks) if not r]
        rows.append(("accuracy, GOLDEN в reference", _pct(sum(c_in), len(c_in))))
        rows.append(("accuracy, GOLDEN не в reference", _pct(sum(c_out), len(c_out))))

    w = max(len(r[0]) for r in rows)
    if skipped:
        print(f"пропущено строк: {len(skipped)}")
    print(f"| {'Метрика':<{w}} | Значение |")
    print(f"|{'-' * (w + 2)}|----------|")
    for name, val in rows:
        print(f"| {name:<{w}} | {val} |")
    print("ранги GOLDEN:", sorted(set(hit)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Проверки качества данных D-01…D-09 (ПР4, пункт 9).

Требования ко входным данным из ПР2 реализованы функциями-проверками в
bank_marketing.validation. Скрипт:

1. Готовит намеренно испорченные файлы из 400 случайных записей порции 1:
   bad_soft.csv — нарушения, которые исправляются или дают предупреждение
   (D-03…D-09), и два файла с критическими нарушениями: bad_schema.csv
   (нет столбца, D-01) и bad_period.csv (период раньше обработанного, D-02).
2. Проверяет их и сравнивает найденное с внесённым.
3. Проверяет порции 1–4 по порядку поступления (D-02 — относительно
   последнего обработанного периода).

Результаты: reports/quality_demo/*.csv, report_assets/pr4/quality_out.txt,
листинг report_assets/pr4/code/листинг_4_1_quality.py.
"""

import inspect

import numpy as np
import pandas as pd

from bank_marketing import validation as V
from bank_marketing.config import ASSETS_DIR, CSV_SEP, REPORTS_DIR
from bank_marketing.data import batch_files
from bank_marketing.features import load_reference
from bank_marketing.report import emit

DEMO_DIR = REPORTS_DIR / "quality_demo"
PR4_DIR = ASSETS_DIR / "pr4"
SEED = 23
STATUS = {V.OK: "ok", V.FIXED: "исправлено", V.WARNING: "предупр.", V.CRITICAL: "критично"}


def read(path) -> pd.DataFrame:
    return pd.read_csv(path, sep=CSV_SEP, dtype=str)


def write(df: pd.DataFrame, name: str):
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    path = DEMO_DIR / name
    df.to_csv(path, sep=CSV_SEP, index=False)
    return path


def make_soft(src: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Испорченная порция: известное число нарушений каждого требования."""
    rng = np.random.default_rng(SEED)
    df = src.sample(400, random_state=SEED).sort_index().reset_index(drop=True)
    # каждая запись портится не больше одного раза; D-07 — только где previous = 0
    clean_rows = rng.permutation(np.flatnonzero(df["previous"] == "0"))
    r_age, r_camp, r_job, r_cont, r_pout = np.split(clean_rows[:21], np.cumsum([4, 3, 6, 3]))
    injected = {}
    df.loc[r_age, "age"] = "-3"
    df.loc[r_camp, "campaign"] = "0"
    injected["D-04"] = f"{len(r_age)} записей с age = −3, {len(r_camp)} с campaign = 0"
    df.loc[r_job, "job"] = "astronaut"
    df.loc[r_cont, "contact"] = "pager"
    injected["D-05"] = f"{len(r_job)} записей job = astronaut, {len(r_cont)} contact = pager"
    df.loc[r_pout, "poutcome"] = "failure"
    injected["D-07"] = f"{len(r_pout)} записей poutcome = failure при previous = 0"
    edu = rng.random(len(df)) < 0.3
    df.loc[edu, "education"] = "unknown"
    injected["D-06"] = f"education = unknown у {int(edu.sum())} записей (30 %)"
    old = rng.random(len(df)) < 0.4
    old[r_age] = False
    df.loc[old, "age"] = (df.loc[old, "age"].astype(int) + 20).astype(str)
    injected["D-09"] = f"возраст увеличен на 20 лет у {int(old.sum())} записей (40 %)"
    dup = df.head(5).copy()
    dup["row_id"] = (df["row_id"].astype(int).max() + 1 + np.arange(5)).astype(str)
    df = pd.concat([df, dup], ignore_index=True)
    injected["D-03"] = "5 полных копий записей с новыми row_id"
    injected["D-08"] = "макропоказатели порции 1 (ноябрь 2008 г.) оставлены как есть"
    return df, injected


def report(results) -> list[str]:
    out = []
    for r in results:
        share = f"{r.violations} из {r.base} {r.unit}"
        out.append(f"    {r.rule}: {STATUS[r.status]:<11} {share:<22} {r.details}"[:200])
    out.append(f"    вердикт: {V.verdict(results)}")
    return out


def write_listing() -> None:
    funcs = [V.check_ranges, V.check_macro_range, V.check_drift]
    body = "\n\n".join(inspect.getsource(f).rstrip() for f in funcs)
    (PR4_DIR / "code").mkdir(parents=True, exist_ok=True)
    (PR4_DIR / "code" / "листинг_4_1_quality.py").write_text(body + "\n", encoding="utf-8")


def main() -> None:
    ref = load_reference()
    first = batch_files()[0]
    src = read(first)
    out = [
        "Проверки качества данных D-01…D-09",
        "",
        f"Справочник: исходная часть, последний период {ref['last_period']}",
        "",
    ]

    soft, injected = make_soft(src)
    path = write(soft, "bad_soft.csv")
    out.append(f"1. {path.name}: {len(soft)} записей (из {first.name}); внесено:")
    out += [f"    {k}: {v}" for k, v in sorted(injected.items())]
    df, res = V.validate(read(path), ref, last_period=ref["last_period"])
    out.append("  результат проверки:")
    out += report(res)
    out.append(f"    после очистки: {len(df)} записей")

    schema = src.head(400).drop(columns=["contact"])
    path = write(schema, "bad_schema.csv")
    _, res = V.validate(read(path), ref, last_period=ref["last_period"])
    out += ["", f"2. {path.name}: удалён столбец contact"]
    out += report(res)

    period = src.head(400).copy()
    period["period"] = "2008-07"
    path = write(period, "bad_period.csv")
    _, res = V.validate(read(path), ref, last_period=ref["last_period"])
    out += ["", f"3. {path.name}: период 2008-07 — раньше последнего обработанного"]
    out += report(res)

    out += ["", "4. Порции в порядке поступления"]
    last = ref["last_period"]
    for p in batch_files():
        raw = read(p)
        df, res = V.validate(raw, ref, last_period=last)
        out.append(
            f"  {p.stem}: {len(raw)} → {len(df) if df is not None else 0} записей, "
            f"периоды {raw['period'].min()}…{raw['period'].max()}"
        )
        out += report(res)
        if df is not None:
            last = str(df["period"].max())
    write_listing()
    emit(out, PR4_DIR / "quality_out.txt")


if __name__ == "__main__":
    main()

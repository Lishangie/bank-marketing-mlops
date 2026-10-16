"""Восстановление года-месяца звонка по порядку строк."""

import numpy as np
import pandas as pd

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
FIRST_PERIOD, LAST_PERIOD = "2008-05", "2010-11"


def restore_periods(month: pd.Series, start_year: int = 2008) -> pd.Series:
    """Год-месяц `YYYY-MM` для каждой строки.

    Данные упорядочены по времени с мая 2008 г.; год растёт, когда номер
    месяца уменьшается относительно предыдущей строки.
    """
    m = month.map({name: i + 1 for i, name in enumerate(MONTHS)}).to_numpy()
    year = start_year + np.concatenate([[0], np.cumsum(np.diff(m) < 0)])
    return pd.Series([f"{y}-{mm:02d}" for y, mm in zip(year, m, strict=True)], index=month.index)


def periods_ok(period: pd.Series) -> tuple[bool, int]:
    """Каждый период — один непрерывный блок, порядок возрастающий, границы 2008-05…2010-11."""
    runs = int((period != period.shift()).sum())
    ok = (
        runs == period.nunique()
        and period.is_monotonic_increasing
        and period.iloc[0] == FIRST_PERIOD
        and period.iloc[-1] == LAST_PERIOD
    )
    return ok, runs


def add_periods(df: pd.DataFrame) -> tuple[pd.DataFrame, bool, str]:
    """Добавляет столбец period; при нарушении порядка — блоки по 2000 строк."""
    period = restore_periods(df["month"])
    ok, runs = periods_ok(period)
    note = (
        f"блоков периода по порядку строк: {runs}, уникальных периодов: {period.nunique()};\n"
        f"  первый {period.iloc[0]}, последний {period.iloc[-1]} -> "
        + (
            "монотонно, периоды = год-месяц"
            if ok
            else "НЕ монотонно, использованы блоки по 2000 строк"
        )
    )
    out = df.copy()
    if ok:
        out["period"] = period
    else:
        out["period"] = [f"блок {i // 2000 + 1:02d}" for i in range(len(df))]
    return out, ok, note

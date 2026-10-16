"""Требования к качеству данных D-01…D-09.

Каждая проверка возвращает CheckResult (число нарушений, действие, статус)
и таблицу после действия. Справочники категорий, доли unknown, диапазоны
макропоказателей и бины PSI
берутся из исходной части (build_reference) и хранятся в
models/reference/data_reference.json.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from bank_marketing.config import CATEGORICAL, ID_COLUMNS, MACRO, RAW_COLUMNS, TARGET
from bank_marketing.periods import MONTHS
from bank_marketing.stats import bin_shares, out_of_range_share, psi_bins, psi_from_shares

OK, FIXED, WARNING, CRITICAL = "ok", "fixed", "warning", "critical"

RULES = {
    "D-01": "все 21 исходный столбец и row_id, period; типы приводятся",
    "D-02": "периоды порции позже последнего обработанного",
    "D-03": "нет полных дубликатов (без row_id)",
    "D-04": "age 17–100, campaign ≥ 1, previous ≥ 0, pdays 0–999, duration ≥ 0",
    "D-05": "значения категориальных признаков из справочника",
    "D-06": "доля unknown выросла не больше чем на 5 п. п.",
    "D-07": "nonexistent ⇔ previous = 0; pdays ≠ 999 ⇒ previous > 0",
    "D-08": "доля записей с макропоказателями вне [min; max] исходной части ≤ 5 % по каждому",
    "D-09": "PSI age и campaign относительно исходной части не больше 0,2",
}
ACTIONS = {
    "D-01": "порция не обрабатывается",
    "D-02": "порция не обрабатывается",
    "D-03": "лишние копии удалены, предупреждение",
    "D-04": "записи исключены, предупреждение",
    "D-05": "значение заменено на unknown, предупреждение",
    "D-06": "предупреждение, проверка источника",
    "D-07": "предупреждение, записи остаются",
    "D-08": "предупреждение: модель вне области обучения, требуется переобучение",
    "D-09": "предупреждение о дрейфе",
}

INT_COLUMNS = ["age", "duration", "campaign", "pdays", "previous"]
FLOAT_COLUMNS = MACRO
# Значения из описания набора UCI, которых может не быть в исходной части
# (она охватывает только май–октябрь 2008 г. и не содержит poutcome = success).
DOMAIN_VALUES = {"month": MONTHS, "poutcome": ["failure", "nonexistent", "success"]}
RANGE_FEATURES = ["euribor3m", "nr.employed", "emp.var.rate", "cons.price.idx", "cons.conf.idx"]
OUT_OF_RANGE_MAX = 0.05
DRIFT_FEATURES = ["age", "campaign"]
UNKNOWN_GROWTH_MAX = 0.05
PSI_MAX = 0.2


@dataclass
class CheckResult:
    rule: str
    violations: int
    base: int  # знаменатель доли: записи или признаки
    unit: str
    status: str
    action: str
    details: str = ""

    @property
    def share(self) -> float:
        return self.violations / self.base if self.base else 0.0

    def to_dict(self) -> dict:
        return {**asdict(self), "share": self.share, "requirement": RULES[self.rule]}


def _result(
    rule: str, n: int, base: int, bad_status: str, unit="записей", details=""
) -> CheckResult:
    status = bad_status if n else OK
    return CheckResult(rule, int(n), int(base), unit, status, ACTIONS[rule] if n else "—", details)


def build_reference(df: pd.DataFrame) -> dict:
    """Справочник по исходной части (после приведения типов)."""
    ref = {
        "source": "data/initial/initial.csv",
        "rows": len(df),
        "last_period": str(df["period"].max()),
        "categories": {c: sorted(df[c].unique().tolist()) for c in CATEGORICAL},
        "domain_values": DOMAIN_VALUES,
        "unknown_share": {c: float((df[c] == "unknown").mean()) for c in CATEGORICAL},
        "ranges": {c: [float(df[c].min()), float(df[c].max())] for c in RANGE_FEATURES},
        "psi": {},
    }
    for c in DRIFT_FEATURES:
        spec = psi_bins(df[c])
        ref["psi"][c] = {"bins": spec, "shares": bin_shares(df[c], spec).tolist()}
    return ref


def check_schema(raw: pd.DataFrame) -> tuple[CheckResult, pd.DataFrame | None]:
    """D-01: состав столбцов и приведение типов (raw прочитан как строки)."""
    expected = ID_COLUMNS + RAW_COLUMNS
    missing = [c for c in expected if c not in raw.columns]
    if missing:
        res = _result("D-01", len(missing), len(expected), CRITICAL, "столбцов")
        res.details = "нет столбцов: " + ", ".join(missing)
        return res, None
    df = raw[expected].copy()
    bad = pd.Series(False, index=df.index)
    for c in ["row_id", *INT_COLUMNS]:
        num = pd.to_numeric(df[c], errors="coerce")
        bad |= num.isna() | (num % 1 != 0)
        df[c] = num.astype("Int64")
    for c in FLOAT_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        bad |= df[c].isna()
    period_ok = df["period"].str.fullmatch(r"\d{4}-\d{2}").fillna(False).astype(bool)
    bad |= ~period_ok | ~df[TARGET].isin(["yes", "no"]) | df[CATEGORICAL].isna().any(axis=1)
    res = _result("D-01", bad.sum(), len(df), CRITICAL)
    if res.violations:
        res.details = "строк, не приводимых к типам"
        return res, None
    df[["row_id", *INT_COLUMNS]] = df[["row_id", *INT_COLUMNS]].astype("int64")
    return res, df


def check_period_order(df: pd.DataFrame, last_period: str | None) -> CheckResult:
    """D-02: все периоды позже последнего обработанного."""
    n = 0 if last_period is None else int((df["period"] <= last_period).sum())
    details = f"последний обработанный: {last_period or 'нет'}"
    return _result("D-02", n, len(df), CRITICAL, details=details)


def check_duplicates(df: pd.DataFrame) -> tuple[CheckResult, pd.DataFrame]:
    """D-03: полные дубликаты без row_id; остаётся первая копия."""
    dup = df.drop(columns="row_id").duplicated()
    return _result("D-03", dup.sum(), len(df), FIXED), df[~dup]


def check_ranges(df: pd.DataFrame) -> tuple[CheckResult, pd.DataFrame]:
    """D-04: допустимые диапазоны; записи вне диапазонов исключаются."""
    rules = {
        "age": ~df["age"].between(17, 100),
        "campaign": df["campaign"] < 1,
        "previous": df["previous"] < 0,
        "pdays": ~df["pdays"].between(0, 999),
        "duration": df["duration"] < 0,
    }
    bad = pd.concat(rules, axis=1)
    details = "; ".join(f"{c} {int(v.sum())}" for c, v in bad.items() if v.any())
    mask = bad.any(axis=1)
    return _result("D-04", mask.sum(), len(df), FIXED, details=details), df[~mask]


def check_categories(df: pd.DataFrame, ref: dict) -> tuple[CheckResult, pd.DataFrame]:
    """D-05: значения вне справочника заменяются на unknown.

    Справочник — значения исходной части плюс значения из описания набора
    (DOMAIN_VALUES); последние, если их не было в исходной части, только
    перечисляются в details как новые.
    """
    out = df.copy()
    bad_rows = pd.Series(False, index=df.index)
    found, new = [], []
    for c in CATEGORICAL:
        seen = set(ref["categories"][c])
        allowed = seen | set(ref["domain_values"].get(c, []))
        bad = ~out[c].isin(allowed)
        if bad.any():
            found.append(f"{c}: {', '.join(sorted(out.loc[bad, c].unique()))}")
            out.loc[bad, c] = "unknown"
            bad_rows |= bad
        fresh = out[c].isin(allowed - seen)
        if fresh.any():
            counts = out.loc[fresh, c].value_counts().sort_index()
            new.append(f"{c}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    details = "; ".join(found)
    if new:
        details += (
            ("; " if details else "") + "новые по сравнению с исходной частью: " + "; ".join(new)
        )
    return _result("D-05", bad_rows.sum(), len(df), FIXED, details=details), out


def check_unknown_growth(df: pd.DataFrame, ref: dict) -> CheckResult:
    """D-06: рост доли unknown по каждому категориальному признаку ≤ 5 п. п."""
    grown = {}
    for c in CATEGORICAL:
        delta = float((df[c] == "unknown").mean()) - ref["unknown_share"][c]
        if delta > UNKNOWN_GROWTH_MAX:
            grown[c] = delta
    details = "; ".join(f"{c} +{d * 100:.1f} п. п." for c, d in grown.items())
    return _result("D-06", len(grown), len(CATEGORICAL), WARNING, "признаков", details)


def check_consistency(df: pd.DataFrame) -> CheckResult:
    """D-07: nonexistent ⇔ previous = 0; pdays ≠ 999 ⇒ previous > 0."""
    nonexist, prev0 = df["poutcome"] == "nonexistent", df["previous"] == 0
    a = nonexist != prev0
    b = (df["pdays"] != 999) & prev0
    details = (
        f"nonexistent ⇔ previous = 0: {int(a.sum())}; pdays ≠ 999 при previous = 0: {int(b.sum())}"
    )
    return _result("D-07", (a | b).sum(), len(df), WARNING, details=details)


def check_macro_range(df: pd.DataFrame, ref: dict) -> CheckResult:
    """D-08: доля записей с макропоказателем вне [min; max] исходной части ≤ 5 %."""
    shares = {c: out_of_range_share(df[c], *ref["ranges"][c]) for c in RANGE_FEATURES}
    n = sum(v > OUT_OF_RANGE_MAX for v in shares.values())
    details = "; ".join(f"{c} {v:.1%}" for c, v in shares.items())
    return _result("D-08", n, len(RANGE_FEATURES), WARNING, "признаков", details)


def check_drift(df: pd.DataFrame, ref: dict) -> CheckResult:
    """D-09: PSI age и campaign относительно исходной части ≤ 0,2."""
    values = {
        c: psi_from_shares(ref["psi"][c]["shares"], df[c], ref["psi"][c]["bins"])
        for c in DRIFT_FEATURES
    }
    n = sum(v > PSI_MAX for v in values.values())
    details = "; ".join(f"{c} {v:.3f}" for c, v in values.items())
    return _result("D-09", n, len(DRIFT_FEATURES), WARNING, "признаков", details)


def validate(
    raw: pd.DataFrame, ref: dict | None, last_period: str | None
) -> tuple[pd.DataFrame | None, list[CheckResult]]:
    """Прогоняет D-01…D-09 по порядку; при критическом нарушении — остановка.

    ref = None — проверка исходной части: справочник строится по ней самой.
    """
    res, df = check_schema(raw)
    results = [res]
    if df is None:
        return None, results
    res = check_period_order(df, last_period)
    results.append(res)
    if res.status == CRITICAL:
        return None, results
    ref = ref or build_reference(df)
    for check in (check_duplicates, check_ranges):
        res, df = check(df)
        results.append(res)
    res, df = check_categories(df, ref)
    results += [
        res,
        check_unknown_growth(df, ref),
        check_consistency(df),
        check_macro_range(df, ref),
        check_drift(df, ref),
    ]
    return df, results


def verdict(results: list[CheckResult]) -> str:
    statuses = {r.status for r in results}
    if CRITICAL in statuses:
        return "не обрабатывается"
    if statuses & {FIXED, WARNING}:
        return "принята с предупреждениями"
    return "принята"


def to_jsonable(obj):
    """Приводит numpy-типы к встроенным для json.dump."""
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def reference_from_initial(raw: pd.DataFrame) -> dict:
    """Справочник по исходной части, прочитанной как строки."""
    _, df = check_schema(raw)
    if df is None:
        raise ValueError("исходная часть не проходит D-01")
    return build_reference(df)

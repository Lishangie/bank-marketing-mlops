"""Статистики для анализа и мониторинга: PSI, V Крамера, ДИ Уилсона."""

import numpy as np
import pandas as pd
from scipy import stats

PSI_BINS = 10
PSI_MAX_UNIQUE = 10  # не больше стольких значений в исходной части — бины по значениям
PSI_EPS = 1e-4  # нижняя граница доли в бине, чтобы не делить на ноль


def psi_bins(ref: pd.Series, bins: int = PSI_BINS, max_unique: int = PSI_MAX_UNIQUE) -> dict:
    """Бины PSI по исходной части.

    Мало значений — каждое значение отдельный бин плюс бин «прочие» для новых;
    иначе квантильные бины (границы — внутренние квантили, крайние открыты).
    """
    values = np.sort(ref.dropna().unique())
    if len(values) <= max_unique:
        return {"kind": "values", "values": values.tolist()}
    qs = np.quantile(ref.dropna(), np.linspace(0, 1, bins + 1)[1:-1])
    return {"kind": "quantile", "edges": np.unique(qs).tolist()}


def bin_shares(x: pd.Series, spec: dict) -> np.ndarray:
    x = x.dropna().to_numpy()
    if spec["kind"] == "values":
        values = np.asarray(spec["values"])
        counts = [(x == v).sum() for v in values]
        counts.append(len(x) - sum(counts))  # прочие значения
    else:
        idx = np.searchsorted(np.asarray(spec["edges"]), x, side="right")
        counts = np.bincount(idx, minlength=len(spec["edges"]) + 1)
    return np.asarray(counts, dtype=float) / max(len(x), 1)


def psi(ref: pd.Series, cur: pd.Series, spec: dict | None = None) -> float:
    """Population Stability Index распределения cur относительно ref."""
    spec = spec or psi_bins(ref)
    return psi_from_shares(bin_shares(ref, spec).tolist(), cur, spec)


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """V Крамера по таблице сопряжённости (без поправки на смещение)."""
    table = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    k = min(table.shape) - 1
    return float(np.sqrt(chi2 / (table.to_numpy().sum() * k))) if k > 0 else float("nan")


def wilson_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Доверительный интервал Уилсона для доли k из n."""
    if n == 0:
        return float("nan"), float("nan")
    z = stats.norm.ppf(0.5 + conf / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(center - half), float(center + half)


def psi_from_shares(ref_shares: list[float], cur: pd.Series, spec: dict) -> float:
    """PSI по сохранённым долям бинов исходной части (без самих данных)."""
    p = np.clip(np.asarray(ref_shares, dtype=float), PSI_EPS, None)
    q = np.clip(bin_shares(cur, spec), PSI_EPS, None)
    return float(np.sum((q - p) * np.log(q / p)))

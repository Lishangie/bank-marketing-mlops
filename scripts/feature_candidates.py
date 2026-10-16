"""Кандидаты в признаки: связь с откликом на исходной части и устойчивость на порциях.

Для групповых признаков — доля отклика по группам, для числовых — Спирмен с y;
то же по каждой порции и на объединении порций. Параметры (P99 campaign,
стандартизация и PCA макропоказателей) оцениваются только по исходной части.
В пайплайн признаки не встраиваются (это ПР3).

Результат: report_assets/pr2/features_out.txt.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from bank_marketing.config import PR2_DIR, TARGET
from bank_marketing.data import load_parts
from bank_marketing.report import emit

PCA_COLUMNS = ["emp.var.rate", "euribor3m", "nr.employed"]
GROUPS = {
    "had_prev_contact": [True, False],
    "age_group": ["<25", "25–59", "60+"],
    "job_group": ["student", "retired", "unemployed", "остальные"],
}
NUMERIC = ["contacts_total", "campaign_capped", "euribor_delta_3m", "macro_pc1"]
COL = 15


def shift_period(period: str, months: int) -> str:
    y, m = map(int, period.split("-"))
    k = y * 12 + (m - 1) - months
    return f"{k // 12}-{k % 12 + 1:02d}"


def euribor_lag_3m(all_rows: pd.DataFrame) -> pd.Series:
    """Средний euribor3m по месяцу на 3 месяца раньше; если месяца нет — ближайший
    доступный предыдущий; для первых месяцев ряда — NaN."""
    monthly = all_rows.groupby("period")["euribor3m"].mean().sort_index()
    lag = {}
    for p in monthly.index:
        prev = monthly.loc[: shift_period(p, 3)]
        lag[p] = prev.iloc[-1] if len(prev) else np.nan
    return all_rows["period"].map(lag)


def add_features(df: pd.DataFrame, cap: float, lag: pd.Series, scaler, pca) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["had_prev_contact"] = df["pdays"] != 999
    out["contacts_total"] = df["campaign"] + df["previous"]
    out["campaign_capped"] = df["campaign"].clip(upper=cap)
    out["age_group"] = pd.cut(df["age"], [0, 24, 59, np.inf], labels=GROUPS["age_group"]).astype(
        str
    )
    out["job_group"] = df["job"].where(df["job"].isin(GROUPS["job_group"][:3]), "остальные")
    out["euribor_delta_3m"] = df["euribor3m"] - lag.loc[df.index]
    out["macro_pc1"] = pca.transform(scaler.transform(df[PCA_COLUMNS]))[:, 0]
    out[TARGET] = df[TARGET]
    return out


def rate_cell(d: pd.DataFrame, col: str, group) -> str:
    s = d.loc[d[col] == group, TARGET]
    return f"{s.mean():.1%} ({len(s)})" if len(s) else "— (0)"


def rho_cell(d: pd.DataFrame, col: str) -> str:
    x = d[[col, TARGET]].dropna()
    if x[col].nunique() < 2:
        return "конст."
    return f"{stats.spearmanr(x[col], x[TARGET])[0]:+.3f}"


def main() -> None:
    parts = load_parts()
    init = parts["initial"]
    batches = [n for n in parts if n != "initial"]

    all_rows = pd.concat(parts.values(), ignore_index=True)
    # Индексы частей внутри общего ряда — для сдвига euribor по всему ряду.
    offsets = np.cumsum([0] + [len(d) for d in parts.values()])
    lag_all = euribor_lag_3m(all_rows)
    cap = float(init["campaign"].quantile(0.99))
    scaler = StandardScaler().fit(init[PCA_COLUMNS])
    pca = PCA(n_components=1).fit(scaler.transform(init[PCA_COLUMNS]))

    feats = {}
    for (name, d), start in zip(parts.items(), offsets[:-1], strict=True):
        lag = pd.Series(lag_all.to_numpy()[start : start + len(d)], index=d.index)
        feats[name] = add_features(d, cap, lag, scaler, pca)
    feats["порции"] = pd.concat([feats[b] for b in batches], ignore_index=True)
    names = ["initial", *batches, "порции"]

    out = [
        "Кандидаты в признаки: связь с y на исходной части, по порциям и на объединении порций",
        f"Параметры по исходной части: P99 campaign = {cap:g}; PCA по стандартизованным "
        f"{', '.join(PCA_COLUMNS)}",
        "",
        "Доля отклика (n) по группам:",
        f"  {'признак = группа':<29}" + "".join(f"{n:>{COL}}" for n in names),
    ]
    for col, groups in GROUPS.items():
        for g in groups:
            label = f"{col} = {g}"
            out.append(
                f"  {label:<29}" + "".join(f"{rate_cell(feats[n], col, g):>{COL}}" for n in names)
            )
    out += ["", "Спирмен с y:", f"  {'признак':<29}" + "".join(f"{n:>{COL}}" for n in names)]
    for col in NUMERIC:
        out.append(f"  {col:<29}" + "".join(f"{rho_cell(feats[n], col):>{COL}}" for n in names))
    out += ["", "Диапазон (min … max; NaN — нет месяца на 3 месяца раньше):"]
    for col in ["euribor_delta_3m", "macro_pc1"]:
        out.append(f"  {col}:")
        for n in names:
            s = feats[n][col]
            out.append(
                f"    {n:<9} {s.min():>9.3f} … {s.max():>8.3f}; среднее {s.mean():>8.3f}; "
                f"NaN {int(s.isna().sum())}"
            )
    load = dict(zip(PCA_COLUMNS, pca.components_[0], strict=True))
    out += [
        "",
        f"macro_pc1: объяснённая доля дисперсии {pca.explained_variance_ratio_[0]:.4f}; "
        "нагрузки " + ", ".join(f"{k} {v:+.3f}" for k, v in load.items()),
        "  стандартизация по исходной части: среднее "
        + ", ".join(f"{m:g}" for m in scaler.mean_)
        + "; std "
        + ", ".join(f"{s:.4g}" for s in scaler.scale_),
        "euribor_delta_3m: месяцы без значения: "
        + ", ".join(sorted(all_rows.loc[lag_all.isna(), "period"].unique())),
    ]
    emit(out, PR2_DIR / "features_out.txt")


if __name__ == "__main__":
    main()

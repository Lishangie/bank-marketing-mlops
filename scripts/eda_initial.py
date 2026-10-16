"""Разведочный анализ исходной части (data/initial/initial.csv).

Все статистики считаются по исходной части; порции используются только
для проверки гипотез на новых данных и оценки сдвига (блоки 8 и 9).

Результаты: report_assets/pr2/eda_initial_out.txt, рисунки рис_1_2 … рис_1_5
в report_assets/pr2/img/ (копия в reports/figures/pr2/).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy import stats
from sklearn.metrics import roc_auc_score

from bank_marketing.config import (
    CATEGORICAL,
    MACRO,
    NUMERIC,
    PR2_DIR,
    PR2_FIGURES_DIR,
    PR2_IMG_DIR,
    RAW_COLUMNS,
    TARGET,
)
from bank_marketing.data import load_parts
from bank_marketing.plotting import (
    C_ACCENT,
    C_MEAN,
    C_NO,
    C_YES,
    CLIP_99,
    CM,
    LABEL_BOX,
    RU,
    WIDTH,
    apply_style,
    category_order,
    comma_axes,
    num,
    pct,
    pct_formatter,
    save,
)
from bank_marketing.report import emit, wrap_items
from bank_marketing.stats import cramers_v, psi, psi_bins, wilson_ci

FIG_DIRS = (PR2_IMG_DIR, PR2_FIGURES_DIR)
FEATURES = [c for c in RAW_COLUMNS if c != TARGET]
CAT_FIG = [("job", "education"), ("month", "default"), ("contact", "poutcome")]
PSI_FEATURES = ["age", "campaign", "euribor3m", "nr.employed", "cons.conf.idx"]
SHIFT_MEANS = ["euribor3m", "nr.employed", "emp.var.rate"]
SHORT = {
    "duration": "durat.",
    "campaign": "camp.",
    "previous": "prev.",
    "emp.var.rate": "emp.var",
    "cons.price.idx": "cpi",
    "cons.conf.idx": "cci",
    "euribor3m": "eur3m",
    "nr.employed": "nr.emp",
}
CAMPAIGN_BINS = [0, 1, 2, 3, 5, np.inf]
CAMPAIGN_LABELS = ["1", "2", "3", "4–5", "6+"]

HYPOTHESES = {
    "H1": "poutcome = success: отклик выше, чем при failure и nonexistent",
    "H2": "contact = cellular: отклик выше, чем при telephone",
    "H3": "отклик падает с ростом campaign (1, 2, 3, 4–5, 6+)",
    "H4": "student и retired откликаются чаще остальных",
}
H_AXIS = {
    "H1": "H1: итог прошлой кампании",
    "H2": "H2: тип связи",
    "H3": "H3: число контактов в кампании",
    "H4": "H4: род занятий",
}

apply_style()


# ---------- группы для гипотез ----------


def hypothesis_groups(df: pd.DataFrame, key: str) -> tuple[pd.Series, list[str]]:
    if key == "H1":
        return df["poutcome"], ["success", "failure", "nonexistent"]
    if key == "H2":
        return df["contact"], ["cellular", "telephone"]
    if key == "H3":
        g = pd.cut(df["campaign"], CAMPAIGN_BINS, labels=CAMPAIGN_LABELS).astype(str)
        return g, CAMPAIGN_LABELS
    job = df["job"].where(df["job"].isin(["student", "retired"]), "остальные")
    return job, ["student", "retired", "остальные"]


def group_rates(df: pd.DataFrame, key: str) -> pd.DataFrame:
    g, order = hypothesis_groups(df, key)
    agg = df[TARGET].groupby(g).agg(n="size", yes="sum").reindex(order).fillna(0).astype(int)
    agg["rate"] = agg["yes"] / agg["n"].replace(0, np.nan)
    ci = [wilson_ci(k, n) for k, n in zip(agg["yes"], agg["n"], strict=True)]
    agg["lo"], agg["hi"] = zip(*ci, strict=True)
    return agg


def chi2_test(df: pd.DataFrame, key: str) -> tuple[float, int, float]:
    g, _ = hypothesis_groups(df, key)
    table = pd.crosstab(g, df[TARGET])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return float("nan"), 0, float("nan")
    chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
    return float(chi2), int(dof), float(p)


# ---------- текстовые блоки ----------


def block_structure(df: pd.DataFrame) -> list[str]:
    out = [
        "== 1. Структура ==",
        f"Строк: {len(df)}; столбцов: {df.shape[1]} (21 исходный + row_id, period); "
        f"признаков: {len(FEATURES)}",
        f"  {'признак':<15}{'тип':<9}{'уник.':>6}  min / медиана / max или unknown",
    ]
    for c in FEATURES:
        s = df[c]
        if c in NUMERIC:
            info = f"{s.min():g} / {s.median():g} / {s.max():g}"
            if c == "pdays":
                info += f"; pdays = 999: {(s == 999).mean():.2%}"
        else:
            unk = s == "unknown"
            info = f"unknown {int(unk.sum())} ({unk.mean():.2%})"
        out.append(f"  {c:<15}{str(s.dtype):<9}{s.nunique():>6}  {info}")
    return out


def block_target(df: pd.DataFrame) -> list[str]:
    n_yes = int(df[TARGET].sum())
    n_no = len(df) - n_yes
    return [
        "== 2. Целевая y ==",
        f"yes: {n_yes} ({n_yes / len(df):.2%}); no: {n_no} ({n_no / len(df):.2%}); "
        f"соотношение yes : no = 1 : {n_no / n_yes:.2f}",
    ]


def block_quality(df: pd.DataFrame) -> list[str]:
    dups = int(df.drop(columns="row_id").duplicated().sum())
    out = ["== 3. Качество ==", f"Полные дубликаты (без row_id): {dups} ({dups / len(df):.3%})"]
    items = []
    for c in NUMERIC:
        q1, q3 = df[c].quantile([0.25, 0.75])
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        n = int(((df[c] < lo) | (df[c] > hi)).sum())
        items.append(f"{c} {n} ({n / len(df):.2%})")
    out.append("Выбросы по правилу 1,5 IQR (число, доля):")
    out += wrap_items(items)
    out.append("  (pdays и previous: IQR = 0, выбросами считаются все pdays ≠ 999 и previous > 0)")
    impossible = {
        "age вне 17–100": ~df["age"].between(17, 100),
        "campaign < 1": df["campaign"] < 1,
        "previous < 0": df["previous"] < 0,
        "duration = 0": df["duration"] == 0,
        "pdays вне 0–999": ~df["pdays"].between(0, 999),
    }
    out.append("Невозможные значения:")
    out += wrap_items([f"{k}: {int(v.sum())}" for k, v in impossible.items()])
    nonexist, prev0 = df["poutcome"] == "nonexistent", df["previous"] == 0
    out += [
        "Согласованность pdays / previous / poutcome:",
        f"  nonexistent и previous > 0: {int((nonexist & ~prev0).sum())}; "
        f"не nonexistent и previous = 0: {int((~nonexist & prev0).sum())}",
        f"  pdays ≠ 999 и previous = 0: {int(((df['pdays'] != 999) & prev0).sum())}",
        f"  previous > 0 и pdays = 999 (был контакт, дата неизвестна): "
        f"{int(((df['pdays'] == 999) & ~prev0).sum())}",
    ]
    return out


def block_numeric(df: pd.DataFrame, corr: pd.DataFrame) -> list[str]:
    out = [
        "== 4. Числовые признаки и y ==",
        f"  {'признак':<15}{'медиана no':>12}{'медиана yes':>13}{'p Манна–Уитни':>15}"
        f"{'Спирмен с y':>13}",
    ]
    no, yes = df[df[TARGET] == 0], df[df[TARGET] == 1]
    for c in NUMERIC:
        p = stats.mannwhitneyu(no[c], yes[c], alternative="two-sided").pvalue
        out.append(
            f"  {c:<15}{no[c].median():>12g}{yes[c].median():>13g}{p:>15.2e}"
            f"{corr.loc[c, TARGET]:>+13.3f}"
        )
    return out


def block_categorical(df: pd.DataFrame) -> list[str]:
    out = ["== 5. Категориальные признаки и y ==", "V Крамера:"]
    vs = {c: cramers_v(df[c], df[TARGET]) for c in CATEGORICAL}
    out += wrap_items([f"{c} {v:.3f}" for c, v in sorted(vs.items(), key=lambda kv: -kv[1])])
    out.append("Доля отклика (n) по значениям, по убыванию доли:")
    for c in CATEGORICAL:
        g = df.groupby(c)[TARGET].agg(["mean", "size"]).sort_values("mean", ascending=False)
        items = [f"{k} {r['mean']:.1%} ({int(r['size'])})" for k, r in g.iterrows()]
        out += wrap_items(items, indent=f"  {c}:", cont="      ")
    return out


def block_pairs(corr: pd.DataFrame) -> list[str]:
    feat = corr.loc[NUMERIC, NUMERIC]
    iu = np.triu_indices(len(NUMERIC), k=1)
    pairs = pd.Series(
        feat.to_numpy()[iu],
        index=[f"{NUMERIC[i]} – {NUMERIC[j]}" for i, j in zip(*iu, strict=True)],
    )
    strong = pairs[pairs.abs() > 0.8].sort_values(key=abs, ascending=False)
    out = ["== 6. Связи признаков (Спирмен) ==", f"Пары с |r| > 0,8: {len(strong)}"]
    out += wrap_items([f"{k} {v:+.3f}" for k, v in strong.items()]) if len(strong) else []
    out.append("Матрица Спирмена (числовые и y):")
    out.append(f"  {'':<15}" + "".join(f"{SHORT.get(c, c):>8}" for c in corr.columns))
    for c in corr.index:
        out.append(f"  {c:<15}" + "".join(f"{v:>8.2f}" for v in corr.loc[c]))
    const = [c for c in corr.columns if corr[c].isna().all()]
    if const:
        out.append(f"  nan: {', '.join(const)} — константа в исходной части (на рисунке опущен)")
    return out


def block_leakage(df: pd.DataFrame) -> list[str]:
    auc = roc_auc_score(df[TARGET], df["duration"])
    out = [
        "== 7. Утечка и доступность ==",
        f"ROC-AUC одного duration: {auc:.4f} (длительность известна только после звонка)",
        "Макропоказатели в исходной части (уникальных значений, min–max):",
    ]
    out += wrap_items([f"{c} {df[c].nunique()}, {df[c].min():g}–{df[c].max():g}" for c in MACRO])
    months = df.groupby("period")["row_id"].size()
    out.append("Месяцы в исходной части (строк):")
    out += wrap_items([f"{p} {n}" for p, n in months.items()])
    return out


def block_hypotheses(parts: dict[str, pd.DataFrame]) -> tuple[list[str], dict]:
    init = parts["initial"]
    out = ["== 8. Гипотезы (доля отклика, 95 % ДИ Уилсона, хи-квадрат) =="]
    tables = {}
    for key, text in HYPOTHESES.items():
        t = group_rates(init, key)
        tables[key] = t
        chi2, dof, p = chi2_test(init, key)
        out.append(f"{key}: {text}")
        out.append(f"  {'группа':<13}{'n':>7}{'yes':>7}{'доля':>9}   95 % ДИ")
        for g, r in t.astype(object).iterrows():
            if r["n"] == 0:
                out.append(f"  {g:<13}{0:>7}{0:>7}{'—':>9}   нет записей в исходной части")
                continue
            out.append(
                f"  {g:<13}{r['n']:>7}{r['yes']:>7}{r['rate']:>9.2%}   "
                f"[{r['lo']:.2%}; {r['hi']:.2%}]"
            )
        hi_g, lo_g = t["rate"].idxmax(), t["rate"].idxmin()
        out.append(
            f"  хи-квадрат = {chi2:.2f}, df = {dof}, p = {p:.2e}; отношение долей "
            f"{hi_g} / {lo_g} = {t.loc[hi_g, 'rate'] / t.loc[lo_g, 'rate']:.2f}"
        )
        if key == "H3":
            rho, p_rho = stats.spearmanr(init["campaign"], init[TARGET])
            out.append(f"  Спирмен campaign – y: {rho:+.3f} (p = {p_rho:.2e})")
        out.append("  По частям: доля отклика (n); последняя строка — p хи-квадрат")
        names = list(parts)
        out.append(f"  {'группа':<13}" + "".join(f"{n:>16}" for n in names))
        per_part = {n: group_rates(d, key) for n, d in parts.items()}
        for g in t.index:
            cells = []
            for n in names:
                r = per_part[n].astype(object).loc[g]
                cells.append(f"{r['rate']:.1%} ({r['n']})" if r["n"] else "— (0)")
            out.append(f"  {g:<13}" + "".join(f"{c:>16}" for c in cells))
        ps = [chi2_test(d, key)[2] for d in parts.values()]
        out.append(f"  {'p':<13}" + "".join(f"{p:>16.1e}" for p in ps))
    return out, tables


def block_shift(parts: dict[str, pd.DataFrame]) -> list[str]:
    out = [
        "== 9. Сдвиг между частями ==",
        f"  {'часть':<11}{'строк':>7}{'отклик':>9}{'euribor3m':>11}{'nr.employed':>13}"
        f"{'emp.var.rate':>14}{'cellular':>10}{'success':>9}",
    ]
    for name, d in parts.items():
        out.append(
            f"  {name:<11}{len(d):>7}{d[TARGET].mean():>9.2%}"
            + "".join(
                f"{d[c].mean():>{w}.3f}" for c, w in zip(SHIFT_MEANS, (11, 13, 14), strict=True)
            )
            + f"{(d['contact'] == 'cellular').mean():>10.2%}"
            + f"{(d['poutcome'] == 'success').mean():>9.2%}"
        )
    init = parts["initial"]
    batches = [n for n in parts if n != "initial"]
    out.append(
        "PSI относительно исходной части — справка, в проверке качества не используется"
        " для макропоказателей (10 квантильных бинов; ≤ 10 значений — по значениям):"
    )
    out.append(f"  {'признак':<15}{'бины':<10}" + "".join(f"{b:>10}" for b in batches))
    for c in PSI_FEATURES:
        spec = psi_bins(init[c])
        kind = "значения" if spec["kind"] == "values" else "квантили"
        vals = [psi(init[c], parts[b][c], spec) for b in batches]
        out.append(f"  {c:<15}{kind:<10}" + "".join(f"{v:>10.3f}" for v in vals))
    return out


# ---------- рисунки ----------


def fig_numeric(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(5, 2, figsize=(WIDTH, 21 * CM), layout="constrained")
    for ax, col in zip(axes.flat, NUMERIC, strict=True):
        data = df[[col, TARGET]]
        if col == "pdays":
            data = data[data[col] != 999]
        if col in CLIP_99:
            data = data[data[col] <= data[col].quantile(0.99)]
        if data.empty:  # в исходной части pdays всегда 999
            ax.text(0.5, 0.5, "все значения = 999", transform=ax.transAxes, ha="center")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(RU[col] + " (без 999)")
            continue
        discrete = data[col].nunique() <= 30 and np.allclose(data[col] % 1, 0)
        for cls, color, label in ((0, C_NO, "нет"), (1, C_YES, "да")):
            sns.histplot(
                data.loc[data[TARGET] == cls, col],
                ax=ax,
                stat="probability",
                discrete=discrete,
                bins="auto" if discrete else 30,
                color=color,
                alpha=0.55,
                edgecolor="none",
                label=f"y = {label}",
                element="bars",
            )
        xlabel = RU[col]
        if col == "pdays":
            xlabel += " (без 999)"
        elif col in CLIP_99:
            xlabel += ", ≤ P99"
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Доля в классе")
        comma_axes(ax, y=False)
        ax.yaxis.set_major_formatter(pct_formatter())
        ax.xaxis.get_major_formatter().set_scientific(False)
        if discrete:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes.flat[0].legend(frameon=False)
    save(fig, "рис_1_2.png", FIG_DIRS)


def fig_categorical(df: pd.DataFrame) -> None:
    ratios = [max(df[a].nunique(), df[b].nunique()) + 1.5 for a, b in CAT_FIG]
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(WIDTH, 19 * CM),
        layout="constrained",
        gridspec_kw={"height_ratios": ratios},
    )
    mean = df[TARGET].mean()
    for row, cols in zip(axes, CAT_FIG, strict=True):
        for ax, col in zip(row, cols, strict=True):
            rates = df.groupby(col)[TARGET].mean()
            order = category_order(col, rates)
            if col in ("month", "education"):
                order = order[::-1]  # сверху вниз в естественном порядке
            vals = rates.reindex(order)
            span = max(vals.max() * 1.35, mean * 1.5)
            ax.barh(order, vals.values, color=C_ACCENT, height=0.65)
            ax.axvline(mean, color=C_MEAN, ls="--", lw=1, zorder=2)
            for i, v in enumerate(vals.values):
                ax.text(
                    v + span * 0.015,
                    i,
                    pct(v, 1),
                    va="center",
                    fontsize=9,
                    bbox=LABEL_BOX,
                    zorder=3,
                )
            ax.set_xlim(0, span)
            ax.xaxis.set_major_formatter(pct_formatter())
            ax.set_xlabel("Доля отклика")
            ax.set_ylabel(RU[col])
            ax.tick_params(axis="y", length=0)
    axes[0, 0].text(
        mean,
        1.0,
        f" средняя {pct(mean, 1)}",
        transform=axes[0, 0].get_xaxis_transform(),
        color=C_MEAN,
        fontsize=9,
        va="bottom",
        ha="left",
    )
    save(fig, "рис_1_3.png", FIG_DIRS)


def fig_corr(corr: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(WIDTH, 15 * CM), layout="constrained")
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    labels = corr.map(lambda v: num(v, 2))
    sns.heatmap(
        corr,
        mask=mask,
        annot=labels,
        fmt="",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        annot_kws={"fontsize": 9},
        ax=ax,
        cbar_kws={"label": "Коэффициент Спирмена", "shrink": 0.8},
    )
    ax.collections[0].colorbar.ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: num(v, 1))
    )
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    ax.set_xlabel("")
    ax.set_ylabel("")
    save(fig, "рис_1_4.png", FIG_DIRS)


def fig_hypotheses(tables: dict, mean: float) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 13.5 * CM), layout="constrained")
    for ax, (key, t) in zip(axes.flat, tables.items(), strict=True):
        x = np.arange(len(t))
        err = np.vstack([t["rate"] - t["lo"], t["hi"] - t["rate"]])
        ax.errorbar(
            x, t["rate"], yerr=err, fmt="o", color=C_ACCENT, ms=5, capsize=4, lw=1.2, zorder=3
        )
        ax.axhline(mean, color=C_MEAN, ls="--", lw=1, zorder=1)
        # Число записей n — второй строкой подписи группы; пустая группа отмечена отдельно.
        labels = [f"{g}\n{num(n, 0)}" for g, n in zip(t.index, t["n"], strict=True)]
        ax.set_xticks(x, labels)
        for xi, n in enumerate(t["n"]):
            if n == 0:
                ax.text(xi, mean * 1.5, "нет записей", ha="center", va="bottom", fontsize=9)
        ax.set_xlim(-0.6, len(t) - 0.4)
        ax.set_ylim(0, t["hi"].max() * 1.1)
        ax.set_xlabel(H_AXIS[key])
        ax.set_ylabel("Доля отклика")
        ax.yaxis.set_major_formatter(pct_formatter())
    handles = [
        Line2D([], [], color=C_ACCENT, marker="o", ms=5, lw=1.2, label="доля отклика и 95 % ДИ"),
        Line2D(
            [], [], color=C_MEAN, ls="--", lw=1, label=f"средняя по исходной части {pct(mean, 1)}"
        ),
        Line2D([], [], ls="none", label="под группой — число записей n"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncols=2, frameon=False)
    save(fig, "рис_1_5.png", FIG_DIRS)


def main() -> None:
    parts = load_parts()
    init = parts["initial"]
    corr = init[NUMERIC + [TARGET]].corr(method="spearman")
    # Постоянные в исходной части признаки (pdays = 999) на рисунке не показываются.
    corr_fig = corr.dropna(how="all").dropna(axis=1, how="all")

    hyp, tables = block_hypotheses(parts)
    blocks = [
        block_structure(init),
        block_target(init),
        block_quality(init),
        block_numeric(init, corr),
        block_categorical(init),
        block_pairs(corr),
        block_leakage(init),
        hyp,
        block_shift(parts),
    ]
    out = [
        f"EDA исходной части: data/initial/initial.csv ({init['period'].iloc[0]}…"
        f"{init['period'].iloc[-1]})"
    ]
    for b in blocks:
        out += ["", *b]

    fig_numeric(init)
    fig_categorical(init)
    fig_corr(corr_fig)
    fig_hypotheses(tables, init[TARGET].mean())
    emit(out, PR2_DIR / "eda_initial_out.txt")


if __name__ == "__main__":
    main()

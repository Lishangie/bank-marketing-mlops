"""Разведочный анализ UCI Bank Marketing (bank-additional-full.csv).

Результаты:
- рисунки рис_3_1 … рис_3_5 в report_assets/img/ (копия в reports/figures/);
- текстовая сводка report_assets/eda_out.txt (не длиннее 150 строк).
"""

import platform
import shutil
import subprocess
from importlib.metadata import PackageNotFoundError, version

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import roc_auc_score

from bank_marketing.config import (
    ASSETS_DIR,
    ASSETS_IMG_DIR,
    CATEGORICAL,
    FIGURES_DIR,
    NUMERIC,
    TARGET,
)
from bank_marketing.data import load_raw
from bank_marketing.periods import add_periods
from bank_marketing.plotting import (
    C_ACCENT,
    C_NO,
    C_YES,
    CLIP_99,
    CM,
    LABEL_BOX,
    RU,
    WIDTH,
    apply_style,
    category_order,
)
from bank_marketing.plotting import save as save_fig
from bank_marketing.report import wrap_items

MAX_LINES = 150
WRAP = 90

apply_style(font_size=9, tick_size=8.5)


def save(fig, name: str) -> None:
    # Как в ПР1: без bbox_inches="tight", размер страницы задаётся figsize.
    save_fig(fig, name, (ASSETS_IMG_DIR, FIGURES_DIR), tight=False)


def fig_balance(df: pd.DataFrame) -> None:
    counts = df[TARGET].value_counts().reindex([0, 1])
    share = counts / counts.sum()
    fig, ax = plt.subplots(figsize=(WIDTH, 7 * CM), layout="constrained")
    bars = ax.bar(["нет", "да"], counts.values, color=[C_NO, C_YES], width=0.55)
    for bar, n, s in zip(bars, counts.values, share.values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            n,
            f"{s:.1%}\n({n:,})".replace(",", " "),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xlabel("Оформил срочный депозит (y)")
    ax.set_ylabel("Число клиентов")
    ax.set_ylim(0, counts.max() * 1.22)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}".replace(",", " "))
    save(fig, "рис_3_1.png")


def fig_numeric(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(5, 2, figsize=(WIDTH, 19.5 * CM), layout="constrained")
    for ax, col in zip(axes.flat, NUMERIC, strict=True):
        data = df[[col, TARGET]]
        if col == "pdays":
            data = data[data[col] != 999]
        if col in CLIP_99:
            data = data[data[col] <= data[col].quantile(0.99)]
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
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes.flat[0].legend(frameon=False)
    save(fig, "рис_3_2.png")


def fig_categorical(df: pd.DataFrame) -> None:
    layout = [("job", "education"), ("month", "day_of_week"), ("contact", "poutcome")]
    ratios = [max(df[a].nunique(), df[b].nunique()) + 1.5 for a, b in layout]
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(WIDTH, 19.5 * CM),
        layout="constrained",
        gridspec_kw={"height_ratios": ratios},
    )
    mean = df[TARGET].mean()
    for row, cols in zip(axes, layout, strict=True):
        for ax, col in zip(row, cols, strict=True):
            rates = df.groupby(col)[TARGET].mean()
            order = category_order(col, rates)
            if col in ("month", "day_of_week", "education"):
                order = order[::-1]  # сверху вниз в естественном порядке
            vals = rates.reindex(order)
            span = max(vals.max() * 1.3, mean * 1.4)
            ax.barh(order, vals.values, color=C_ACCENT, height=0.65)
            ax.axvline(mean, color="#b03a2e", ls="--", lw=1, zorder=2)
            for i, v in enumerate(vals.values):
                ax.text(
                    v + span * 0.015,
                    i,
                    f"{v:.1%}",
                    va="center",
                    fontsize=7.5,
                    bbox=LABEL_BOX,
                    zorder=3,
                )
            ax.set_xlim(0, span)
            ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.set_xlabel("Доля отклика")
            ax.set_ylabel(RU[col])
            ax.tick_params(axis="y", length=0)
    axes[0, 0].text(
        mean,
        1.0,
        f" средняя {mean:.1%}",
        transform=axes[0, 0].get_xaxis_transform(),
        color="#b03a2e",
        fontsize=8,
        va="bottom",
        ha="left",
    )
    save(fig, "рис_3_3.png")


def fig_corr(corr: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(WIDTH, 14 * CM), layout="constrained")
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        annot_kws={"fontsize": 7.5},
        ax=ax,
        cbar_kws={"label": "Коэффициент Спирмена", "shrink": 0.8},
    )
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    ax.set_xlabel("")
    ax.set_ylabel("")
    save(fig, "рис_3_4.png")


def fig_time(per: pd.DataFrame) -> None:
    fig, (ax1, ax3) = plt.subplots(
        2,
        1,
        figsize=(WIDTH, 13 * CM),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [3, 2]},
    )
    x = np.arange(len(per))
    ax1.bar(x, per["n"], color=C_NO, width=0.75, label="Число звонков")
    ax1.set_ylabel("Число звонков")
    ax1.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}".replace(",", " "))
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(x, per["rate"], color=C_YES, marker="o", ms=3, lw=1.5, label="Доля отклика")
    ax2.set_ylabel("Доля отклика")
    ax2.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax2.set_ylim(0, 1)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, loc="upper right")

    ax3.plot(x, per["euribor"], color=C_ACCENT, marker="o", ms=3, lw=1.5)
    ax3.set_ylabel("Euribor 3M, %")
    ax3.set_xlabel("Период (год-месяц)")
    ax3.set_xticks(x, per.index, rotation=90)
    ax3.grid(axis="y", color="#e5e5e5", lw=0.6)
    save(fig, "рис_3_5.png")


def tool_version(cmd: list[str]) -> str:
    if shutil.which(cmd[0]) is None:
        return "нет в PATH"
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return res.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return "не удалось определить"


def pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "не установлен"


def main() -> None:
    df = load_raw()
    y = df[TARGET]
    out: list[str] = []
    add = out.append

    # 1. Размер и столбцы
    add(f"1. Размер: {df.shape[0]} строк x {df.shape[1]} столбцов")
    add("  столбец: тип, число уникальных")
    out.extend(wrap_items([f"{c}: {df[c].dtype}, {df[c].nunique()}" for c in df.columns]))

    # 2. Целевая
    vc = y.value_counts().reindex([0, 1])
    add(f"2. Целевая y: no = {vc[0]} ({vc[0] / len(y):.2%}), yes = {vc[1]} ({vc[1] / len(y):.2%})")

    # 3. Дубликаты
    add(f"3. Полных дубликатов строк: {int(df.duplicated().sum())}")

    # 4. unknown
    add("4. Значения 'unknown' (число, доля):")
    cat_cols = [c for c in df.columns if c not in NUMERIC and c != TARGET]
    unk = [
        f"{c} {int((df[c] == 'unknown').sum())} ({(df[c] == 'unknown').mean():.2%})"
        for c in cat_cols
    ]
    out.extend(wrap_items(unk))

    # 5. Специальные значения
    add(
        "5. Доли: pdays = 999 {:.2%}; previous = 0 {:.2%}; poutcome = nonexistent {:.2%}".format(
            (df["pdays"] == 999).mean(),
            (df["previous"] == 0).mean(),
            (df["poutcome"] == "nonexistent").mean(),
        )
    )

    # 6. Числовые
    add("6. Числовые признаки:")
    add(f"  {'признак':<15}{'mean':>10}{'std':>10}{'min':>10}{'median':>10}{'max':>10}")
    for c in NUMERIC:
        s = df[c]
        add(
            f"  {c:<15}{s.mean():>10.3f}{s.std():>10.3f}{s.min():>10.3f}"
            f"{s.median():>10.3f}{s.max():>10.3f}"
        )

    # 7. Доля отклика по категориям
    add("7. Доля отклика (n) по категориям:")
    for c in CATEGORICAL:
        g = df.groupby(c)[TARGET].agg(["mean", "size"])
        order = category_order(c, g["mean"])
        if c not in ("month", "day_of_week", "education"):
            order = order[::-1]  # по убыванию доли отклика
        items = [f"{k} {r['mean']:.1%} ({int(r['size'])})" for k, r in g.reindex(order).iterrows()]
        out.extend(wrap_items(items, indent=f"  {c}:", cont="      "))

    # 8. Спирмен
    corr = df[NUMERIC + [TARGET]].corr(method="spearman")
    feat = corr.loc[NUMERIC, NUMERIC]
    iu = np.triu_indices(len(NUMERIC), k=1)
    pairs = pd.Series(
        feat.to_numpy()[iu],
        index=[f"{NUMERIC[i]} - {NUMERIC[j]}" for i, j in zip(*iu, strict=True)],
    )
    top = pairs.reindex(pairs.abs().sort_values(ascending=False).index)[:10]
    add("8. Спирмен: 10 самых сильных пар (по модулю) среди числовых признаков:")
    out.extend(wrap_items([f"{k} {v:+.3f}" for k, v in top.items()]))
    add("   Корреляция с y:")
    out.extend(wrap_items([f"{c} {corr.loc[c, TARGET]:+.3f}" for c in NUMERIC]))

    # 9. Утечка duration
    auc = roc_auc_score(y, df["duration"])
    zero = df[df["duration"] == 0]
    add(
        f"9. Утечка duration: ROC-AUC одного duration = {auc:.4f}; duration = 0: "
        f"{len(zero)} строк, y = yes у {int(zero[TARGET].sum())}"
    )

    # 10. Время
    dft, ok, note = add_periods(df)
    per = dft.groupby("period", sort=False).agg(
        n=(TARGET, "size"), rate=(TARGET, "mean"), euribor=("euribor3m", "mean")
    )
    out.extend(f"10. Время: {note}".split("\n"))
    add(f"  {'период':<9}{'n':>6}{'отклик':>9}{'euribor3m':>11}")
    for p, r in per.iterrows():
        add(f"  {p:<9}{int(r['n']):>6}{r['rate']:>9.2%}{r['euribor']:>11.3f}")
    cut = int(len(df) * 0.8)
    head, tail = dft.iloc[:cut], dft.iloc[cut:]
    add("  Разбиение по порядку строк 80/20:")
    for name, part in (("первые 80%", head), ("последние 20%", tail)):
        add(
            f"    {name}: {len(part)} строк, {part['period'].iloc[0]}..{part['period'].iloc[-1]}, "
            f"отклик {part[TARGET].mean():.2%}"
        )

    # 11. Версии
    add("11. Версии:")
    vers = [
        f"Python {platform.python_version()}",
        f"uv {tool_version(['uv', '--version']).removeprefix('uv ')}",
        f"git {tool_version(['git', '--version']).removeprefix('git version ')}",
        f"dvc {pkg_version('dvc')}",
    ]
    vers += [
        f"{p} {pkg_version(p)}"
        for p in ("pandas", "scikit-learn", "catboost", "mlflow", "evidently", "matplotlib")
    ]
    out.extend(wrap_items(vers))

    # Рисунки
    fig_balance(df)
    fig_numeric(df)
    fig_categorical(df)
    fig_corr(corr)
    fig_time(per)

    long_lines = [s for s in out if len(s) > WRAP]
    if len(out) > MAX_LINES:
        raise RuntimeError(f"eda_out.txt: {len(out)} строк > {MAX_LINES}")
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    (ASSETS_DIR / "eda_out.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    print(
        f"\n[eda_out.txt: {len(out)} строк; длинных строк: {len(long_lines)}; "
        f"периоды {'по месяцам' if ok else 'блоками по 2000'}]"
    )


if __name__ == "__main__":
    main()

"""Разбиение набора на исходную часть и порции по времени.

Исходная часть — первые месяцы, доля записей которых ближе всего к
split.initial_share (допустимо 55–65 %). Остальные месяцы делятся на
split.n_batches непрерывных порций с границами по смене месяца так, чтобы
максимальная порция была минимальной (перебор всех вариантов границ).

Строки исходного файла копируются без изменений, впереди добавляются
row_id (номер строки в исходном файле, с 1) и period (год-месяц).

Результаты: data/initial/initial.csv, data/incoming/batch_NN.csv,
report_assets/pr2/split_out.txt, рисунок рис_1_1.png.
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from bank_marketing.config import (
    CSV_SEP,
    INCOMING_DIR,
    INITIAL_DIR,
    INITIAL_FILE,
    PARAMS_FILE,
    PR2_DIR,
    PR2_FIGURES_DIR,
    PR2_IMG_DIR,
    RAW_FILE,
    ROOT,
    TARGET,
)
from bank_marketing.data import load_raw
from bank_marketing.periods import periods_ok, restore_periods
from bank_marketing.plotting import (
    C_MEAN,
    CM,
    WIDTH,
    apply_style,
    pct_formatter,
    save,
    thousands_formatter,
)
from bank_marketing.report import emit

SHARE_MIN, SHARE_MAX = 0.55, 0.65
EOL = "\r\n"  # как в исходном файле
PART_COLORS = ["#7f93a8", "#3c5a78", "#5f9e6e", "#c9a227", "#8e6bb0"]

apply_style()


def initial_months(counts: pd.Series, share: float) -> int:
    """Число первых месяцев, доля записей которых ближе всего к share."""
    cum = counts.cumsum().to_numpy() / counts.sum()
    k = int(np.argmin(np.abs(cum - share))) + 1
    if not SHARE_MIN <= cum[k - 1] <= SHARE_MAX:
        raise ValueError(
            f"доля исходной части {cum[k - 1]:.2%} вне {SHARE_MIN:.0%}–{SHARE_MAX:.0%}"
        )
    return k


def batch_bounds(counts: np.ndarray, n: int) -> tuple[int, ...]:
    """Индексы начала порций 2…n: минимум максимальной порции, затем минимум размаха.

    Перебираются все сочетания n − 1 границ между соседними месяцами.
    """
    best, best_key = (), None
    for cuts in itertools.combinations(range(1, len(counts)), n - 1):
        sizes = [int(s.sum()) for s in np.split(counts, cuts)]
        key = (max(sizes), max(sizes) - min(sizes))
        if best_key is None or key < best_key:
            best, best_key = cuts, key
    return best


def assign_parts(period: pd.Series, share: float, n_batches: int) -> pd.Series:
    """Имя части для каждой строки: initial, batch_01, …"""
    counts = period.value_counts(sort=False).reindex(period.unique())
    k = initial_months(counts, share)
    rest = counts.iloc[k:]
    cuts = batch_bounds(rest.to_numpy(), n_batches)
    names = ["initial"] * k
    for i, chunk in enumerate(np.split(np.arange(len(rest)), cuts), start=1):
        names += [f"batch_{i:02d}"] * len(chunk)
    return period.map(dict(zip(counts.index, names, strict=True)))


def part_title(name: str) -> str:
    return "Исходная часть" if name == "initial" else f"Порция {int(name[-2:])}"


def write_parts(lines: list[str], header: str, period: pd.Series, part: pd.Series) -> dict:
    """Пишет файлы частей: строки исходного файла с префиксом row_id;"period";."""
    INITIAL_DIR.mkdir(parents=True, exist_ok=True)
    INCOMING_DIR.mkdir(parents=True, exist_ok=True)
    for old in INCOMING_DIR.glob("batch_*.csv"):
        old.unlink()
    files = {}
    for name in part.unique():
        path = INITIAL_FILE if name == "initial" else INCOMING_DIR / f"{name}.csv"
        idx = np.flatnonzero(part.to_numpy() == name)
        body = [f'{i + 1};"{period.iat[i]}";{lines[i]}' for i in idx]
        text = EOL.join([f'"row_id";"period";{header}', *body]) + EOL
        path.write_bytes(text.encode("ascii"))
        files[name] = path
    return files


def check_parts(files: dict, raw: pd.DataFrame) -> list[str]:
    """Сумма строк, порядок row_id, отсутствие пересечений, совпадение значений с исходными."""
    parts = [pd.read_csv(path, sep=CSV_SEP) for path in files.values()]
    joined = pd.concat(parts, ignore_index=True)
    ids = joined["row_id"]
    checks = {
        f"сумма строк = {len(raw)}": len(joined) == len(raw),
        "row_id возрастает внутри и между частями": bool(ids.is_monotonic_increasing),
        "пересечений нет (row_id уникальны)": bool(ids.is_unique),
        "row_id = 1…N без пропусков": ids.tolist() == list(range(1, len(raw) + 1)),
        "значения совпадают с исходными": joined.drop(columns=["row_id", "period"]).equals(raw),
    }
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise AssertionError("проверки разбиения не прошли: " + "; ".join(failed))
    return [f"  [ok] {k}" for k in checks]


def summary(df: pd.DataFrame, files: dict) -> pd.DataFrame:
    g = df.groupby("part", sort=False)
    out = pd.DataFrame(
        {
            "file": [files[p].relative_to(ROOT).as_posix() for p in g.groups],
            "first": g["period"].first(),
            "last": g["period"].last(),
            "n": g.size(),
            "yes": g[TARGET].sum(),
            "euribor": g["euribor3m"].mean(),
        }
    )
    out["share"] = out["n"] / len(df)
    out["yes_rate"] = out["yes"] / out["n"]
    return out


def fig_split(df: pd.DataFrame) -> None:
    per = df.groupby("period", sort=False).agg(
        n=(TARGET, "size"), rate=(TARGET, "mean"), part=("part", "first")
    )
    parts = list(per["part"].unique())
    color = dict(zip(parts, PART_COLORS, strict=True))
    x = np.arange(len(per))

    fig, ax = plt.subplots(figsize=(WIDTH, 9 * CM), layout="constrained")
    ax.bar(x, per["n"], color=per["part"].map(color), width=0.75)
    ax.set_ylabel("Число звонков")
    ax.yaxis.set_major_formatter(thousands_formatter())
    ax.set_xticks(x, per.index, rotation=90)
    ax.set_xlabel("Период (год-месяц)")
    ax.set_xlim(-0.6, len(per) - 0.4)
    ax.set_ylim(0, per["n"].max() * 1.08)

    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(x, per["rate"], color=C_MEAN, marker="o", ms=3, lw=1.4)
    ax2.set_ylabel("Доля отклика")
    ax2.yaxis.set_major_formatter(pct_formatter(0))
    ax2.set_ylim(0, 1)

    # Границы и подписи частей над областью графика.
    starts = [int(np.flatnonzero(per["part"].to_numpy() == p)[0]) for p in parts]
    ends = [*starts[1:], len(per)]
    for i, (p, s, e) in enumerate(zip(parts, starts, ends, strict=True)):
        if i:
            ax.axvline(s - 0.5, color="#333333", ls="--", lw=0.9)
        # Узкие части подписаны в две строки, части из 1–2 месяцев — коротко: «П1» = порция 1.
        label = part_title(p) if e - s > 5 else part_title(p).replace(" ", "\n")
        if e - s < 3:
            label = f"П{int(p[-2:])}"
        ax.text(
            (s + e - 1) / 2,
            1.01,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    handles = [Patch(color=color[p], label=part_title(p)) for p in parts]
    handles.append(Line2D([], [], color=C_MEAN, marker="o", ms=3, lw=1.4, label="Доля отклика"))
    ax.legend(handles=handles, frameon=False, loc="upper right", ncols=2)
    save(fig, "рис_1_1.png", (PR2_IMG_DIR, PR2_FIGURES_DIR))


def main() -> None:
    params = yaml.safe_load(PARAMS_FILE.read_text(encoding="utf-8"))["split"]
    raw = pd.read_csv(RAW_FILE, sep=CSV_SEP)
    text = RAW_FILE.read_bytes().decode("ascii")
    header, *lines = text.splitlines()
    if len(lines) != len(raw):
        raise ValueError(f"строк в файле {len(lines)}, в таблице {len(raw)}")

    period = restore_periods(raw["month"])
    ok, runs = periods_ok(period)
    if not ok:
        raise ValueError(f"периоды не монотонны (блоков {runs}): разбиение по месяцам невозможно")
    part = assign_parts(period, params["initial_share"], params["n_batches"])
    files = write_parts(lines, header, period, part)
    checks = check_parts(files, raw)

    df = load_raw().assign(period=period, part=part)
    tab = summary(df, files)
    out = [
        f"Разбиение {RAW_FILE.name}: {len(df)} строк, {period.nunique()} месяцев "
        f"({period.iloc[0]}…{period.iloc[-1]})",
        f"Параметры: initial_share = {params['initial_share']}, n_batches = {params['n_batches']}",
        "",
        f"{'часть':<15}{'файл':<29}{'период':<18}{'строк':>7}{'% набора':>10}"
        f"{'yes':>6}{'доля yes':>10}{'euribor3m':>11}",
    ]
    for name, r in tab.iterrows():
        out.append(
            f"{part_title(name):<15}{r['file']:<29}{r['first'] + '…' + r['last']:<18}"
            f"{r['n']:>7}{r['share']:>10.2%}{r['yes']:>6}{r['yes_rate']:>10.2%}"
            f"{r['euribor']:>11.3f}"
        )
    out.append(
        f"{'итого':<15}{'':<29}{'':<18}{tab['n'].sum():>7}{tab['share'].sum():>10.2%}"
        f"{tab['yes'].sum():>6}{tab['yes'].sum() / tab['n'].sum():>10.2%}"
    )
    out += ["", "Месяцы по частям:"]
    for name, months in df.groupby("part", sort=False)["period"].unique().items():
        out.append(f"  {part_title(name)}: {', '.join(months)}")
    out += ["", "Проверки:", *checks]

    fig_split(df)
    emit(out, PR2_DIR / "split_out.txt")


if __name__ == "__main__":
    main()

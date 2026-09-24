"""Схемы для раздела 1 отчёта: жизненный цикл модели и конвейер проекта.

Координаты задаются в сантиметрах: ось занимает всю фигуру шириной 16 см,
поэтому размер шрифта в pt соответствует размеру на странице отчёта.
После отрисовки проверяется, что каждый текст помещается в свой блок.
"""

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from bank_marketing.config import ASSETS_IMG_DIR

CM = 1 / 2.54
PT_CM = 2.54 / 72
WIDTH_CM = 16
DPI = 200
LINESPACING = 1.25
PAD_CM = 0.12
EDGE = "#4a4a4a"
ARROW = "#3d3d3d"
RED = "#a33a3a"
DASH = (0, (4, 2))

PALETTE = {
    "business": "#e8eef6",
    "data": "#e3eef7",
    "prep": "#e6f1ea",
    "ml": "#e9f0e0",
    "ops": "#f6ede0",
    "monitor": "#f5e3e3",
    "base": "#ececec",
}

plt.rcParams.update({"font.family": "DejaVu Sans"})


def canvas(height_cm: float):
    fig = plt.figure(figsize=(WIDTH_CM * CM, height_cm * CM), dpi=DPI, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH_CM)
    ax.set_ylim(0, height_cm)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.fit_checks = []
    return fig, ax


def box(ax, cx, cy, w, h, title, body="", color="#ffffff", size=9.5):
    """Блок: жирный заголовок и пояснение, текст центрирован по вертикали."""
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2),
        w,
        h,
        boxstyle="round,pad=0,rounding_size=0.25",
        facecolor=color,
        edgecolor=EDGE,
        linewidth=0.9,
    )
    ax.add_patch(patch)
    lh = size * LINESPACING * PT_CM
    gap = 0.18 if body else 0.0
    nt = title.count("\n") + 1
    nb = body.count("\n") + 1 if body else 0
    top = cy + (nt * lh + gap + nb * lh) / 2
    texts = [
        ax.text(
            cx,
            top,
            title,
            ha="center",
            va="top",
            fontweight="bold",
            fontsize=size,
            linespacing=LINESPACING,
        )
    ]
    if body:
        texts.append(
            ax.text(
                cx,
                top - nt * lh - gap,
                body,
                ha="center",
                va="top",
                fontsize=size,
                linespacing=LINESPACING,
                color="#222222",
            )
        )
    ax.fit_checks.append(((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), texts))
    return patch


def arrow(ax, a, b, rad=0.0, pa=None, pb=None, color=ARROW, lw=1.1, ls="-"):
    arr = FancyArrowPatch(
        a,
        b,
        patchA=pa,
        patchB=pb,
        arrowstyle="-|>",
        mutation_scale=11,
        connectionstyle=f"arc3,rad={rad}",
        color=color,
        linewidth=lw,
        linestyle=ls,
        shrinkA=3,
        shrinkB=3,
    )
    ax.add_patch(arr)
    return arr


def check_fit(fig, ax, name: str) -> None:
    """Падает, если текст выходит за блок с отступом PAD_CM."""
    fig.canvas.draw()
    inv = ax.transData.inverted()
    for (x0, y0, x1, y1), texts in ax.fit_checks:
        for t in texts:
            bb = t.get_window_extent().transformed(inv)
            if (
                bb.x0 < x0 + PAD_CM
                or bb.x1 > x1 - PAD_CM
                or bb.y0 < y0 + PAD_CM
                or bb.y1 > y1 - PAD_CM
            ):
                raise RuntimeError(f"{name}: текст «{t.get_text()!r}» не помещается в блок")


def lifecycle():
    """рис_1_1: замкнутый цикл из шести этапов и обратная связь 6 → 4."""
    fig, ax = canvas(13.4)
    cx, cy, rx, ry = 8.0, 6.7, 5.3, 5.2
    w, h = 5.2, 2.4
    # Этапы по часовой стрелке, начиная сверху.
    stages = [
        ("1. Бизнес-постановка", "конверсия звонков,\nстоимость привлечения", "business"),
        ("2. Сбор и поставка\nданных", "CRM, итоги звонков,\nмакропоказатели", "data"),
        ("3. Подготовка данных\nи EDA", "", "prep"),
        ("4. Обучение\nи валидация", "временное разбиение,\nPR-AUC, lift@k", "ml"),
        ("5. Эксплуатация", "пакетный скоринг\nперед кампанией →\nсписок обзвона", "ops"),
        ("6. Мониторинг", "дрейф признаков\nи отклика", "monitor"),
    ]
    angles = [90, 30, -30, -90, -150, 150]

    centers, patches = [], []
    for (title, body, key), ang in zip(stages, angles, strict=True):
        x = cx + rx * math.cos(math.radians(ang))
        y = cy + ry * math.sin(math.radians(ang))
        centers.append((x, y))
        patches.append(box(ax, x, y, w, h, title, body, PALETTE[key]))

    for i in range(6):
        j = (i + 1) % 6
        arrow(ax, centers[i], centers[j], rad=-0.18, pa=patches[i], pb=patches[j])

    # Обратная связь: мониторинг → обучение, дугой через центр цикла.
    arrow(
        ax,
        centers[5],
        centers[3],
        rad=-0.32,
        pa=patches[5],
        pb=patches[3],
        color=RED,
        lw=1.3,
        ls=DASH,
    )
    ax.text(
        7.55,
        6.5,
        "переобучение\nпри дрейфе",
        ha="left",
        va="center",
        color=RED,
        fontsize=9.5,
        fontstyle="italic",
        linespacing=1.2,
    )
    return fig, ax


def pipeline():
    """рис_1_2: конвейер проекта в два ряда, мониторинг и полоса-основание."""
    fig, ax = canvas(11.8)
    w, h = 3.4, 2.9
    size = 9
    xs = [1.9, 5.85, 9.8, 13.75]
    y1, y2, y3 = 9.95, 6.05, 3.0

    row1 = [
        ("Источник", "UCI CSV", "data"),
        ("DVC", "версионирование\nданных", "data"),
        ("Предобработка\nи признаки", "pandas,\nscikit-learn", "prep"),
        ("Обучение\n(CatBoost)", "+ трекинг\nэкспериментов\n(MLflow)", "ml"),
    ]
    row2 = [
        ("Реестр\nмоделей", "MLflow", "ml"),
        ("Пакетный\nскоринг", "клиентской\nбазы", "ops"),
        ("Витрина", "список обзвона,\nParquet", "ops"),
        ("Колл-центр /\nCRM", "", "ops"),
    ]
    p1 = [
        box(ax, x, y1, w, h, t, b, PALETTE[k], size) for x, (t, b, k) in zip(xs, row1, strict=True)
    ]
    p2 = [
        box(ax, x, y2, w, h, t, b, PALETTE[k], size) for x, (t, b, k) in zip(xs, row2, strict=True)
    ]

    for row, y in ((p1, y1), (p2, y2)):
        for i in range(3):
            arrow(ax, (xs[i], y), (xs[i + 1], y), pa=row[i], pb=row[i + 1])

    # Переход с конца первого ряда на начало второго (ломаная).
    ymid = (y1 + y2) / 2
    ax.plot([xs[3], xs[3], xs[0]], [y1 - h / 2 - 0.1, ymid, ymid], color=ARROW, lw=1.1)
    arrow(ax, (xs[0], ymid + 0.1), (xs[0], y2 + h / 2 + 0.02))

    # Мониторинг: наблюдает за скорингом и витриной.
    mx, mw, mh = (xs[1] + xs[2]) / 2, 4.2, 1.5
    box(ax, mx, y3, mw, mh, "Мониторинг", "Evidently", PALETTE["monitor"], size)
    for x, dx in ((xs[1], -0.9), (xs[2], 0.9)):
        arrow(ax, (x, y2 - h / 2), (mx + dx, y3 + mh / 2), color="#777777", lw=0.9)

    # Обратная связь: вдоль правого края вверх к блоку «Обучение».
    xr = WIDTH_CM - 0.25
    ax.plot([mx + mw / 2 + 0.05, xr, xr], [y3, y3, y1], color=RED, lw=1.3, ls=DASH)
    arrow(ax, (xr + 0.1, y1), (xs[3] + w / 2 + 0.02, y1), color=RED, lw=1.3)
    ax.text(
        xs[3],
        y3 + 0.15,
        "дрейф →\nпереобучение",
        ha="center",
        va="bottom",
        color=RED,
        fontsize=size,
        fontstyle="italic",
        linespacing=1.2,
    )

    # Полоса-основание.
    ax.add_patch(
        FancyBboxPatch(
            (0.2, 0.3),
            WIDTH_CM - 0.4,
            0.95,
            boxstyle="round,pad=0,rounding_size=0.15",
            facecolor=PALETTE["base"],
            edgecolor=EDGE,
            linewidth=0.9,
        )
    )
    ax.text(
        WIDTH_CM / 2,
        0.775,
        "Код: Git/GitHub  ·  Окружение: uv, uv.lock  ·  Контейнер: Docker",
        ha="center",
        va="center",
        fontsize=9.5,
    )
    return fig, ax


def main() -> None:
    ASSETS_IMG_DIR.mkdir(parents=True, exist_ok=True)
    for name, build in (("рис_1_1.png", lifecycle), ("рис_1_2.png", pipeline)):
        fig, ax = build()
        check_fit(fig, ax, name)
        out = ASSETS_IMG_DIR / name
        fig.savefig(out, dpi=DPI, facecolor="white")
        plt.close(fig)
        print(f"Сохранено: {out}")


if __name__ == "__main__":
    main()

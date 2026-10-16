"""Общий стиль рисунков отчёта: размеры, цвета, десятичная запятая, сохранение."""

from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter

CM = 1 / 2.54
DPI = 200
WIDTH = 16 * CM

C_NO, C_YES, C_ACCENT, C_MEAN = "#7f93a8", "#d9822b", "#3c5a78", "#b03a2e"
LABEL_BOX = {"boxstyle": "square,pad=0.1", "facecolor": "white", "edgecolor": "none", "alpha": 0.85}


def apply_style(font_size: float = 9, tick_size: float = 9) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": tick_size,
            "ytick.labelsize": tick_size,
            "legend.fontsize": tick_size,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save(fig, name: str, folders: Iterable[Path], tight: bool = True) -> None:
    """Сохраняет рисунок в каждую папку (PNG, белый фон) и закрывает его."""
    extra = {"bbox_inches": "tight"} if tight else {}
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / name, dpi=DPI, facecolor="white", **extra)
    plt.close(fig)


def num(v: float, decimals: int = 2) -> str:
    """Число с десятичной запятой и пробелом между разрядами."""
    return f"{v:,.{decimals}f}".replace(",", " ").replace(".", ",").replace("-", "−")


def pct(v: float, decimals: int = 1) -> str:
    """Доля 0…1 в процентах с десятичной запятой."""
    return num(v * 100, decimals) + "%"


class CommaFormatter(ScalarFormatter):
    """Стандартные подписи делений matplotlib, но с десятичной запятой."""

    def __call__(self, x, pos=None):
        return super().__call__(x, pos).replace(".", ",")


def comma_axes(*axes, x: bool = True, y: bool = True) -> None:
    """Ставит десятичную запятую на линейных осях."""
    for ax in axes:
        if x and ax.get_xscale() == "linear":
            ax.xaxis.set_major_formatter(CommaFormatter(useOffset=False))
        if y and ax.get_yscale() == "linear":
            ax.yaxis.set_major_formatter(CommaFormatter(useOffset=False))


def pct_formatter(decimals: int = 0) -> FuncFormatter:
    return FuncFormatter(lambda v, _: pct(v, decimals))


def thousands_formatter() -> FuncFormatter:
    return FuncFormatter(lambda v, _: num(v, 0))

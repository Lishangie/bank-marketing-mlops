"""Ключевые показатели без модели по исходной части и порциям.

Число записей, доля отклика по меткам, доля unknown в default, доля записей
с euribor3m вне диапазона [min; max] исходной части. Показатели, требующие модели,
только перечисляются (считаются с ПР3).

Результат: report_assets/pr2/kpi_out.txt.
"""

from bank_marketing.config import PR2_DIR, TARGET
from bank_marketing.data import load_parts
from bank_marketing.report import emit
from bank_marketing.stats import out_of_range_share

MODEL_KPI = [
    "ожидаемое число вкладов в топ-k = сумма прогнозных вероятностей k лучших клиентов волны",
    "Lift@k = доля отклика в топ-k / доля отклика по всей волне",
    "доля клиентов выше порога = доля клиентов волны с вероятностью ≥ порога обзвона",
]


def main() -> None:
    parts = load_parts()
    init = parts["initial"]
    lo, hi = init["euribor3m"].min(), init["euribor3m"].max()
    out = [
        "Ключевые показатели без модели",
        "",
        f"{'часть':<11}{'строк':>7}{'yes':>6}{'доля отклика':>14}{'unknown default':>17}"
        f"{'euribor3m вне диапазона':>25}",
    ]
    for name, d in parts.items():
        out.append(
            f"{name:<11}{len(d):>7}{int(d[TARGET].sum()):>6}{d[TARGET].mean():>14.2%}"
            f"{(d['default'] == 'unknown').mean():>17.2%}"
            f"{out_of_range_share(d['euribor3m'], lo, hi):>25.2%}"
        )
    out += [
        "",
        f"Диапазон euribor3m исходной части: [{lo:g}; {hi:g}]; доля вне него > 5 % —",
        "модель вне области обучения (требование D-08).",
        "",
        "Показатели с моделью (определены, не считаются до ПР3):",
        *[f"  - {k}" for k in MODEL_KPI],
    ]
    emit(out, PR2_DIR / "kpi_out.txt")


if __name__ == "__main__":
    main()

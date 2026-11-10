"""Показатели порции и правила мониторинга (таблица правил — отчёт ПР5, п. 1.2).

Уровни: предметная область (ключевые показатели ПР2), данные (проверки
D-01…D-09, пропуски, PSI, область обучения), модель (по пришедшим меткам),
техническое состояние (статус и время обработки). Пороги — params.yaml →
evaluate.monitor (ПР4) и service.
"""

import numpy as np
import pandas as pd

from bank_marketing.config import CATEGORICAL
from bank_marketing.evaluation import metrics as quality_metrics
from bank_marketing.experiments import top_k_index
from bank_marketing.stats import psi_from_shares
from bank_marketing.validation import OK


def num(x: float, digits: int = 2) -> str:
    """Число с десятичной запятой для сообщений."""
    return f"{x:.{digits}f}".replace(".", ",")


def pp(x: float) -> str:
    return f"{x * 100:+.1f}".replace(".", ",") + " п. п."


def batch_metrics(
    model,
    X: pd.DataFrame,
    y: pd.Series | None,
    proba: np.ndarray,
    card: dict,
    results,
    rows_in: int,
    k: float,
) -> dict:
    """Показатели порции. X — после prepare(), proba — оценки действующей модели."""
    prof = card["profile"]
    lo, hi = prof["euribor_range"]
    top = top_k_index(proba, k)
    m = {
        # предметная область (ключевые показатели ПР2)
        "call_list_size": len(top),
        "expected_deposits": float(proba[top].sum()),
        "above_threshold_share": float((proba >= card["threshold"]).mean()),
        "out_of_domain_share": float((~X["euribor3m"].between(lo, hi)).mean()),
        # данные
        "rows_in": rows_in,
        "rows_saved": len(X),
        "unknown_share": float((X[CATEGORICAL] == "unknown").to_numpy().mean()),
        "quality_warnings": sum(r.status != OK for r in results),
        "pred_mean": float(proba.mean()),
    }
    Z = model.named_steps["features"].transform(X)
    for f, ref in prof["psi"].items():
        m[f"psi_{f}"] = psi_from_shares(ref["shares"], Z[f], ref["bins"])
    # модель — когда известны фактические значения
    if y is not None and 0 < y.sum() < len(y):
        q = quality_metrics(y, proba, k)
        m.update(
            response_rate=float(y.mean()),
            pr_auc=q["pr_auc"],
            roc_auc=q["roc_auc"],
            lift_k=q["lift_k"],
            calib_pp=float(proba.mean() - y.mean()),
            deposits_in_call_list=int(np.asarray(y)[top].sum()),
        )
    return m


def check_rules(m: dict, prev: dict | None, results, P: dict) -> list[dict]:
    """Предупреждения по порогам; retrain = 1 — сигнал к переобучению."""
    mon, srv = P["evaluate"]["monitor"], P["service"]
    out = []

    def add(rule, level, message, retrain=0):
        out.append({"rule": rule, "level": level, "message": message, "retrain": retrain})

    # предметная область: изменение относительно предыдущей порции
    if prev:
        for key, name in [
            ("above_threshold_share", "доля клиентов выше порога"),
            ("response_rate", "доля отклика"),
        ]:
            if key in m and key in prev and abs(m[key] - prev[key]) > srv["kpi_change_pp"]:
                add(
                    f"K-{key}",
                    "warning",
                    f"{name.capitalize()} изменилась на "
                    f"{pp(m[key] - prev[key])} относительно предыдущей порции",
                )
    # данные
    bad = [r for r in results if r.status != OK]
    if bad:
        add(
            "Q",
            "warning",
            "Проверка качества: "
            + ", ".join(r.rule for r in bad)
            + " (подробности — таблица quality)",
        )
    if m["out_of_domain_share"] > srv["out_of_domain_max"]:
        add(
            "OOD",
            "retrain",
            f"Ставка вне области обучения модели у "
            f"{num(m['out_of_domain_share'] * 100, 1)} % записей",
            1,
        )
    psi = {k[4:]: v for k, v in m.items() if k.startswith("psi_")}
    alert = {f: v for f, v in psi.items() if v > mon["psi_alert"]}
    warn = {f: v for f, v in psi.items() if mon["psi_warn"] < v <= mon["psi_alert"]}
    if alert:
        add(
            "PSI",
            "retrain",
            "Существенный дрейф: " + "; ".join(f"{f} PSI = {num(v)}" for f, v in alert.items()),
            1,
        )
    if warn:
        add(
            "PSI-warn",
            "warning",
            "Умеренный дрейф: " + "; ".join(f"{f} PSI = {num(v)}" for f, v in warn.items()),
        )
    # модель
    if "lift_k" in m and m["lift_k"] < mon["lift_k"]:
        add("LIFT", "retrain", f"Lift@10 % = {num(m['lift_k'])} ниже {num(mon['lift_k'], 1)}", 1)
    if "calib_pp" in m and abs(m["calib_pp"]) > mon["calib_pp"]:
        add(
            "CALIB",
            "retrain",
            f"Прогноз доли отклика отличается от факта на {pp(m['calib_pp'])}",
            1,
        )
    return out

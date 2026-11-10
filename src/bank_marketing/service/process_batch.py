"""Сценарий обновления: одна команда на порцию.

    uv run python -m bank_marketing.service.process_batch data/incoming/batch_01.csv

Цепочка: загрузка → проверки D-01…D-09 (ПР4) → сохранение исходных данных →
очистка и пайплайн действующей модели (ПР3) → предсказания с версией модели и
список обзвона → показатели порции → правила мониторинга → журнал. Порция с
критическим нарушением (D-01, D-02) не обрабатывается, причина пишется в журнал.
Повторный запуск на той же порции перезаписывает её строки (INSERT OR REPLACE).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from bank_marketing.config import CSV_SEP, INITIAL_FILE, TARGET
from bank_marketing.features import clean, load_reference, prepare
from bank_marketing.service import registry, storage
from bank_marketing.service.monitoring import batch_metrics, check_rules, num
from bank_marketing.validation import CRITICAL, RULES, validate


def ensure_registry() -> None:
    """При первом запуске — карточка v1 и указатель current.json."""
    if registry.card_path("v1").exists() and registry.CURRENT_FILE.exists():
        return
    X, y = prepare(INITIAL_FILE)
    train = X["period"] < registry.params()["train"]["test_from"]
    registry.bootstrap_v1(X[train].reset_index(drop=True), y[train].reset_index(drop=True))


def previous_metrics(con, batch: str) -> dict | None:
    """Показатели порции, обработанной перед этой (для изменения на панели)."""
    order = storage.processed_batches(con)
    before = order[: order.index(batch)] if batch in order else order
    hist = storage.history(con)
    if not before or hist.empty or before[-1] not in hist.index:
        return None
    return hist.loc[before[-1]].dropna().to_dict()


def process_batch(path: str | Path) -> dict:
    t0, started = time.perf_counter(), storage.now()
    path = Path(path)
    batch = path.stem
    P = registry.params()
    k = P["train"]["top_k"]
    ensure_registry()
    model, card = registry.load()
    version = card["version"]
    con = storage.connect()
    try:
        raw = pd.read_csv(path, sep=CSV_SEP, dtype=str)
        ref = load_reference()
        last = storage.last_period_before(con, batch, ref["last_period"])
        df, results = validate(raw, ref, last_period=last)
        storage.save_quality(con, batch, results)
        if df is None:
            reasons = "; ".join(
                f"{r.rule} {RULES[r.rule]} ({r.details})" for r in results if r.status == CRITICAL
            )
            storage.log_run(
                con,
                batch,
                started,
                "rejected",
                len(raw),
                0,
                version,
                time.perf_counter() - t0,
                reasons,
            )
            con.commit()
            return {"batch": batch, "status": "rejected", "rows": len(raw), "reasons": reasons}

        storage.save_raw(con, batch, df)
        X = clean(df).sort_values("row_id").reset_index(drop=True)
        labels = X.pop(TARGET)
        y = (labels == "yes").astype(int) if labels.notna().all() else None
        proba = model.predict_proba(X)[:, 1]

        rank = pd.Series(-proba).rank(method="first").astype(int).to_numpy()
        n_call = max(1, int(round(len(X) * k)))
        preds = pd.DataFrame(
            {
                "row_id": X["row_id"].astype(int),
                "probability": proba.round(6),
                "rank": rank,
                "in_call_list": (rank <= n_call).astype(int),
                "above_threshold": (proba >= card["threshold"]).astype(int),
                "model_version": version,
                "scored_at": storage.now(),
            }
        )
        storage.save_predictions(con, batch, preds)
        call_dir = storage.db_path().parent / "call_lists"  # рядом с базой
        call_dir.mkdir(parents=True, exist_ok=True)
        preds[preds["in_call_list"] == 1].sort_values("rank")[
            ["row_id", "probability", "rank", "model_version"]
        ].to_csv(call_dir / f"{batch}.csv", index=False)  # список обзвона для CRM

        m = batch_metrics(model, X, y, proba, card, results, len(raw), k)
        alerts = check_rules(m, previous_metrics(con, batch), results, P)
        m["duration_s"] = time.perf_counter() - t0
        if m["duration_s"] > P["service"]["max_batch_seconds"]:
            alerts.append(
                {
                    "rule": "TECH",
                    "level": "warning",
                    "retrain": 0,
                    "message": f"Обработка заняла {num(m['duration_s'], 1)} с",
                }
            )
        m["retrain_signal"] = int(any(a["retrain"] for a in alerts))
        m["alerts"] = len(alerts)
        storage.save_metrics(con, batch, m, version)
        storage.save_alerts(con, batch, alerts)
        message = f"предупреждений {len(alerts)}; сигнал к переобучению: " + (
            "да" if m["retrain_signal"] else "нет"
        )
        storage.log_run(
            con, batch, started, "ok", len(raw), len(X), version, time.perf_counter() - t0, message
        )
        con.commit()
    finally:
        con.close()
    out = {"batch": batch, "status": "ok", "model_version": version, "rows": len(X)}
    out.update(
        {
            key: (round(v, 4) if isinstance(v, float) else v)
            for key, v in m.items()
            if not isinstance(v, np.ndarray)
        }
    )
    out["alerts"] = [a["message"] for a in alerts]
    return out


def main(argv: list[str]) -> None:
    if len(argv) != 2:
        sys.exit("использование: python -m bank_marketing.service.process_batch <порция.csv>")
    say(json.dumps(process_batch(argv[1]), ensure_ascii=False, indent=1))


def say(text: str) -> None:
    """Печать, не падающая на кодировке консоли Windows."""
    enc = sys.stdout.encoding or "utf-8"
    print(text.encode(enc, errors="replace").decode(enc))


if __name__ == "__main__":
    main(sys.argv)

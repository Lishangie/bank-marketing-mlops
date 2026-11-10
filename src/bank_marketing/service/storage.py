"""Хранилище системы (SQLite): исходные данные порций, предсказания с версией
модели, история показателей, отчёты о проверке, предупреждения, журнал запусков
и переобучений.

Идемпотентность: у таблиц порции первичный ключ «порция + row_id» (или
«порция + показатель»), запись — INSERT OR REPLACE; повторная обработка той же
порции перезаписывает её строки, а не добавляет новые.
"""

import os
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from bank_marketing.config import PARAMS_FILE, RAW_COLUMNS, ROOT

INT_COLUMNS = ["row_id", "age", "duration", "campaign", "pdays", "previous"]
FLOAT_COLUMNS = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
DATA_COLUMNS = ["row_id", "period", *RAW_COLUMNS]


def _plain(col: str) -> str:
    return col.replace(".", "_")


def _sql_name(col: str) -> str:
    """Имя столбца в SQL: точки → «_», в кавычках (default — служебное слово)."""
    return f'"{_plain(col)}"'


def _sql_type(col: str) -> str:
    if col in INT_COLUMNS:
        return "INTEGER"
    if col in FLOAT_COLUMNS:
        return "REAL"
    return "TEXT"


SCHEMA = f"""
CREATE TABLE IF NOT EXISTS raw_data (
    batch TEXT NOT NULL,
    {", ".join(f"{_sql_name(c)} {_sql_type(c)}" for c in DATA_COLUMNS)},
    PRIMARY KEY (batch, row_id));
CREATE TABLE IF NOT EXISTS predictions (
    batch TEXT NOT NULL, row_id INTEGER NOT NULL, probability REAL, rank INTEGER,
    in_call_list INTEGER, above_threshold INTEGER, model_version TEXT, scored_at TEXT,
    PRIMARY KEY (batch, row_id));
CREATE TABLE IF NOT EXISTS metrics_history (
    batch TEXT NOT NULL, metric TEXT NOT NULL, value REAL, model_version TEXT,
    processed_at TEXT, PRIMARY KEY (batch, metric));
CREATE TABLE IF NOT EXISTS quality (
    batch TEXT NOT NULL, rule TEXT NOT NULL, status TEXT, violations INTEGER, base INTEGER,
    details TEXT, PRIMARY KEY (batch, rule));
CREATE TABLE IF NOT EXISTS alerts (
    batch TEXT NOT NULL, rule TEXT NOT NULL, level TEXT, message TEXT, retrain INTEGER,
    PRIMARY KEY (batch, rule));
CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT, batch TEXT, started_at TEXT, finished_at TEXT,
    status TEXT, rows_in INTEGER, rows_saved INTEGER, model_version TEXT, duration_s REAL,
    message TEXT);
CREATE TABLE IF NOT EXISTS retrains (
    batch TEXT PRIMARY KEY, started_at TEXT, current_version TEXT, candidate TEXT,
    train_rows INTEGER, holdout_rows INTEGER,
    cur_pr_auc REAL, cand_pr_auc REAL, cur_roc_auc REAL, cand_roc_auc REAL,
    cur_lift_k REAL, cand_lift_k REAL, cur_calib REAL, cand_calib REAL,
    decision TEXT, new_version TEXT, duration_s REAL);
"""


def db_path() -> Path:
    """Путь к базе: переменная BANK_DB (панель на снимке базы) или params.yaml."""
    if os.environ.get("BANK_DB"):
        return Path(os.environ["BANK_DB"])
    params = yaml.safe_load(PARAMS_FILE.read_text(encoding="utf-8"))
    return ROOT / params["service"]["db"]


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = Path(path or db_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    return con


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _replace(con: sqlite3.Connection, table: str, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0])
    sql = (
        f"INSERT OR REPLACE INTO {table} ({', '.join(_sql_name(c) for c in cols)}) "
        f"VALUES ({', '.join('?' * len(cols))})"
    )
    con.executemany(sql, [tuple(r[c] for c in cols) for r in rows])


def save_raw(con, batch: str, df: pd.DataFrame) -> None:
    """Принятые записи порции после проверки (до очистки), y — yes/no."""
    data = df[DATA_COLUMNS].astype(object).where(df[DATA_COLUMNS].notna(), None)
    rows = [{"batch": batch, **r} for r in data.to_dict("records")]
    _replace(con, "raw_data", rows)


def save_predictions(con, batch: str, frame: pd.DataFrame) -> None:
    _replace(con, "predictions", [{"batch": batch, **r} for r in frame.to_dict("records")])


def save_metrics(con, batch: str, metrics: dict, version: str) -> None:
    ts = now()
    rows = [
        {
            "batch": batch,
            "metric": k,
            "value": None if pd.isna(v) else float(v),
            "model_version": version,
            "processed_at": ts,
        }
        for k, v in metrics.items()
    ]
    _replace(con, "metrics_history", rows)


def save_quality(con, batch: str, results) -> None:
    con.execute("DELETE FROM quality WHERE batch = ?", (batch,))
    rows = [
        {
            "batch": batch,
            "rule": r.rule,
            "status": r.status,
            "violations": r.violations,
            "base": r.base,
            "details": r.details,
        }
        for r in results
    ]
    _replace(con, "quality", rows)


def save_alerts(con, batch: str, alerts: list[dict]) -> None:
    con.execute("DELETE FROM alerts WHERE batch = ?", (batch,))
    _replace(con, "alerts", [{"batch": batch, **a} for a in alerts])


def log_run(con, batch, started, status, rows_in, rows_saved, version, duration, message):
    con.execute(
        "INSERT INTO runs (batch, started_at, finished_at, status, rows_in, rows_saved, "
        "model_version, duration_s, message) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (batch, started, now(), status, rows_in, rows_saved, version, round(duration, 3), message),
    )


def save_retrain(con, row: dict) -> None:
    _replace(con, "retrains", [row])


# ---------------------------------------------------------------- чтение
def processed_batches(con) -> list[str]:
    """Порции с успешной обработкой в порядке первой успешной обработки."""
    rows = con.execute(
        "SELECT batch, MIN(run_id) AS first FROM runs WHERE status = 'ok' "
        "GROUP BY batch ORDER BY first"
    ).fetchall()
    return [b for b, _ in rows]


def last_period_before(con, batch: str, initial_last: str) -> str:
    """Последний обработанный период для D-02.

    Учитываются порции, впервые обработанные раньше этой; повторная обработка
    той же порции не считается нарушением порядка (идемпотентность), а та же
    выгрузка под другим именем — считается.
    """
    order = processed_batches(con)
    before = order[: order.index(batch)] if batch in order else order
    if not before:
        return initial_last
    marks = ", ".join("?" * len(before))
    row = con.execute(
        f"SELECT MAX(period) FROM raw_data WHERE batch IN ({marks})", before
    ).fetchone()
    return max(row[0] or initial_last, initial_last)


def load_batches(con, batches: list[str]) -> pd.DataFrame:
    """Принятые записи порций в исходных именах столбцов, по возрастанию row_id."""
    if not batches:
        return pd.DataFrame(columns=DATA_COLUMNS)
    marks = ", ".join("?" * len(batches))
    df = pd.read_sql(f"SELECT * FROM raw_data WHERE batch IN ({marks})", con, params=batches)
    df = df.rename(columns={_plain(c): c for c in DATA_COLUMNS})
    return df.sort_values("row_id").reset_index(drop=True)


def history(con) -> pd.DataFrame:
    """История показателей: строки — порции в порядке обработки, столбцы — показатели."""
    h = pd.read_sql("SELECT batch, metric, value FROM metrics_history", con)
    if h.empty:
        return h
    order = processed_batches(con)
    wide = h.pivot(index="batch", columns="metric", values="value")
    return wide.reindex([b for b in order if b in wide.index])


def table(con, name: str) -> pd.DataFrame:
    return pd.read_sql(f"SELECT * FROM {name}", con)


def counts(con) -> dict:
    return {
        t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        for t in ("raw_data", "predictions", "metrics_history", "runs")
    }

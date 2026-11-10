"""Тесты API: состояние, прогноз, проверка формата запроса, показатели и
согласованность прогноза API с пакетной обработкой (один файл модели)."""

import sqlite3

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from bank_marketing.config import CSV_SEP, INCOMING_DIR

CLIENT = {
    "client_id": "C-24065",
    "period": "2008-11",
    "age": 31,
    "job": "admin.",
    "marital": "married",
    "education": "university.degree",
    "default": "no",
    "housing": "yes",
    "loan": "no",
    "contact": "telephone",
    "month": "nov",
    "day_of_week": "mon",
    "campaign": 1,
    "pdays": 999,
    "previous": 0,
    "poutcome": "nonexistent",
    "emp.var.rate": -0.1,
    "cons.price.idx": 93.2,
    "cons.conf.idx": -42.0,
    "euribor3m": 4.733,
    "nr.employed": 5195.8,
}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Отдельная база: порция 1 обрабатывается во временный каталог."""
    db = tmp_path_factory.mktemp("db") / "test.db"
    mp = pytest.MonkeyPatch()
    mp.setenv("BANK_DB", str(db))
    from bank_marketing.service.process_batch import process_batch

    process_batch(INCOMING_DIR / "batch_01.csv")
    from app.api import app

    yield TestClient(app), db
    mp.undo()


def test_health(client):
    c, _ = client
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and r.json()["batches"] == 1


def test_predict_ok(client):
    c, _ = client
    r = c.post("/predict", json=CLIENT)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probability"] <= 1
    assert body["model_version"].startswith("v")


def test_predict_invalid(client):
    c, _ = client
    r = c.post("/predict", json={**CLIENT, "age": 5, "contact": "pager"})
    assert r.status_code == 422
    fields = {e["loc"][-1] for e in r.json()["detail"]}
    assert fields == {"age", "contact"}


def test_metrics(client):
    c, _ = client
    r = c.get("/metrics")
    assert r.status_code == 200
    assert r.json()["batch"] == "batch_01" and r.json()["rows_saved"] > 0


def test_same_model_as_batch(client):
    """Прогноз API по клиенту порции совпадает с его прогнозом в хранилище."""
    c, db = client
    row = pd.read_csv(INCOMING_DIR / "batch_01.csv", sep=CSV_SEP, nrows=1).iloc[0]
    payload = {
        k: (row[k].item() if hasattr(row[k], "item") else row[k])
        for k in CLIENT
        if k != "client_id"
    }
    api = c.post("/predict", json=payload).json()["probability"]
    with sqlite3.connect(db) as con:
        stored = con.execute(
            "SELECT probability FROM predictions WHERE row_id = ?", (int(row["row_id"]),)
        ).fetchone()[0]
    assert abs(api - stored) < 1e-4

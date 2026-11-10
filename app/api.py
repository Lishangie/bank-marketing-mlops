"""API прогноза (FastAPI): /predict, /metrics, /health.

    uv run uvicorn app.api:app --port 8000      # документация: http://127.0.0.1:8000/docs

Модель — та же действующая версия, что у сценария обработки порций
(models/current.json читается при каждом запросе, поэтому замена и откат
применяются без перезапуска сервиса).
"""

from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from bank_marketing.features import clean
from bank_marketing.service import registry, storage

Job = Literal[
    "admin.",
    "blue-collar",
    "entrepreneur",
    "housemaid",
    "management",
    "retired",
    "self-employed",
    "services",
    "student",
    "technician",
    "unemployed",
    "unknown",
]
Education = Literal[
    "basic.4y",
    "basic.6y",
    "basic.9y",
    "high.school",
    "illiterate",
    "professional.course",
    "university.degree",
    "unknown",
]
YesNo = Literal["no", "yes", "unknown"]
Month = Literal["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


class Client(BaseModel):
    """Данные клиента в формате порции (без duration — она известна только после звонка)."""

    model_config = ConfigDict(populate_by_name=True)

    client_id: str = Field("—", max_length=64)
    period: str = Field(pattern=r"^\d{4}-(0[1-9]|1[0-2])$", description="год-месяц звонка")
    age: int = Field(ge=17, le=100)
    job: Job
    marital: Literal["divorced", "married", "single", "unknown"]
    education: Education
    default: YesNo
    housing: YesNo
    loan: YesNo
    contact: Literal["cellular", "telephone"]
    month: Month
    day_of_week: Literal["mon", "tue", "wed", "thu", "fri"]
    campaign: int = Field(ge=1, le=100)
    pdays: int = Field(999, ge=0, le=999)
    previous: int = Field(0, ge=0, le=50)
    poutcome: Literal["failure", "nonexistent", "success"] = "nonexistent"
    emp_var_rate: float = Field(alias="emp.var.rate", ge=-5, le=5)
    cons_price_idx: float = Field(alias="cons.price.idx", ge=80, le=110)
    cons_conf_idx: float = Field(alias="cons.conf.idx", ge=-60, le=0)
    euribor3m: float = Field(ge=-1, le=10)
    nr_employed: float = Field(alias="nr.employed", ge=4000, le=6000)


app = FastAPI(title="Прогноз отклика клиентов банка", version="1.0")


@app.get("/health")
def health() -> dict:
    db = storage.db_path()
    out = {"status": "ok", "model_version": registry.current()["version"], "db": Path(db).exists()}
    if out["db"]:
        con = storage.connect(db)
        out["batches"] = len(storage.processed_batches(con))
        con.close()
    return out


@app.post("/predict")
def predict(client: Client) -> dict:
    model, card = registry.load()
    row = client.model_dump(by_alias=True)
    client_id = row.pop("client_id")
    X = clean(pd.DataFrame([{"row_id": 0, **row}]))
    proba = float(model.predict_proba(X)[0, 1])
    return {
        "client_id": client_id,
        "probability": round(proba, 4),
        "above_threshold": proba >= card["threshold"],
        "threshold": card["threshold"],
        "model_version": card["version"],
    }


@app.get("/metrics")
def metrics() -> dict:
    con = storage.connect()
    hist = storage.history(con)
    if hist.empty:
        con.close()
        raise HTTPException(404, "порции ещё не обрабатывались")
    batch = hist.index[-1]
    alerts = storage.table(con, "alerts")
    con.close()
    values = {k: round(float(v), 4) for k, v in hist.loc[batch].dropna().items()}
    return {
        "batch": batch,
        **values,
        "alerts": alerts.loc[alerts["batch"] == batch, "message"].tolist(),
    }

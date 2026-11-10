class Client(BaseModel):
    period: str = Field(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")
    age: int = Field(ge=17, le=100)
    job: Job
    contact: Literal["cellular", "telephone"]
    campaign: int = Field(ge=1, le=100)
    pdays: int = Field(999, ge=0, le=999)
    euribor3m: float = Field(ge=-1, le=10)
    ...  # всего 20 полей порции, без duration


@app.post("/predict")
def predict(client: Client) -> dict:
    model, card = registry.load()            # current.json → model_vN.joblib
    row = client.model_dump(by_alias=True)
    client_id = row.pop("client_id")
    X = clean(pd.DataFrame([{"row_id": 0, **row}]))
    proba = float(model.predict_proba(X)[0, 1])
    return {"client_id": client_id, "probability": round(proba, 4),
            "above_threshold": proba >= card["threshold"],
            "threshold": card["threshold"], "model_version": card["version"]}

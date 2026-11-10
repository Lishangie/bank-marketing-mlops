raw = pd.read_csv(path, sep=CSV_SEP, dtype=str)
last = storage.last_period_before(con, batch, ref["last_period"])
df, results = validate(raw, ref, last_period=last)       # D-01…D-09, ПР4
storage.save_quality(con, batch, results)
if df is None:                                           # D-01 или D-02
    storage.log_run(con, batch, started, "rejected", len(raw), 0, version, ...)
    return {"batch": batch, "status": "rejected", "reasons": reasons}
storage.save_raw(con, batch, df)                         # INSERT OR REPLACE
X = clean(df).sort_values("row_id").reset_index(drop=True)
y = (X.pop("y") == "yes").astype(int)
proba = model.predict_proba(X)[:, 1]                     # действующая версия
storage.save_predictions(con, batch, preds)              # + model_version
m = batch_metrics(model, X, y, proba, card, results, len(raw), k)
alerts = check_rules(m, previous_metrics(con, batch), results, P)
storage.save_metrics(con, batch, m, version)
storage.save_alerts(con, batch, alerts)
storage.log_run(con, batch, started, "ok", len(raw), len(X), version, ...)
con.commit()

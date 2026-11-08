selection = json.loads((REPORTS_DIR / "selection.json").read_text(encoding="utf-8"))
X, y = load_train(params)                        # май – июль 2008 г.
model, hp = build(selection, params["seed"])     # CatBoost, глубина 3, 500 итераций
model.fit(X, y)
joblib.dump(model, "models/model_v1.joblib")
info = {"version": "v1", "trained": date.today().isoformat(),
        "git_commit": git_commit(), "data_md5": md5(INITIAL_FILE),
        "algorithm": selection["model"], "hyperparameters": hp,
        "features": {"numeric": num, "categorical": cat},
        "cv": {"scheme": "TimeSeriesSplit(5)", "pr_auc": selection["cv_pr_auc"]}}
Path("models/model_v1.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
with mlflow.start_run(run_name="model_v1"):
    mlflow.sklearn.log_model(model, name="model",
                             registered_model_name="bank-marketing-response",
                             serialization_format="cloudpickle")

"""Обучение и сохранение итоговой модели ПР3.

Модель и её параметры берутся из reports/selection.json (результат
train_experiments.py), обучение — на всей обучающей выборке исходной части.
Сохраняются models/model_v1.joblib и описание models/model_v1.json; модель
регистрируется в реестре MLflow как bank-marketing-response.

Результат: report_assets/pr3/train_out.txt.
"""

import hashlib
import json
import platform
import subprocess
from datetime import date

import catboost
import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression

from bank_marketing.config import INITIAL_FILE, MODELS_DIR, REPORTS_DIR, ROOT
from bank_marketing.experiments import MLFLOW_DIR, MLFLOW_EXPERIMENT, load_params, load_train
from bank_marketing.features import feature_columns, make_pipeline
from bank_marketing.models import CatBoostModel
from bank_marketing.report import emit

MODEL_FILE = MODELS_DIR / "model_v1.joblib"
INFO_FILE = MODELS_DIR / "model_v1.json"
REGISTRY_NAME = "bank-marketing-response"


def build(selection: dict, seed: int):
    """Пайплайн итоговой модели по описанию из selection.json."""
    fam, p = selection["family"], selection["params"]
    if fam == "linear":
        l1 = p.get("l1_ratio", 1 if p.get("penalty") == "L1" else 0)
        model = LogisticRegression(
            C=p["C"],
            l1_ratio=l1,
            solver="liblinear",
            class_weight=p.get("class_weight"),
            max_iter=5000,
            random_state=seed,
            intercept_scaling=100,
        )
        return make_pipeline(model), {
            "C": p["C"],
            "l1_ratio": l1,
            "class_weight": p.get("class_weight"),
        }
    if fam == "boosting":
        kw = {k: v for k, v in p.items() if k in CatBoostModel().get_params()}
        return make_pipeline(CatBoostModel(random_seed=seed, **kw), native_cat=True), kw
    raise ValueError(f"семейство {fam} не предусмотрено скриптом train_final")


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    params = load_params()
    selection = json.loads((REPORTS_DIR / "selection.json").read_text(encoding="utf-8"))
    X, y = load_train(params)
    model, hp = build(selection, params["seed"])
    model.fit(X, y)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    num, cat = feature_columns("new")
    info = {
        "version": "v1",
        "trained": date.today().isoformat(),
        "git_commit": git_commit(),
        "data": "data/initial/initial.csv, обучающая выборка "
        f"{X['period'].min()}…{X['period'].max()} (до {params['test_from']})",
        "data_md5": hashlib.md5(INITIAL_FILE.read_bytes()).hexdigest(),
        "rows": len(X),
        "positive_share": round(float(y.mean()), 5),
        "algorithm": selection["model"],
        "family": selection["family"],
        "hyperparameters": hp,
        "features": {"numeric": num, "categorical": cat},
        "cv": {
            "scheme": f"TimeSeriesSplit({params['cv_splits']})",
            "pr_auc": selection["cv_pr_auc"],
            "pr_auc_std": selection["cv_pr_auc_std"],
            "roc_auc": selection["cv_roc_auc"],
            f"lift_at_{int(params['top_k'] * 100)}": selection["cv_lift_k"],
            "baseline_pr_auc": selection["baseline_pr_auc"],
        },
        "environment": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "catboost": catboost.__version__,
        },
        "reproduce": "uv run dvc repro train",
    }
    INFO_FILE.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    # Проверка: загруженная из файла модель даёт те же вероятности.
    loaded = joblib.load(MODEL_FILE)
    same = bool(np.array_equal(model.predict_proba(X), loaded.predict_proba(X)))

    import mlflow
    from mlflow.models import infer_signature

    mlflow.set_tracking_uri(f"sqlite:///{(MLFLOW_DIR / 'mlflow.db').as_posix()}")
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT) is None:
        mlflow.create_experiment(
            MLFLOW_EXPERIMENT, artifact_location=(MLFLOW_DIR / "artifacts").as_uri()
        )
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    sample = X.head(50)
    with mlflow.start_run(run_name="model_v1") as run:
        mlflow.set_tags({"stage": "итоговая", "family": selection["family"]})
        mlflow.log_params({k: str(v) for k, v in hp.items()})
        mlflow.log_metrics(
            {
                "cv_pr_auc": selection["cv_pr_auc"],
                "cv_roc_auc": selection["cv_roc_auc"],
                "cv_lift_k": selection["cv_lift_k"],
            }
        )
        mlflow.log_artifact(str(INFO_FILE))
        mv = mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name=REGISTRY_NAME,
            signature=infer_signature(sample, model.predict_proba(sample)[:, 1]),
            serialization_format="cloudpickle",
        )
        run_id = run.info.run_id
    version = getattr(mv, "registered_model_version", None)

    coef_line, coef_rows = "", []
    if selection["family"] == "linear":
        coef = model.named_steps["model"].coef_.ravel()
        names = [n.split("__", 1)[1] for n in model.named_steps["prep"].get_feature_names_out()]
        coef_line = f"ненулевых коэффициентов: {int((np.abs(coef) > 1e-8).sum())} из {len(coef)}"
        order = np.argsort(-np.abs(coef))[:14]
        coef_rows = ["  коэффициенты, 14 наибольших по модулю:"]
        coef_rows += [f"    {names[i]:<28} {coef[i]:+.4f}" for i in order]
    out = [
        "Итоговая модель ПР3",
        f"  алгоритм: {selection['model']}; гиперпараметры: {hp}",
        f"  обучающая выборка: {len(X)} записей, доля отклика {y.mean():.2%}",
        f"  PR-AUC по КВ {selection['cv_pr_auc']:.4f} ± {selection['cv_pr_auc_std']:.4f}; "
        f"базовое {selection['baseline_pr_auc']:.4f}",
        f"  {coef_line}" if coef_line else "",
        f"  файлы: {MODEL_FILE.relative_to(ROOT).as_posix()} "
        f"({MODEL_FILE.stat().st_size} байт), {INFO_FILE.relative_to(ROOT).as_posix()}",
        f"  загрузка из файла даёт те же вероятности: {'да' if same else 'НЕТ'}",
        f"  MLflow: запуск {run_id[:8]}, реестр {REGISTRY_NAME}, версия {version}",
    ]
    if selection["family"] == "boosting":
        names = list(model.named_steps["prep"].get_feature_names_out())
        imp = model.named_steps["model"].model_.get_feature_importance()
        order = np.argsort(-imp)[:10]
        coef_rows = ["  важность признаков CatBoost (PredictionValuesChange), 10 наибольших:"]
        coef_rows += [f"    {names[i]:<28} {imp[i]:6.2f}" for i in order]
    out += coef_rows
    emit([s for s in out if s], ROOT / "report_assets" / "pr3" / "train_out.txt")


if __name__ == "__main__":
    main()

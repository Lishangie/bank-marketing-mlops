"""Единая схема экспериментов ПР3: разбиение, метрики, кросс-валидация, учёт.

Каждый эксперимент — одна строка reports/experiments.csv и один запуск MLflow
(эксперимент «pr3-model-selection», хранилище mlruns/mlflow.db): модель,
набор признаков, параметры, метрики по блокам, среднее и стандартное
отклонение, время обучения, дата.
"""

import json
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import make_scorer
from sklearn.model_selection import TimeSeriesSplit, cross_validate

from bank_marketing.config import INITIAL_FILE, PARAMS_FILE, ROOT

MLFLOW_DIR = ROOT / "mlruns"
MLFLOW_EXPERIMENT = "pr3-model-selection"
METRICS = ["pr_auc", "roc_auc", "lift_k", "recall_k"]


def load_params() -> dict:
    return yaml.safe_load(PARAMS_FILE.read_text(encoding="utf-8"))["train"]


def top_k_index(score: np.ndarray, k: float) -> np.ndarray:
    n = max(1, int(round(len(score) * k)))
    return np.argsort(-np.asarray(score), kind="stable")[:n]


def lift_at_k(y_true, score, k: float = 0.1) -> float:
    """Доля отклика среди k·100 % клиентов с наибольшей оценкой / средняя доля отклика."""
    y = np.asarray(y_true)
    return float(y[top_k_index(score, k)].mean() / y.mean())


def recall_at_k(y_true, score, k: float = 0.1) -> float:
    """Доля всех откликнувшихся, попавших в верхние k·100 % списка."""
    y = np.asarray(y_true)
    return float(y[top_k_index(score, k)].sum() / y.sum())


def scorers(k: float) -> dict:
    return {
        "pr_auc": "average_precision",
        "roc_auc": "roc_auc",
        "lift_k": make_scorer(lift_at_k, response_method="predict_proba", k=k),
        "recall_k": make_scorer(recall_at_k, response_method="predict_proba", k=k),
    }


def split_train_test(X: pd.DataFrame, y: pd.Series, test_from: str):
    """Временное разбиение исходной части по границе месяца: тест — с test_from."""
    test = X["period"] >= test_from
    return X[~test], y[~test], X[test], y[test]


def load_train(params: dict | None = None):
    """Обучающая выборка исходной части (тестовая в ПР3 не используется)."""
    from bank_marketing.features import prepare

    params = params or load_params()
    X, y = prepare(INITIAL_FILE)
    X_tr, y_tr, _, _ = split_train_test(X, y, params["test_from"])
    return X_tr.reset_index(drop=True), y_tr.reset_index(drop=True)


def time_cv(params: dict) -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=params["cv_splits"])


@dataclass
class Result:
    name: str
    family: str
    stage: str
    feature_set: str
    params: dict
    folds: dict  # метрика -> список значений по блокам
    fit_time: float
    train_folds: dict = field(default_factory=dict)

    def row(self) -> dict:
        out = {
            "date": date.today().isoformat(),
            "stage": self.stage,
            "family": self.family,
            "model": self.name,
            "feature_set": self.feature_set,
            "params": json.dumps(self.params, ensure_ascii=False, sort_keys=True),
        }
        for m in METRICS:
            v = np.asarray(self.folds[m])
            out[f"{m}_mean"] = round(float(v.mean()), 5)
            out[f"{m}_std"] = round(float(v.std()), 5)
            out[f"{m}_folds"] = " ".join(f"{x:.4f}" for x in v)
        if self.train_folds:
            out["pr_auc_train_mean"] = round(float(np.mean(self.train_folds["pr_auc"])), 5)
        out["fit_time_s"] = round(self.fit_time, 3)
        return out


def evaluate(
    name, family, stage, pipeline, X, y, cv, k, feature_set="new", params=None, train=False
) -> Result:
    """Кросс-валидация пайплайна; train=True — ещё и метрики на обучающих блоках."""
    r = cross_validate(pipeline, X, y, cv=cv, scoring=scorers(k), return_train_score=train)
    folds = {m: r[f"test_{m}"].tolist() for m in METRICS}
    tr = {m: r[f"train_{m}"].tolist() for m in METRICS} if train else {}
    fit_time = float(np.mean(r["fit_time"]))
    return Result(name, family, stage, feature_set, params or {}, folds, fit_time, tr)


def score_rule(name, family, X, y, cv, k, score_fn, params=None) -> Result:
    """Базовое решение без обучения: оценка — функция признаков, метрики по тем же блокам."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    folds = {m: [] for m in METRICS}
    for _, val in cv.split(X):
        yv, s = y.iloc[val], score_fn(X.iloc[val])
        folds["pr_auc"].append(average_precision_score(yv, s))
        folds["roc_auc"].append(roc_auc_score(yv, s))
        folds["lift_k"].append(lift_at_k(yv, s, k))
        folds["recall_k"].append(recall_at_k(yv, s, k))
    return Result(name, family, "сравнение", "—", params or {}, folds, 0.0)


class Tracker:
    """Запись результатов в CSV и MLflow."""

    def __init__(self, use_mlflow: bool = True):
        self.rows: list[dict] = []
        self.results: list[Result] = []
        self.mlflow = None
        if use_mlflow:
            import mlflow

            MLFLOW_DIR.mkdir(exist_ok=True)
            mlflow.set_tracking_uri(f"sqlite:///{(MLFLOW_DIR / 'mlflow.db').as_posix()}")
            if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT) is None:
                mlflow.create_experiment(
                    MLFLOW_EXPERIMENT, artifact_location=(MLFLOW_DIR / "artifacts").as_uri()
                )
            mlflow.set_experiment(MLFLOW_EXPERIMENT)
            self.mlflow = mlflow

    def add(self, res: Result) -> Result:
        self.results.append(res)
        self.rows.append(res.row())
        if self.mlflow:
            with self.mlflow.start_run(run_name=res.name):
                self.mlflow.set_tags(
                    {"stage": res.stage, "family": res.family, "feature_set": res.feature_set}
                )
                self.mlflow.log_params({k: str(v)[:250] for k, v in res.params.items()})
                for m in METRICS:
                    v = np.asarray(res.folds[m])
                    self.mlflow.log_metric(f"{m}_mean", float(v.mean()))
                    self.mlflow.log_metric(f"{m}_std", float(v.std()))
                    for i, x in enumerate(v):
                        self.mlflow.log_metric(m, float(x), step=i + 1)
                self.mlflow.log_metric("fit_time_s", res.fit_time)
        return res

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

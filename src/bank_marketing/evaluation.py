"""Оценка итоговой модели (ПР4): метрики, бутстреп, порог по цене ошибок,
срезы, перестановочная важность, частичная зависимость, локальные объяснения,
устойчивость и PSI признаков модели.

Функции принимают таблицы после prepare() — исходные столбцы, как их получает
пайплайн модели. Важность и локальные вклады считаются по группам исходных
столбцов (GROUPS): признаки модели, построенные из одного столбца (age и
age_group, job и job_group) или описывающие одно состояние экономики
(macro_pc1, cons.price.idx, cons.conf.idx), меняются вместе, иначе
перемешивание создало бы невозможные сочетания.
"""

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from bank_marketing.config import MACRO, PARAMS_FILE
from bank_marketing.experiments import lift_at_k, recall_at_k
from bank_marketing.stats import bin_shares, psi_bins, psi_from_shares

GROUPS = {
    "Возраст": ["age"],
    "Род занятий": ["job"],
    "Семейное положение": ["marital"],
    "Образование": ["education"],
    "Просрочка по кредиту": ["default"],
    "Ипотека": ["housing"],
    "Потребительский кредит": ["loan"],
    "Тип связи": ["contact"],
    "Месяц": ["month"],
    "День недели": ["day_of_week"],
    "Число контактов": ["campaign"],
    "История контактов": ["pdays", "previous", "poutcome"],
    "Макропоказатели": MACRO,
}


def load_eval_params() -> dict:
    return yaml.safe_load(PARAMS_FILE.read_text(encoding="utf-8"))["evaluate"]


# ---------------------------------------------------------------- метрики
def metrics(y, s, k: float = 0.1) -> dict:
    """PR-AUC, ROC-AUC, Lift@k, Recall@k; для постоянной оценки Lift не определён."""
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    const = np.ptp(s) == 0
    return {
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan"),
        "lift_k": float("nan") if const else lift_at_k(y, s, k),
        "recall_k": float("nan") if const else recall_at_k(y, s, k),
        "base": float(y.mean()),
    }


def at_threshold(y, s, t: float) -> dict:
    """Матрица ошибок и метрики при пороге t (клиент в списке обзвона, если s ≥ t)."""
    y, pred = np.asarray(y), (np.asarray(s) >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "accuracy": (tp + tn) / len(y),
        "called": int(tp + fp),
    }


def bootstrap(y, scores: dict, n: int, seed: int, k: float = 0.1, pairs=()) -> dict:
    """95 %-ные интервалы метрик и разностей пар моделей по одним и тем же выборкам.

    scores — {имя: оценки}; pairs — [(a, b), …], разность a − b.
    Возвращает {имя или «a − b»: {метрика: (оценка, нижняя, верхняя)}}.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    names = list(scores)
    keys = ["pr_auc", "roc_auc", "lift_k"]
    draws = {m: {key: [] for key in keys} for m in names}
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        for m in names:
            r = metrics(y[idx], np.asarray(scores[m])[idx], k)
            for key in keys:
                draws[m][key].append(r[key])
    out = {}
    for m in names:
        point = metrics(y, scores[m], k)
        out[m] = {key: (point[key], *_ci(draws[m][key])) for key in keys}
    for a, b in pairs:
        pa, pb = metrics(y, scores[a], k), metrics(y, scores[b], k)
        out[f"{a} − {b}"] = {
            key: (pa[key] - pb[key], *_ci(np.asarray(draws[a][key]) - np.asarray(draws[b][key])))
            for key in keys
        }
    return out


def _ci(values) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if not len(v):
        return float("nan"), float("nan")
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------------------------------------------------------- порог
def loss_curve(y, s, thresholds, c_fp: float, c_fn: float) -> np.ndarray:
    """Ожидаемые потери FN(t)·c_FN + FP(t)·c_FP для каждого порога, руб."""
    y, s = np.asarray(y), np.asarray(s)
    out = []
    for t in thresholds:
        pred = s >= t
        out.append(((~pred) & (y == 1)).sum() * c_fn + (pred & (y == 0)).sum() * c_fp)
    return np.asarray(out, dtype=float)


def oof_scores(pipeline_factory, X: pd.DataFrame, y: pd.Series, cv):
    """Прогнозы вне блока на обучающей выборке (TimeSeriesSplit).

    Возвращает (индексы, оценки, [(модель блока, индексы проверки), …]).
    """
    idx, sc, folds = [], [], []
    for tr, va in cv.split(X):
        m = pipeline_factory().fit(X.iloc[tr], y.iloc[tr])
        idx.append(va)
        sc.append(m.predict_proba(X.iloc[va])[:, 1])
        folds.append((m, va))
    return np.concatenate(idx), np.concatenate(sc), folds


# ---------------------------------------------------------------- срезы
def slice_table(X: pd.DataFrame, y, s, t: float, slices: dict, k: float = 0.1) -> pd.DataFrame:
    """Метрики по срезам: slices = {название среза: Series с группой для каждой строки}."""
    y, s = np.asarray(y), np.asarray(s)
    rows = []
    for title, groups in slices.items():
        g = np.asarray(groups)
        for value in pd.unique(g):
            m = g == value
            yy, ss = y[m], s[m]
            r = {
                "slice": title,
                "group": value,
                "n": int(m.sum()),
                "pos": int(yy.sum()),
                "base": float(yy.mean()),
                "mean_pred": float(ss.mean()),
            }
            ok = 0 < yy.sum() < len(yy)
            r["pr_auc"] = float(average_precision_score(yy, ss)) if ok else float("nan")
            r["roc_auc"] = float(roc_auc_score(yy, ss)) if ok else float("nan")
            r["recall_t"] = at_threshold(yy, ss, t)["recall"] if yy.sum() else float("nan")
            r["called_share"] = float((ss >= t).mean())
            rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- интерпретация
def permutation_importance(
    model, X: pd.DataFrame, y, n_repeats: int, seed: int, groups: dict = GROUPS
) -> pd.DataFrame:
    """Снижение PR-AUC и ROC-AUC при совместном перемешивании столбцов группы."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    s0 = model.predict_proba(X)[:, 1]
    base_pr, base_roc = average_precision_score(y, s0), roc_auc_score(y, s0)
    rows = []
    for name, cols in groups.items():
        d_pr, d_roc = [], []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            for c in cols:  # одна перестановка на группу, типы столбцов сохраняются
                Xp[c] = X[c].to_numpy()[perm]
            s = model.predict_proba(Xp)[:, 1]
            d_pr.append(base_pr - average_precision_score(y, s))
            d_roc.append(base_roc - roc_auc_score(y, s))
        rows.append(
            {
                "group": name,
                "d_pr_auc": float(np.mean(d_pr)),
                "d_pr_auc_std": float(np.std(d_pr)),
                "d_roc_auc": float(np.mean(d_roc)),
                "d_roc_auc_std": float(np.std(d_roc)),
            }
        )
    return pd.DataFrame(rows).sort_values("d_roc_auc", ascending=False).reset_index(drop=True)


def typical_values(X: pd.DataFrame) -> dict:
    """Типичные значения по обучающей выборке: медиана числовых, мода категориальных."""
    out = {}
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            out[c] = X[c].median()
        else:
            out[c] = X[c].mode().iat[0]
    return out


def local_contributions(
    model, row: pd.DataFrame, typical: dict, groups: dict = GROUPS
) -> pd.Series:
    """Вклад группы = p(x) − p(x с типичными значениями группы)."""
    p0 = model.predict_proba(row)[:, 1][0]
    out = {}
    for name, cols in groups.items():
        r = row.copy()
        for c in cols:
            r[c] = typical[c]
        r = r.astype(row.dtypes.to_dict())
        out[name] = p0 - model.predict_proba(r)[:, 1][0]
    return pd.Series(out)


def partial_dependence(tail, Z: pd.DataFrame, feature: str, grid) -> np.ndarray:
    """Средняя вероятность при подстановке значения признака модели всем объектам.

    tail — пайплайн без FeatureBuilder (prep + model), Z — таблица после FeatureBuilder.
    """
    out = []
    for v in grid:
        Zv = Z.copy()
        Zv[feature] = v
        out.append(float(tail.predict_proba(Zv)[:, 1].mean()))
    return np.asarray(out)


# ---------------------------------------------------------------- устойчивость
def perturb_noise(X: pd.DataFrame, rng, rel: float = 0.1) -> pd.DataFrame:
    """Шум измерений: возраст × (1 + N(0; rel)), число контактов ± 1."""
    Xp = X.copy()
    Xp["age"] = np.clip(np.round(X["age"] * (1 + rng.normal(0, rel, len(X)))), 17, 100).astype(
        X["age"].dtype
    )
    step = rng.choice([-1, 0, 1], len(X))
    Xp["campaign"] = np.clip(X["campaign"] + step, 1, None).astype(X["campaign"].dtype)
    return Xp


def perturb_missing(X: pd.DataFrame, rng, cols, share: float, fill: dict) -> pd.DataFrame:
    """Пропуски в доле share записей по каждому столбцу, заполнение типичным значением."""
    Xp = X.copy()
    for c in cols:
        m = rng.random(len(X)) < share
        Xp.loc[m, c] = fill[c]
    return Xp


# ---------------------------------------------------------------- PSI
def feature_psi(ref: pd.DataFrame, cur: pd.DataFrame, features) -> dict:
    """PSI признаков модели: бины по обучающей выборке (квантили или значения)."""
    out = {}
    for f in features:
        spec = psi_bins(ref[f])
        out[f] = psi_from_shares(bin_shares(ref[f], spec).tolist(), cur[f], spec)
    return out

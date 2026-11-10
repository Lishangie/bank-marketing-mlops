"""Переобучение по правилу и решение о замене модели.

    uv run python -m bank_marketing.service.retrain [порция]

По умолчанию — после последней обработанной порции. Шаги:

1. Данные: исходная часть + принятые порции до этой (из хранилища) — «прошлое»;
   эта порция по row_id делится на первые 70 % (дообучение) и последние 30 %
   (отложенная часть, на ней не обучается ни один претендент).
2. Семейство претендента: LR и CatBoost обучаются на прошлом и сравниваются
   по PR-AUC на первых 70 % порции; CatBoost берётся, только если выигрывает
   не меньше select_delta (иначе — более простая LR, правило ПР3).
3. Выбранное семейство обучается на прошлом + первых 70 % порции.
4. Претендент и действующая модель оцениваются на отложенной части. Замена,
   если PR-AUC претендента не хуже больше чем на pr_auc_tol и |прогноз − факт|
   меньше; иначе действующая остаётся. Предыдущая версия всегда сохраняется.
"""

import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from bank_marketing.config import INITIAL_FILE, TARGET
from bank_marketing.evaluation import metrics as quality_metrics
from bank_marketing.features import clean, make_pipeline, prepare
from bank_marketing.models import CatBoostModel
from bank_marketing.service import registry, storage
from bank_marketing.service.features_delta import make_delta_pipeline
from bank_marketing.service.process_batch import ensure_registry, say

NAMES = {"lr": "Логистическая регрессия", "catboost": "CatBoost"}
FAMILY = {"lr": "linear", "catboost": "boosting"}


def build(name: str, p: dict, feature_set: str, seed: int):
    """Претендент: набор new (как v1) или delta (new + euribor_delta_3m)."""
    if name == "lr":
        model = LogisticRegression(
            C=p["C"], solver="liblinear", max_iter=5000, random_state=seed, intercept_scaling=100
        )
        native = False
    else:
        model, native = CatBoostModel(random_seed=seed, **p), True
    if feature_set == "delta":
        return make_delta_pipeline(model, native_cat=native)
    return make_pipeline(model, feature_set=feature_set, native_cat=native)


def to_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = clean(df).sort_values("row_id").reset_index(drop=True)
    y = (X.pop(TARGET) == "yes").astype(int)
    return X, y


def evaluate(y, s, k: float) -> dict:
    q = quality_metrics(y, s, k)
    return {
        "pr_auc": q["pr_auc"],
        "roc_auc": q["roc_auc"],
        "lift_k": q["lift_k"],
        "pred_mean": float(np.mean(s)),
        "fact": float(np.mean(y)),
        "calib_pp": float(np.mean(s) - np.mean(y)),
    }


def retrain(batch: str | None = None) -> dict:
    t0, started = time.perf_counter(), storage.now()
    ensure_registry()
    P = registry.params()
    S, k, seed = P["service"], P["train"]["top_k"], P["train"]["seed"]
    con = storage.connect()
    order = storage.processed_batches(con)
    if not order:
        raise SystemExit("нет обработанных порций")
    batch = batch or order[-1]
    past_batches = order[: order.index(batch)]

    X0, y0 = prepare(INITIAL_FILE)
    Xp, yp = to_xy(storage.load_batches(con, past_batches)) if past_batches else (None, None)
    Xb, yb = to_xy(storage.load_batches(con, [batch]))
    cut = int(round(len(Xb) * (1 - S["holdout_share"])))
    X_past = pd.concat([X0, Xp], ignore_index=True) if Xp is not None else X0
    y_past = pd.concat([y0, yp], ignore_index=True) if yp is not None else y0
    X_fit, y_fit = Xb.iloc[:cut], yb.iloc[:cut]
    X_hold, y_hold = Xb.iloc[cut:], yb.iloc[cut:]

    # 2. выбор семейства на первых 70 % порции
    sel = {}
    for name, p in S["candidates"].items():
        m = build(name, p, S["feature_set"], seed).fit(X_past, y_past)
        sel[name] = float(average_precision_score(y_fit, m.predict_proba(X_fit)[:, 1]))
    chosen = "catboost" if sel["catboost"] - sel["lr"] >= P["train"]["select_delta"] else "lr"

    # 3. претендент на прошлом + первых 70 % порции
    X_train = pd.concat([X_past, X_fit], ignore_index=True)
    y_train = pd.concat([y_past, y_fit], ignore_index=True)
    cand = build(chosen, S["candidates"][chosen], S["feature_set"], seed).fit(X_train, y_train)

    # 4. сравнение на отложенной части
    cur_model, cur_card = registry.load()
    cur = evaluate(y_hold, cur_model.predict_proba(X_hold)[:, 1], k)
    new = evaluate(y_hold, cand.predict_proba(X_hold)[:, 1], k)
    replace = new["pr_auc"] >= cur["pr_auc"] - S["replace"]["pr_auc_tol"] and abs(
        new["calib_pp"]
    ) < abs(cur["calib_pp"])

    new_version = None
    if replace:
        hp = ", ".join(f"{a} = {b}" for a, b in S["candidates"][chosen].items())
        card = {
            "created": storage.now(),
            "algorithm": f"{NAMES[chosen]} ({hp})",
            "family": FAMILY[chosen],
            "params": S["candidates"][chosen],
            "feature_set": S["feature_set"],
            "threshold": cur_card["threshold"],
            "trained_on": {
                "parts": [
                    "initial",
                    *past_batches,
                    f"{batch} (первые {round((1 - S['holdout_share']) * 100)} %)",
                ],
                "rows": len(X_train),
                "positive_share": float(y_train.mean()),
                "periods": [str(X_train["period"].min()), str(X_train["period"].max())],
                "rows_md5": registry.rows_md5(X_train["row_id"]),
            },
            "git_commit": registry.git_commit(),
            "selection_pr_auc": sel,
            "holdout": {
                "batch": batch,
                "rows": len(X_hold),
                "candidate": new,
                "replaced": {"version": cur_card["version"], **cur},
            },
            "reason": f"замена после {batch}",
            "profile": registry.profile(cand, X_train),
        }
        new_version = registry.register(cand, card)["version"]

    row = {
        "batch": batch,
        "started_at": started,
        "current_version": cur_card["version"],
        "candidate": NAMES[chosen],
        "train_rows": len(X_train),
        "holdout_rows": len(X_hold),
        "cur_pr_auc": cur["pr_auc"],
        "cand_pr_auc": new["pr_auc"],
        "cur_roc_auc": cur["roc_auc"],
        "cand_roc_auc": new["roc_auc"],
        "cur_lift_k": cur["lift_k"],
        "cand_lift_k": new["lift_k"],
        "cur_calib": cur["calib_pp"],
        "cand_calib": new["calib_pp"],
        "decision": "замена" if replace else "оставить действующую",
        "new_version": new_version,
        "duration_s": round(time.perf_counter() - t0, 2),
    }
    storage.save_retrain(con, row)
    con.commit()
    con.close()
    return {
        **row,
        "selection_pr_auc": sel,
        "fact": new["fact"],
        "cur_pred": cur["pred_mean"],
        "cand_pred": new["pred_mean"],
    }


def main(argv: list[str]) -> None:
    r = retrain(argv[1] if len(argv) > 1 else None)
    say("\n".join(f"{key}: {val}" for key, val in r.items()))
    say(registry.status())


if __name__ == "__main__":
    main(sys.argv)

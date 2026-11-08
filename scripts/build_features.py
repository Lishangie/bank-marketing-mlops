"""Таблицы признаков для обучающей, тестовой выборок и порций (ПР3).

Каждый файл проходит prepare (проверка D-01…D-09 и очистка), затем
FeatureBuilder, обученный только на обучающей выборке (май – июль 2008 г.).
Параметры признаков сохраняются в models/reference/feature_params.json,
таблицы — в data/processed/*.parquet. Проверка повторяемости: обработка
выполняется дважды, таблицы и их MD5 должны совпасть.

Результат: report_assets/pr3/features_out.txt.
"""

import hashlib
import json

import pandas as pd

from bank_marketing.config import INITIAL_FILE, PROCESSED_DIR, REFERENCE_FILE
from bank_marketing.data import batch_files
from bank_marketing.experiments import load_params, split_train_test
from bank_marketing.features import FeatureBuilder, load_reference, prepare
from bank_marketing.report import emit

PARAMS_OUT = REFERENCE_FILE.parent / "feature_params.json"


def md5(path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def build() -> tuple[dict[str, pd.DataFrame], dict]:
    params = load_params()
    ref = load_reference()
    X, y = prepare(INITIAL_FILE, ref)
    X_tr, y_tr, X_te, y_te = split_train_test(X, y, params["test_from"])
    fb = FeatureBuilder().fit(X_tr)
    parts = {"train": (X_tr, y_tr), "test": (X_te, y_te)}
    for path in batch_files():
        parts[path.stem] = prepare(path, ref)
    tables = {}
    for name, (Xp, yp) in parts.items():
        t = fb.transform(Xp)
        t["y"] = yp.to_numpy()
        tables[name] = t.reset_index(drop=True)
    return tables, fb.params()


def save(tables: dict[str, pd.DataFrame], fparams: dict) -> dict[str, str]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    sums = {}
    for name, t in tables.items():
        path = PROCESSED_DIR / f"{name}.parquet"
        t.to_parquet(path, index=False)
        sums[name] = md5(path)
    PARAMS_OUT.write_text(json.dumps(fparams, ensure_ascii=False, indent=2), encoding="utf-8")
    sums["feature_params"] = md5(PARAMS_OUT)
    return sums


def main() -> None:
    tables1, p1 = build()
    sums1 = save(tables1, p1)
    tables2, p2 = build()
    sums2 = save(tables2, p2)

    out = ["Таблицы признаков (параметры FeatureBuilder — по обучающей выборке)", ""]
    out.append(f"  {'часть':<10}{'строк':>7}{'столбцов':>10}{'доля отклика':>14}  период")
    for name, t in tables1.items():
        per = f"{t['period'].min()}…{t['period'].max()}"
        out.append(f"  {name:<10}{len(t):>7}{t.shape[1]:>10}{t['y'].mean():>14.2%}  {per}")
    out += [
        "",
        "Параметры признаков:",
        f"  campaign_cap (P99 campaign) = {p1['campaign_cap']:g}",
        "  macro_pc1: доля дисперсии "
        f"{p1['pca_explained']:.4f}; нагрузки "
        + ", ".join(
            f"{c} {v:+.3f}" for c, v in zip(p1["pca_columns"], p1["pca_loadings"], strict=True)
        ),
        "",
        "Новые признаки на обучающей выборке:",
    ]
    t = tables1["train"]
    out.append(
        f"  had_prev_contact = 1: {int(t['had_prev_contact'].sum())} записей; "
        f"campaign > cap: {int((t['campaign'] > p1['campaign_cap']).sum())} записей"
    )
    for col in ["age_group", "job_group"]:
        g = t.groupby(col)["y"].agg(["size", "mean"])
        out.append(
            f"  {col}: "
            + "; ".join(f"{k} {int(r['size'])} ({r['mean']:.1%})" for k, r in g.iterrows())
        )
    out.append(
        f"  macro_pc1: от {t['macro_pc1'].min():.3f} до {t['macro_pc1'].max():.3f}, "
        f"уникальных значений {t['macro_pc1'].nunique()}"
    )
    same_tables = all(tables1[n].equals(tables2[n]) for n in tables1) and p1 == p2
    same_md5 = sums1 == sums2
    out += [
        "",
        "Проверка повторяемости (два запуска подряд):",
        f"  таблицы совпадают: {'да' if same_tables else 'НЕТ'}",
        f"  MD5 файлов совпадают: {'да' if same_md5 else 'НЕТ'} ({len(sums1)} файлов)",
    ]
    out += [f"    {n}: {s}" for n, s in sums1.items()]
    emit(out, PROCESSED_DIR.parents[1] / "report_assets" / "pr3" / "features_out.txt")


if __name__ == "__main__":
    main()

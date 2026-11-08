"""Пайплайн подготовки данных и признаков (ПР3).

Три уровня, как в проекте пайплайна из ПР2:

1. prepare(path) — загрузка любого файла (исходная часть или порция), проверка
   D-01…D-09 со справочником исходной части и очистка без параметров.
2. FeatureBuilder — новые признаки по плану ПР2; параметры (P99 campaign,
   стандартизация и главная компонента макропоказателей) оцениваются в fit
   только по обучающим данным.
3. make_pipeline(model) — FeatureBuilder + ColumnTransformer + модель одним
   объектом Pipeline: при кросс-валидации параметры оцениваются заново на
   каждом обучающем блоке, утечки через предобработку нет.

Командная строка: `python -m bank_marketing.features <файл.csv>` печатает
размер и первые строки таблицы, готовой для модели.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bank_marketing.config import CSV_SEP, REFERENCE_FILE, TARGET
from bank_marketing.validation import CRITICAL, validate

# Признаки, известные только после звонка (C-02), и служебные столбцы.
LEAK = ["duration"]
ID_COLS = ["row_id", "period"]

# Редкие категории объединяются с близкими (ПР2, шаг «Очистка»).
RARE = {"default": {"yes": "unknown"}, "education": {"illiterate": "basic.4y"}}

PCA_COLUMNS = ["emp.var.rate", "euribor3m", "nr.employed"]
AGE_BINS = [0, 24, 59, np.inf]
AGE_LABELS = ["<25", "25-59", "60+"]
JOB_GROUPS = ["student", "retired", "unemployed"]

CAT_BASE = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]
# Наборы признаков: base — исходные столбцы без duration и pdays;
# new — с признаками плана (campaign_capped вместо campaign, macro_pc1 вместо
# трёх дублирующих друг друга макропоказателей).
FEATURE_SETS = {
    "base": {
        "num": [
            "age",
            "campaign",
            "previous",
            "emp.var.rate",
            "cons.price.idx",
            "cons.conf.idx",
            "euribor3m",
            "nr.employed",
        ],
        "cat": CAT_BASE,
    },
    "new": {
        "num": [
            "age",
            "campaign_capped",
            "previous",
            "had_prev_contact",
            "cons.price.idx",
            "cons.conf.idx",
            "macro_pc1",
        ],
        "cat": [*CAT_BASE, "age_group", "job_group"],
    },
}
# Признаки, кодирующие время (месяц и макропоказатели, постоянные внутри месяца).
TIME_PROXIES = ["month", "cons.price.idx", "cons.conf.idx", "macro_pc1"]


def feature_columns(feature_set: str = "new", drop: tuple[str, ...] = ()) -> tuple[list, list]:
    spec = FEATURE_SETS[feature_set]
    num = [c for c in spec["num"] if c not in drop]
    cat = [c for c in spec["cat"] if c not in drop]
    return num, cat


def load_reference(path: Path = REFERENCE_FILE) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Очистка без параметров: удаление duration, pdays = 999 → пусто, редкие категории."""
    out = df.drop(columns=[c for c in LEAK if c in df.columns]).copy()
    out["pdays"] = out["pdays"].where(out["pdays"] != 999).astype("float64")
    for col, mapping in RARE.items():
        out[col] = out[col].replace(mapping)
    return out


def prepare(path: Path | str, ref: dict | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Любой файл исходной части или порции → (X, y), готовые для make_pipeline.

    Проверка D-01…D-09 выполняется со справочником исходной части (дубликаты и
    записи вне диапазонов удаляются, неизвестные категории → unknown). Порядок
    строк сохраняется (по row_id), y — 0/1.
    """
    raw = pd.read_csv(path, sep=CSV_SEP, dtype=str)
    ref = ref if ref is not None else load_reference()
    df, results = validate(raw, ref, last_period=None)
    if df is None:
        bad = "; ".join(f"{r.rule}: {r.details}" for r in results if r.status == CRITICAL)
        raise ValueError(f"{path}: файл не прошёл проверку ({bad})")
    df = clean(df).sort_values("row_id").reset_index(drop=True)
    y = (df.pop(TARGET) == "yes").astype(int)
    return df, y


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Новые признаки плана ПР2; параметры оцениваются в fit по обучающим данным.

    had_prev_contact — был ли прошлый контакт (pdays не пусто);
    campaign_capped — campaign, ограниченный сверху квантилем cap_q;
    age_group, job_group — группы возраста и рода занятий;
    macro_pc1 — первая главная компонента стандартизованных PCA_COLUMNS.
    """

    def __init__(self, cap_q: float = 0.99):
        self.cap_q = cap_q

    def fit(self, X: pd.DataFrame, y=None):
        self.cap_ = float(X["campaign"].quantile(self.cap_q))
        self.scaler_ = StandardScaler().fit(X[PCA_COLUMNS])
        self.pca_ = PCA(n_components=1).fit(self.scaler_.transform(X[PCA_COLUMNS]))
        # Знак компоненты фиксируется: рост ставки → рост macro_pc1.
        self.sign_ = float(np.sign(self.pca_.components_[0][PCA_COLUMNS.index("euribor3m")]))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        out["had_prev_contact"] = out["pdays"].notna().astype(int)
        out["campaign_capped"] = out["campaign"].clip(upper=self.cap_)
        out["age_group"] = pd.cut(out["age"], AGE_BINS, labels=AGE_LABELS).astype(str)
        out["job_group"] = out["job"].where(out["job"].isin(JOB_GROUPS), "other")
        pc = self.pca_.transform(self.scaler_.transform(out[PCA_COLUMNS]))[:, 0]
        out["macro_pc1"] = self.sign_ * pc
        return out

    def params(self) -> dict:
        """Параметры для сохранения (feature_params.json)."""
        return {
            "campaign_cap": self.cap_,
            "pca_columns": PCA_COLUMNS,
            "scaler_mean": self.scaler_.mean_.tolist(),
            "scaler_scale": self.scaler_.scale_.tolist(),
            "pca_loadings": (self.sign_ * self.pca_.components_[0]).tolist(),
            "pca_explained": float(self.pca_.explained_variance_ratio_[0]),
        }


def make_preprocessor(num: list, cat: list, scale: bool = True) -> ColumnTransformer:
    """Числовые: медиана + стандартизация; категориальные: мода + one-hot."""
    numeric = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric.append(("scale", StandardScaler()))
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer([("num", Pipeline(numeric), num), ("cat", categorical, cat)])


def make_pipeline(
    model,
    feature_set: str = "new",
    drop: tuple[str, ...] = (),
    scale: bool = True,
    native_cat: bool = False,
) -> Pipeline:
    """FeatureBuilder → предобработка → модель.

    native_cat=True — для CatBoost: категориальные признаки передаются строками
    без one-hot, модель кодирует их сама (см. bank_marketing.models.CatBoostModel).
    """
    num, cat = feature_columns(feature_set, drop)
    if native_cat:
        prep = ColumnTransformer(
            [
                ("num", "passthrough", num),
                ("cat", "passthrough", cat),
            ],
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")
    else:
        prep = make_preprocessor(num, cat, scale=scale)
    return Pipeline([("features", FeatureBuilder()), ("prep", prep), ("model", model)])


def main(argv: list[str]) -> None:
    if len(argv) != 2:
        sys.exit("использование: python -m bank_marketing.features <файл.csv>")
    X, y = prepare(argv[1])
    print(f"{argv[1]}: {len(X)} записей, {X.shape[1]} столбцов, доля отклика {y.mean():.2%}")
    print(X.head().to_string())


if __name__ == "__main__":
    main(sys.argv)

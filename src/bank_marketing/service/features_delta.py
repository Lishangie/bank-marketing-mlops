"""Признак euribor_delta_3m для моделей, переобучаемых системой (ПР5).

Вывод ПР4: модель должна видеть не уровень ставки, вышедший за область
обучения, а её изменение. Набор признаков — набор new из ПР3 плюс
euribor_delta_3m; пайплайн ПР3 (FeatureBuilder → предобработка → модель) не
меняется, поэтому стадии ПР3–ПР4 не пересчитываются.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from bank_marketing.features import FeatureBuilder, feature_columns, make_preprocessor


def monthly_rates(X: pd.DataFrame) -> pd.Series:
    """Средняя ставка euribor3m по периодам (внутри месяца она постоянна)."""
    return X.groupby("period")["euribor3m"].mean().sort_index()


def euribor_delta(period: pd.Series, rates: pd.Series) -> pd.Series:
    """Ставка периода минус ставка самого раннего известного периода из трёх
    предыдущих месяцев; если таких нет — 0 (начало истории)."""
    months = pd.PeriodIndex(rates.index, freq="M")
    delta = {}
    for p, rate in rates.items():
        cur = pd.Period(p, "M")
        earlier = rates[(months >= cur - 3) & (months < cur)]
        delta[p] = float(rate - earlier.iloc[0]) if len(earlier) else 0.0
    return period.map(delta).astype("float64")


class DeltaFeatureBuilder(FeatureBuilder):
    """FeatureBuilder + euribor_delta_3m.

    В fit запоминается ставка по периодам обучающих данных; в transform история
    дополняется периодами самой таблицы, поэтому признак считается и для
    порции, и для одного клиента (API передаёт period и euribor3m).
    """

    def fit(self, X: pd.DataFrame, y=None):
        super().fit(X, y)
        self.rates_ = monthly_rates(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = super().transform(X)
        new = monthly_rates(X)
        rates = pd.concat([self.rates_, new[~new.index.isin(self.rates_.index)]]).sort_index()
        out["euribor_delta_3m"] = euribor_delta(out["period"], rates)
        return out


def make_delta_pipeline(model, native_cat: bool = False) -> Pipeline:
    """Пайплайн ПР3 с DeltaFeatureBuilder и набором new + euribor_delta_3m."""
    num, cat = feature_columns("new")
    num = [*num, "euribor_delta_3m"]
    if native_cat:
        prep = ColumnTransformer(
            [("num", "passthrough", num), ("cat", "passthrough", cat)],
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")
    else:
        prep = make_preprocessor(num, cat)
    return Pipeline([("features", DeltaFeatureBuilder()), ("prep", prep), ("model", model)])

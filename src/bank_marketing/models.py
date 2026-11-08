"""Обёртки моделей для единой схемы экспериментов."""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin


class CatBoostModel(ClassifierMixin, BaseEstimator):
    """CatBoost с совместимым со scikit-learn клонированием.

    Категориальные признаки определяются по типу столбцов (строки) в fit,
    поэтому cat_features не нужно передавать заранее.
    """

    def __init__(
        self,
        iterations: int = 500,
        depth: int = 6,
        learning_rate: float | None = None,
        l2_leaf_reg: float = 3.0,
        auto_class_weights: str | None = None,
        random_seed: int = 23,
        thread_count: int = 4,
    ):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.auto_class_weights = auto_class_weights
        self.random_seed = random_seed
        self.thread_count = thread_count

    def fit(self, X: pd.DataFrame, y):
        cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        self.model_ = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            auto_class_weights=self.auto_class_weights,
            random_seed=self.random_seed,
            thread_count=self.thread_count,
            allow_writing_files=False,
            verbose=0,
        )
        self.model_.fit(X, np.asarray(y), cat_features=cat)
        self.classes_ = self.model_.classes_
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X).astype(int).ravel()

def prepare(path, ref=None):
    """Любой файл исходной части или порции -> (X, y) для make_pipeline."""
    raw = pd.read_csv(path, sep=";", dtype=str)
    df, results = validate(raw, ref or load_reference(), last_period=None)
    if df is None:
        raise ValueError(f"{path}: файл не прошёл проверку")
    df = clean(df).sort_values("row_id").reset_index(drop=True)
    return df, (df.pop("y") == "yes").astype(int)

class FeatureBuilder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):           # параметры — только по обучающим данным
        self.cap_ = float(X["campaign"].quantile(self.cap_q))
        self.scaler_ = StandardScaler().fit(X[PCA_COLUMNS])
        self.pca_ = PCA(n_components=1).fit(self.scaler_.transform(X[PCA_COLUMNS]))
        return self

    def transform(self, X):
        out = X.copy()
        out["had_prev_contact"] = out["pdays"].notna().astype(int)
        out["campaign_capped"] = out["campaign"].clip(upper=self.cap_)
        out["age_group"] = pd.cut(out["age"], AGE_BINS, labels=AGE_LABELS).astype(str)
        out["job_group"] = out["job"].where(out["job"].isin(JOB_GROUPS), "other")
        pc = self.pca_.transform(self.scaler_.transform(out[PCA_COLUMNS]))[:, 0]
        out["macro_pc1"] = self.sign_ * pc
        return out

def make_pipeline(model, feature_set="new", drop=(), native_cat=False):
    num, cat = feature_columns(feature_set, drop)
    if native_cat:                      # CatBoost кодирует категории сам
        prep = ColumnTransformer([("num", "passthrough", num),
                                  ("cat", "passthrough", cat)],
                                 verbose_feature_names_out=False
                                 ).set_output(transform="pandas")
    else:                               # медиана + стандартизация, мода + one-hot
        prep = make_preprocessor(num, cat)
    return Pipeline([("features", FeatureBuilder()), ("prep", prep),
                     ("model", model)])

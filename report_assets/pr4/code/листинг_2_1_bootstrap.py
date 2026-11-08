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

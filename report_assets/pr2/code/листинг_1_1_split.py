def initial_months(counts: pd.Series, share: float) -> int:
    """Число первых месяцев, доля записей которых ближе всего к share."""
    cum = counts.cumsum().to_numpy() / counts.sum()
    k = int(np.argmin(np.abs(cum - share))) + 1
    if not SHARE_MIN <= cum[k - 1] <= SHARE_MAX:
        raise ValueError(
            f"доля исходной части {cum[k - 1]:.2%} вне {SHARE_MIN:.0%}–{SHARE_MAX:.0%}"
        )
    return k


def batch_bounds(counts: np.ndarray, n: int) -> tuple[int, ...]:
    """Индексы начала порций 2…n: минимум максимальной порции, затем минимум размаха.

    Перебираются все сочетания n − 1 границ между соседними месяцами.
    """
    best, best_key = (), None
    for cuts in itertools.combinations(range(1, len(counts)), n - 1):
        sizes = [int(s.sum()) for s in np.split(counts, cuts)]
        key = (max(sizes), max(sizes) - min(sizes))
        if best_key is None or key < best_key:
            best, best_key = cuts, key
    return best


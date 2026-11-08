def check_ranges(df: pd.DataFrame) -> tuple[CheckResult, pd.DataFrame]:
    """D-04: допустимые диапазоны; записи вне диапазонов исключаются."""
    rules = {
        "age": ~df["age"].between(17, 100),
        "campaign": df["campaign"] < 1,
        "previous": df["previous"] < 0,
        "pdays": ~df["pdays"].between(0, 999),
        "duration": df["duration"] < 0,
    }
    bad = pd.concat(rules, axis=1)
    details = "; ".join(f"{c} {int(v.sum())}" for c, v in bad.items() if v.any())
    mask = bad.any(axis=1)
    return _result("D-04", mask.sum(), len(df), FIXED, details=details), df[~mask]

def check_macro_range(df: pd.DataFrame, ref: dict) -> CheckResult:
    """D-08: доля записей с макропоказателем вне [min; max] исходной части ≤ 5 %."""
    shares = {c: out_of_range_share(df[c], *ref["ranges"][c]) for c in RANGE_FEATURES}
    n = sum(v > OUT_OF_RANGE_MAX for v in shares.values())
    details = "; ".join(f"{c} {v:.1%}" for c, v in shares.items())
    return _result("D-08", n, len(RANGE_FEATURES), WARNING, "признаков", details)

def check_drift(df: pd.DataFrame, ref: dict) -> CheckResult:
    """D-09: PSI age и campaign относительно исходной части ≤ 0,2."""
    values = {
        c: psi_from_shares(ref["psi"][c]["shares"], df[c], ref["psi"][c]["bins"])
        for c in DRIFT_FEATURES
    }
    n = sum(v > PSI_MAX for v in values.values())
    details = "; ".join(f"{c} {v:.3f}" for c, v in values.items())
    return _result("D-09", n, len(DRIFT_FEATURES), WARNING, "признаков", details)

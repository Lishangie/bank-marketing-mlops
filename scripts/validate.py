"""Проверка требований к качеству D-01…D-08 по исходной части и порциям.

Порции обрабатываются по порядку; D-02 сравнивает периоды порции с последним
периодом последней принятой части. Справочник исходной части сохраняется в
models/reference/data_reference.json.

Результаты: report_assets/pr2/validate_out.txt, reports/validation/<часть>.json.
"""

import json
import statistics
import time
import tracemalloc

import pandas as pd

from bank_marketing.config import (
    CSV_SEP,
    INITIAL_FILE,
    PR2_DIR,
    REFERENCE_FILE,
    ROOT,
    VALIDATION_DIR,
)
from bank_marketing.data import batch_files
from bank_marketing.report import emit
from bank_marketing.validation import (
    CRITICAL,
    FIXED,
    OK,
    RULES,
    WARNING,
    build_reference,
    check_schema,
    to_jsonable,
    validate,
    verdict,
)

REPEATS = 5  # повторов для замера времени, берётся медиана
STATUS_RU = {OK: "ok", FIXED: "исправлено", WARNING: "предупр.", CRITICAL: "КРИТИЧНО"}
TARGET_ROWS = 10_000


def read_raw(path) -> pd.DataFrame:
    return pd.read_csv(path, sep=CSV_SEP, dtype=str)


def timed(raw: pd.DataFrame, ref: dict | None, last: str | None) -> tuple[float, float]:
    """Медиана времени проверки (мс) и пиковая память Python-аллокаций (МБ)."""
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        validate(raw, ref, last)
        times.append((time.perf_counter() - t0) * 1000)
    tracemalloc.start()
    validate(raw, ref, last)
    peak = tracemalloc.get_traced_memory()[1] / 2**20
    tracemalloc.stop()
    return statistics.median(times), peak


def main() -> None:
    raws = {"initial": read_raw(INITIAL_FILE)}
    raws |= {p.stem: read_raw(p) for p in batch_files()}

    # Справочник строится по исходной части после приведения типов (D-01).
    _, init_df = check_schema(raws["initial"])
    if init_df is None:
        raise ValueError("исходная часть не проходит D-01")
    ref = build_reference(init_df)
    REFERENCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    REFERENCE_FILE.write_text(
        json.dumps(to_jsonable(ref), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    last = None
    report = {}
    for name, raw in raws.items():
        is_initial = name == "initial"
        df, results = validate(raw, None if is_initial else ref, last)
        ms, peak = timed(raw, None if is_initial else ref, last)
        report[name] = {
            "part": name,
            "rows_in": len(raw),
            "rows_out": None if df is None else len(df),
            "periods": sorted(raw["period"].unique().tolist()),
            "last_processed_before": last,
            "verdict": verdict(results),
            "time_ms": round(ms, 1),
            "peak_mb": round(peak, 1),
            "checks": [r.to_dict() for r in results],
        }
        if df is not None:
            last = str(df["period"].max())
        (VALIDATION_DIR / f"{name}.json").write_text(
            json.dumps(to_jsonable(report[name]), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # Оценка на 10 000 записей: самая большая порция, сдублированная до ≥ 10 000 строк.
    big = max((n for n in raws if n != "initial"), key=lambda n: len(raws[n]))
    reps = -(-TARGET_ROWS // len(raws[big]))
    doubled = pd.concat([raws[big]] * reps, ignore_index=True)
    doubled["row_id"] = [str(i) for i in range(1, len(doubled) + 1)]
    ms_doubled, peak_doubled = timed(doubled, ref, ref["last_period"])

    names = list(raws)
    width = 19
    out = [
        "Проверка качества данных: требования D-01…D-08",
        f"Справочник: {REFERENCE_FILE.relative_to(ROOT).as_posix()} (по исходной части, "
        f"последний период {ref['last_period']})",
        "",
        "Нарушения: число (доля от записей или от проверяемых признаков) и статус",
        f"{'ID':<6}" + "".join(f"{n:>{width}}" for n in names),
    ]
    for rule in RULES:
        cells, stats_ = [], []
        for n in names:
            r = next((c for c in report[n]["checks"] if c["rule"] == rule), None)
            if r is None:
                cells.append("не проверялось")
                stats_.append("—")
                continue
            share = (
                f"{r['share']:.2%}" if r["unit"] == "записей" else f"{r['violations']}/{r['base']}"
            )
            cells.append(f"{r['violations']} ({share})")
            stats_.append(STATUS_RU[r["status"]])
        out.append(f"{rule:<6}" + "".join(f"{c:>{width}}" for c in cells))
        out.append(f"{'':<6}" + "".join(f"{s:>{width}}" for s in stats_))
    out.append(
        f"{'строк':<6}"
        + "".join(
            f"{str(report[n]['rows_in']) + ' → ' + str(report[n]['rows_out']):>{width}}"
            for n in names
        )
    )
    out += ["", "Требования:"]
    out += [f"  {k}: {v}" for k, v in RULES.items()]
    out += ["", "Вердикт по частям, время проверки (медиана из 5) и пиковая память:"]
    for n in names:
        r = report[n]
        out.append(
            f"  {n:<9} {r['verdict']:<27} {r['time_ms']:>7.1f} мс  {r['peak_mb']:>6.1f} МБ  "
            f"({r['rows_in']} строк)"
        )
    out += ["", "Подробности нарушений:"]
    for n in names:
        for c in report[n]["checks"]:
            if c["violations"] or "новые" in c["details"]:
                out.append(f"  {n} {c['rule']}: {c['details'] or c['action']}")
    per_10k_prop = report[big]["time_ms"] * TARGET_ROWS / report[big]["rows_in"]
    per_10k_dup = ms_doubled * TARGET_ROWS / len(doubled)
    out += [
        "",
        f"Оценка на 10 000 записей (самая большая порция {big}, "
        f"{report[big]['rows_in']} строк, {report[big]['time_ms']:.1f} мс):",
        f"  пропорционально: {per_10k_prop:.1f} мс",
        f"  прогон на сдублированной порции ({len(doubled)} строк): {ms_doubled:.1f} мс, "
        f"на 10 000 записей {per_10k_dup:.1f} мс, пиковая память {peak_doubled:.1f} МБ",
    ]
    emit(out, PR2_DIR / "validate_out.txt")


if __name__ == "__main__":
    main()

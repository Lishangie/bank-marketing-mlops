"""Проверка требований к качеству D-01…D-09 по исходной части и порциям.

Порции обрабатываются по порядку; D-02 сравнивает периоды порции с последним
периодом последней принятой части. Справочник исходной части сохраняется в
models/reference/data_reference.json. Замеры времени — scripts/benchmark.py.

Результаты: report_assets/pr2/validate_out.txt, reports/validation/<часть>.json.
"""

import json

from bank_marketing.config import PR2_DIR, REFERENCE_FILE, ROOT, VALIDATION_DIR
from bank_marketing.data import load_raw_parts
from bank_marketing.report import emit
from bank_marketing.validation import (
    ACTIONS,
    CRITICAL,
    FIXED,
    OK,
    RULES,
    WARNING,
    reference_from_initial,
    to_jsonable,
    validate,
    verdict,
)

STATUS_RU = {OK: "ok", FIXED: "исправлено", WARNING: "предупр.", CRITICAL: "КРИТИЧНО"}
WIDTH = 19


def write_json(path, obj) -> None:
    path.write_text(json.dumps(to_jsonable(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    raws = load_raw_parts()
    ref = reference_from_initial(raws["initial"])
    REFERENCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    write_json(REFERENCE_FILE, ref)

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    last = None
    report = {}
    for name, raw in raws.items():
        df, results = validate(raw, None if name == "initial" else ref, last)
        report[name] = {
            "part": name,
            "rows_in": len(raw),
            "rows_out": None if df is None else len(df),
            "periods": sorted(raw["period"].unique().tolist()),
            "last_processed_before": last,
            "verdict": verdict(results),
            "checks": [r.to_dict() for r in results],
        }
        if df is not None:
            last = str(df["period"].max())
        write_json(VALIDATION_DIR / f"{name}.json", report[name])

    names = list(raws)
    out = [
        "Проверка качества данных: требования D-01…D-09",
        f"Справочник: {REFERENCE_FILE.relative_to(ROOT).as_posix()} (по исходной части, "
        f"последний период {ref['last_period']})",
        "",
        "Нарушения: число (доля от записей или от проверяемых признаков) и статус",
        f"{'ID':<6}" + "".join(f"{n:>{WIDTH}}" for n in names),
    ]
    for rule in RULES:
        cells, statuses = [], []
        for n in names:
            r = next((c for c in report[n]["checks"] if c["rule"] == rule), None)
            if r is None:
                cells.append("не проверялось")
                statuses.append("—")
                continue
            share = (
                f"{r['share']:.2%}" if r["unit"] == "записей" else f"{r['violations']}/{r['base']}"
            )
            cells.append(f"{r['violations']} ({share})")
            statuses.append(STATUS_RU[r["status"]])
        out.append(f"{rule:<6}" + "".join(f"{c:>{WIDTH}}" for c in cells))
        out.append(f"{'':<6}" + "".join(f"{s:>{WIDTH}}" for s in statuses))
    rows = [f"{report[n]['rows_in']} → {report[n]['rows_out']}" for n in names]
    out.append(f"{'строк':<6}" + "".join(f"{r:>{WIDTH}}" for r in rows))
    out += ["", "Требования и действия при нарушении:"]
    out += [f"  {k}: {v} → {ACTIONS[k]}" for k, v in RULES.items()]
    out += ["", "Вердикт по частям:"]
    out += [f"  {n:<9} {report[n]['verdict']}" for n in names]
    out += ["", "Подробности (нарушения; новые значения; значения D-08, D-09):"]
    for n in names:
        for c in report[n]["checks"]:
            if c["violations"] or "новые" in c["details"] or c["rule"] in ("D-08", "D-09"):
                out.append(f"  {n} {c['rule']}: {c['details'] or c['action']}")
    emit(out, PR2_DIR / "validate_out.txt")


if __name__ == "__main__":
    main()

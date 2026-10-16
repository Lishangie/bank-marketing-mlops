"""Замер времени проверки качества (вне dvc.yaml: результат зависит от машины и нагрузки).

Время validate() для исходной части, каждой порции и порции ровно из 10 000
записей (самая большая порция, повторённая и обрезанная), медиана из 5 прогонов;
пиковая память Python-аллокаций (tracemalloc) — отдельным прогоном.

Результат: report_assets/pr2/benchmark_out.txt.
"""

import platform
import statistics
import time
import tracemalloc

import pandas as pd

from bank_marketing.config import PR2_DIR
from bank_marketing.data import load_raw_parts
from bank_marketing.report import emit
from bank_marketing.validation import reference_from_initial, validate

REPEATS = 5
TARGET_ROWS = 10_000


def measure(raw: pd.DataFrame, ref: dict | None, last: str | None) -> tuple[float, list, float]:
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        validate(raw, ref, last)
        times.append((time.perf_counter() - t0) * 1000)
    tracemalloc.start()
    validate(raw, ref, last)
    peak = tracemalloc.get_traced_memory()[1] / 2**20
    tracemalloc.stop()
    return statistics.median(times), times, peak


def main() -> None:
    raws = load_raw_parts()
    ref = reference_from_initial(raws["initial"])

    big = max((n for n in raws if n != "initial"), key=lambda n: len(raws[n]))
    reps = -(-TARGET_ROWS // len(raws[big]))
    sample = pd.concat([raws[big]] * reps, ignore_index=True).iloc[:TARGET_ROWS].copy()
    sample["row_id"] = [str(i) for i in range(1, TARGET_ROWS + 1)]

    cases = {"initial": (raws["initial"], None, None)}
    cases |= {n: (raws[n], ref, ref["last_period"]) for n in raws if n != "initial"}
    cases[f"10 000 ({big} ×{reps})"] = (sample, ref, ref["last_period"])

    out = [
        f"Время проверки качества D-01…D-09: медиана из {REPEATS} прогонов",
        f"Python {platform.python_version()}, {platform.system()} {platform.release()}, "
        f"{platform.processor() or platform.machine()}",
        "",
        f"  {'часть':<24}{'строк':>7}{'медиана, мс':>13}{'мс на 10 000':>14}{'память, МБ':>12}"
        "   все прогоны, мс",
    ]
    for name, (raw, r, last) in cases.items():
        med, times, peak = measure(raw, r, last)
        out.append(
            f"  {name:<24}{len(raw):>7}{med:>13.1f}{med * TARGET_ROWS / len(raw):>14.1f}"
            f"{peak:>12.1f}   " + ", ".join(f"{t:.1f}" for t in times)
        )
    out += [
        "",
        "Порция 10 000 записей — самая большая порция, повторённая до 10 000 строк;",
        "D-03 в ней находит повторы (как и при реальных дубликатах), что входит в замер.",
    ]
    emit(out, PR2_DIR / "benchmark_out.txt")


if __name__ == "__main__":
    main()

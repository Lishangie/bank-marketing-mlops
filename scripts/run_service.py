"""ПР5, пункты 4, 8, 9: работа системы на всех порциях с нуля.

1. Хранилище и реестр очищаются; v1 — модель ПР3.
2. Испорченные порции ПР4 (нет столбца, период раньше обработанного) — отклонение.
3. Порции 1–4 по очереди: сценарий обновления отдельным процессом (замер с
   запуском интерпретатора), повтор порции 1 — проверка идемпотентности;
   после каждой порции — переобучение, если сработало правило, и снимок базы
   для скриншота панели.
4. Порция из 10 758 записей (порции 3 и 4 вместе) во временной базе — замер N-02.
5. API: обращения, 200 запросов /predict подряд, откат и возврат версии.
6. Автотесты.

Выход: report_assets/pr5/service_out.txt, service.json, img/рис_3_5.png,
data/system/snapshots/*.db.
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bank_marketing.config import ASSETS_DIR, CSV_SEP, INCOMING_DIR, REPORTS_DIR, ROOT
from bank_marketing.data import batch_files
from bank_marketing.plotting import apply_style
from bank_marketing.report import emit
from bank_marketing.service import registry, storage

OUT = ASSETS_DIR / "pr5"
SNAP = ROOT / "data" / "system" / "snapshots"
PY = sys.executable
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


def run(args: list[str], env: dict | None = None) -> tuple[str, float]:
    t0 = time.perf_counter()
    res = subprocess.run(
        [PY, *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", env=env or ENV
    )
    if res.returncode != 0:
        raise RuntimeError(res.stderr)
    return res.stdout, time.perf_counter() - t0


def process(path: Path, env=None) -> tuple[dict, float]:
    out, dt = run(["-m", "bank_marketing.service.process_batch", str(path)], env)
    return json.loads(out), dt


def ru(x: float, d: int = 3) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def reset() -> None:
    shutil.rmtree(ROOT / "data" / "system", ignore_errors=True)
    shutil.rmtree(registry.REGISTRY_DIR, ignore_errors=True)
    registry.CURRENT_FILE.unlink(missing_ok=True)


def http(method: str, url: str, body: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def api_demo(out: list[str]) -> dict:
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        [PY, "-m", "uvicorn", "app.api:app", "--port", str(port)],
        cwd=ROOT,
        env=ENV,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(100):
            try:
                http("GET", base + "/health")
                break
            except OSError:
                time.sleep(0.2)
        row = pd.read_csv(INCOMING_DIR / "batch_04.csv", sep=CSV_SEP).iloc[-1]
        client = {"client_id": f"C-{row['row_id']}"}
        for k in [
            "period",
            "age",
            "job",
            "marital",
            "education",
            "default",
            "housing",
            "loan",
            "contact",
            "month",
            "day_of_week",
            "campaign",
            "pdays",
            "previous",
            "poutcome",
            "emp.var.rate",
            "cons.price.idx",
            "cons.conf.idx",
            "euribor3m",
            "nr.employed",
        ]:
            client[k] = row[k].item() if hasattr(row[k], "item") else row[k]
        lines = []
        code, body = http("GET", base + "/health")
        lines += ["GET /health", f"  -> {code} {json.dumps(body, ensure_ascii=False)}"]
        code, body = http("POST", base + "/predict", client)
        short = {
            k: client[k]
            for k in ["client_id", "period", "age", "job", "contact", "poutcome", "euribor3m"]
        }
        lines += [
            "POST /predict " + json.dumps(short, ensure_ascii=False)[:-1] + ", ...}",
            f"  -> {code} {json.dumps(body, ensure_ascii=False)}",
        ]
        code, body = http("POST", base + "/predict", {**client, "age": 5, "contact": "pager"})
        errs = "; ".join(f"{e['loc'][-1]}: {e['msg']}" for e in body["detail"])
        lines += ['POST /predict {..., "age": 5, "contact": "pager"}', f"  -> {code}: {errs}"]
        code, body = http("GET", base + "/metrics")
        keep = {
            k: body[k]
            for k in [
                "batch",
                "expected_deposits",
                "above_threshold_share",
                "response_rate",
                "lift_k",
            ]
            if k in body
        }
        lines += ["GET /metrics", f"  -> {code} {json.dumps(keep, ensure_ascii=False)[:-1]}, ...}}"]
        # время ответа
        times = []
        for _ in range(200):
            t0 = time.perf_counter()
            http("POST", base + "/predict", client)
            times.append((time.perf_counter() - t0) * 1000)
        # откат и возврат
        before = http("GET", base + "/health")[1]["model_version"]
        run(["-m", "bank_marketing.service.registry", "rollback"])
        after = http("GET", base + "/health")[1]["model_version"]
        p_after = http("POST", base + "/predict", client)[1]["probability"]
        run(["-m", "bank_marketing.service.registry", "activate", before])
        back = http("GET", base + "/health")[1]["model_version"]
        p_back = http("POST", base + "/predict", client)[1]["probability"]
    finally:
        proc.terminate()
        proc.wait()
    out += ["", "5. API", *["  " + s for s in lines]]
    out.append(
        f"  200 запросов /predict: среднее {ru(np.mean(times), 1)} мс, медиана "
        f"{ru(np.median(times), 1)} мс, "
        f"95-й процентиль {ru(np.percentile(times, 95), 1)} мс"
    )
    out.append(
        f"  откат: {before} -> {after} (вероятность клиента {ru(p_after)}), "
        f"возврат activate {before} -> {back} (вероятность {ru(p_back)})"
    )
    return {
        "lines": lines,
        "ms_mean": float(np.mean(times)),
        "ms_median": float(np.median(times)),
        "ms_p95": float(np.percentile(times, 95)),
        "rollback": [before, after, back],
        "p_rollback": p_after,
        "p_back": p_back,
    }


def figure(retr: pd.DataFrame) -> None:
    apply_style(font_size=10, tick_size=10)
    steps = [f"{i + 1}" for i in range(len(retr))]
    x = np.arange(len(steps))
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.2))
    w = 0.38
    for a, (cur, cand, title) in zip(
        ax,
        [
            ("cur_pr_auc", "cand_pr_auc", "PR-AUC"),
            ("cur_calib", "cand_calib", "Прогноз − факт, п. п."),
        ],
        strict=True,
    ):
        k = 100 if "calib" in cur else 1
        a.bar(x - w / 2, retr[cur] * k, w, label="действующая", color="#7f8c8d")
        a.bar(x + w / 2, retr[cand] * k, w, label="претендент", color="#2e86c1")
        a.set_xticks(x, steps)
        a.set_xlabel("Порция")
        a.set_ylabel(title)
        a.axhline(0, color="black", lw=0.6)
        a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−")))
    ax[0].legend(loc="upper left", fontsize=9)
    fig.tight_layout(w_pad=3)
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "img" / "рис_3_5.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    reset()
    SNAP.mkdir(parents=True, exist_ok=True)
    res: dict = {"batches": {}, "rejected": {}, "timing": {}}
    out = ["Работа системы на порциях (ПР5)", ""]

    out.append("1. Испорченные порции")
    for name in ["bad_schema", "bad_period"]:
        r, dt = process(REPORTS_DIR / "quality_demo" / f"{name}.csv")
        res["rejected"][name] = r
        out.append(f"  {name}: {r['status']}, {r['rows']} записей; {r.get('reasons', '')}")
    con = storage.connect()
    out.append(f"  записей в хранилище после отклонения: {storage.counts(con)}")
    con.close()

    out += ["", "2. Порции по очереди"]
    for path in batch_files():
        r, dt = process(path)
        b = r["batch"]
        res["batches"][b] = {**r, "cli_s": dt}
        out.append(
            f"  {b}: {r['status']}, модель {r['model_version']}, {r['rows']} из "
            f"{r['rows_in']} записей, {ru(r['duration_s'], 2)} с внутри сценария, "
            f"{ru(dt, 2)} с с запуском интерпретатора"
        )
        for a in r["alerts"]:
            out.append(f"    ! {a}")
        if b == "batch_01":
            con = storage.connect()
            c1 = storage.counts(con)
            con.close()
            r2, dt2 = process(path)
            con = storage.connect()
            c2 = storage.counts(con)
            con.close()
            same = all(r2[k] == r[k] for k in ["expected_deposits", "pr_auc", "rows"])
            res["idempotency"] = {"before": c1, "after": c2, "same_metrics": same}
            out.append(f"    повтор {b}: до {c1}, после {c2}; показатели совпали: {same}")
        if r["retrain_signal"]:
            rr, dtr = run(["-m", "bank_marketing.service.retrain", b])
            con = storage.connect()
            row = storage.table(con, "retrains").set_index("batch").loc[b].to_dict()
            con.close()
            res["batches"][b]["retrain"] = {**row, "cli_s": dtr}
            out.append(
                f"    переобучение: претендент {row['candidate']} ({row['train_rows']} записей), "
                f"отложено {row['holdout_rows']}; PR-AUC {ru(row['cur_pr_auc'])} -> "
                f"{ru(row['cand_pr_auc'])}, ROC {ru(row['cur_roc_auc'])} -> "
                f"{ru(row['cand_roc_auc'])}, Lift {ru(row['cur_lift_k'], 2)} -> "
                f"{ru(row['cand_lift_k'], 2)}, прогноз − факт {ru(row['cur_calib'] * 100, 1)} -> "
                f"{ru(row['cand_calib'] * 100, 1)} п. п.; {row['decision']}"
                + (f" -> {row['new_version']}" if isinstance(row["new_version"], str) else "")
                + f"; {ru(dtr, 1)} с"
            )
            sel = [s for s in rr.splitlines() if s.startswith("selection_pr_auc")]
            out.append(f"      выбор семейства на первых 70 %: {sel[0].split(': ', 1)[1]}")
        shutil.copy(storage.db_path(), SNAP / f"after_{b}.db")

    con = storage.connect()
    res["counts"] = storage.counts(con)
    hist = storage.history(con)
    retr = storage.table(con, "retrains")
    runs = storage.table(con, "runs")
    con.close()
    out += ["", "3. Ключевые показатели по порциям"]
    cols = [
        "model_version_",
        "rows_saved",
        "expected_deposits",
        "deposits_in_call_list",
        "above_threshold_share",
        "response_rate",
        "pred_mean",
        "pr_auc",
        "roc_auc",
        "lift_k",
        "out_of_domain_share",
        "psi_macro_pc1",
        "psi_contact",
        "psi_age",
        "psi_campaign_capped",
        "psi_cons.conf.idx",
        "unknown_share",
    ]
    hist["model_version_"] = [res["batches"][b]["model_version"] for b in hist.index]
    out.append(hist[cols].T.to_string(float_format=lambda v: f"{v:.4f}"))
    out.append("")
    out.append(f"  записей в хранилище: {res['counts']}")
    out.append("  журнал запусков:")
    for _, r in runs.iterrows():
        out.append(
            f"    {r['run_id']} {r['batch']:<11} {r['status']:<8} {r['rows_in']:>5} "
            f"{r['rows_saved']:>5} {r['model_version']} {r['duration_s']:.2f} с "
            f"{r['message'][:90]}"
        )
    out += ["", "  реестр:", *("    " + s for s in registry.status().splitlines())]

    # 4. N-02: большая порция во временной базе
    with tempfile.TemporaryDirectory() as tmp:
        big = Path(tmp) / "batch_big.csv"
        pd.concat([pd.read_csv(p, sep=CSV_SEP, dtype=str) for p in batch_files()[2:]]).to_csv(
            big, sep=CSV_SEP, index=False
        )
        env = {**ENV, "BANK_DB": str(Path(tmp) / "big.db")}
        rb, dtb = process(big, env)
        res["timing"]["big"] = {"rows": rb["rows_in"], "inside_s": rb["duration_s"], "cli_s": dtb}
        out += [
            "",
            f"4. Порция из {rb['rows_in']} записей: {ru(rb['duration_s'], 2)} с внутри "
            f"сценария, {ru(dtb, 2)} с с запуском интерпретатора",
        ]

    res["api"] = api_demo(out)

    test_out, dt = run(["-m", "pytest", "-q", "-p", "no:cacheprovider"])
    res["tests"] = test_out.strip().splitlines()[-1]
    out += ["", "6. Автотесты (pytest -q)", *("  " + s for s in test_out.strip().splitlines())]

    figure(retr)
    res["history"] = hist.drop(columns="model_version_").to_dict(orient="index")
    res["retrains"] = retr.to_dict(orient="records")
    res["registry"] = registry.status()
    (OUT / "service.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
    )
    emit(out, OUT / "service_out.txt")


if __name__ == "__main__":
    main()

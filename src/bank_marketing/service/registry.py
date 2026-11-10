"""Версии моделей: файл модели model_vN.joblib, карточка vN.json и указатель
models/current.json на действующую версию.

Карточка хранит происхождение модели: алгоритм и параметры, на каких частях
данных и строках она обучена (с контрольной суммой row_id), коммит Git, порог,
метрики сравнения и профиль обучающих данных (область значений ставки и бины
PSI) — по нему мониторинг считает дрейф относительно данных именно этой модели.

Командная строка:
    python -m bank_marketing.service.registry status
    python -m bank_marketing.service.registry rollback
    python -m bank_marketing.service.registry activate v2
"""

import hashlib
import json
import subprocess
import sys
from functools import lru_cache

import joblib
import pandas as pd
import yaml

from bank_marketing.config import MODELS_DIR, PARAMS_FILE, REPORTS_DIR, ROOT
from bank_marketing.stats import bin_shares, psi_bins
from bank_marketing.validation import to_jsonable

REGISTRY_DIR = MODELS_DIR / "registry"
CURRENT_FILE = MODELS_DIR / "current.json"


def params() -> dict:
    return yaml.safe_load(PARAMS_FILE.read_text(encoding="utf-8"))


def git_commit() -> str:
    """Коммит кода; «+изменения», если в отслеживаемых файлах есть незакоммиченное."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return (head or "—") + ("+изменения" if dirty else "")
    except OSError:
        return "—"


def rows_md5(row_ids) -> str:
    """Контрольная сумма набора строк обучения (row_id по возрастанию)."""
    text = ",".join(str(int(r)) for r in sorted(row_ids))
    return hashlib.md5(text.encode()).hexdigest()


def profile(model, X: pd.DataFrame) -> dict:
    """Профиль обучающих данных модели: область значений ставки и бины PSI."""
    feats = params()["evaluate"]["psi_features"]
    Z = model.named_steps["features"].transform(X)
    psi = {}
    for f in feats:
        spec = psi_bins(Z[f])
        psi[f] = {"bins": spec, "shares": bin_shares(Z[f], spec).tolist()}
    return {"euribor_range": [float(X["euribor3m"].min()), float(X["euribor3m"].max())], "psi": psi}


def card_path(version: str):
    return REGISTRY_DIR / f"{version}.json"


def read_card(version: str) -> dict:
    return json.loads(card_path(version).read_text(encoding="utf-8"))


def write_json(path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(to_jsonable(data), ensure_ascii=False, indent=2)
    path.write_text(text + "\n", encoding="utf-8")


def versions() -> list[str]:
    return sorted((p.stem for p in REGISTRY_DIR.glob("v*.json")), key=lambda v: int(v[1:]))


def current() -> dict:
    """Указатель на действующую версию: {"version", "previous", "history"}."""
    return json.loads(CURRENT_FILE.read_text(encoding="utf-8"))


def _set_current(version: str, reason: str) -> dict:
    from bank_marketing.service.storage import now

    old = current() if CURRENT_FILE.exists() else {"version": None, "history": []}
    data = {
        "version": version,
        "previous": old["version"],
        "activated": now(),
        "reason": reason,
        "history": [
            *old.get("history", []),
            {"version": version, "activated": now(), "reason": reason},
        ],
    }
    write_json(CURRENT_FILE, data)
    return data


@lru_cache(maxsize=8)
def _load(path: str):
    return joblib.load(ROOT / path)


def load(version: str | None = None):
    """(модель, карточка) указанной или действующей версии."""
    card = read_card(version or current()["version"])
    return _load(card["path"]), card


def bootstrap_v1(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Карточка v1 по итогам ПР3–ПР4 (модель не переобучается) и указатель на неё."""
    info = json.loads((MODELS_DIR / "model_v1.json").read_text(encoding="utf-8"))
    model = joblib.load(MODELS_DIR / "model_v1.joblib")
    evaluation = json.loads((REPORTS_DIR / "evaluation.json").read_text(encoding="utf-8"))
    card = {
        "version": "v1",
        "created": info["trained"],
        "path": "models/model_v1.joblib",
        "algorithm": info["algorithm"],
        "family": info["family"],
        "params": info["hyperparameters"],
        "feature_set": "new",
        "threshold": evaluation["threshold"],  # ПР4: t0 = c_FP / (c_FP + c_FN), округлён
        "trained_on": {
            "parts": ["initial (2008-05…2008-07)"],
            "rows": len(X_train),
            "positive_share": float(y_train.mean()),
            "periods": [str(X_train["period"].min()), str(X_train["period"].max())],
            "rows_md5": rows_md5(X_train["row_id"]),
        },
        "git_commit": info.get("git_commit", "—"),
        "note": "итоговая модель ПР3; по ПР4 возвращена на доработку",
        "previous": None,
        "profile": profile(model, X_train),
    }
    write_json(card_path("v1"), card)
    if not CURRENT_FILE.exists() or current()["version"] not in versions():
        CURRENT_FILE.unlink(missing_ok=True)
        _set_current("v1", "начальная версия (ПР3)")
    return card


def register(model, card: dict) -> dict:
    """Сохраняет новую версию и делает её действующей; предыдущая остаётся на диске."""
    version = f"v{max(int(v[1:]) for v in versions()) + 1}"
    path = REGISTRY_DIR / f"model_{version}.joblib"
    joblib.dump(model, path)
    card = {"version": version, "path": path.relative_to(ROOT).as_posix(), **card}
    card["previous"] = current()["version"]
    write_json(card_path(version), card)
    _set_current(version, card.get("reason", "замена по правилу"))
    return card


def rollback() -> dict:
    """Откат: указатель возвращается на предыдущую версию, без переобучения."""
    prev = read_card(current()["version"]).get("previous")
    if not prev:
        raise SystemExit("у действующей версии нет предыдущей")
    return _set_current(prev, "откат")


def activate(version: str) -> dict:
    if version not in versions():
        raise SystemExit(f"нет версии {version}; есть: {', '.join(versions())}")
    return _set_current(version, "ручная активация")


def status() -> str:
    cur = current()
    when = cur["activated"].replace("T", " ")
    lines = [f"действующая версия: {cur['version']} ({cur['reason']}, {when})"]
    for v in versions():
        c = read_card(v)
        mark = "*" if v == cur["version"] else " "
        t = c["trained_on"]
        name = c["algorithm"].split(" (")[0]
        period = f"{t['periods'][0]}…{t['periods'][1]}"
        lines.append(
            f"{mark} {v}: {name}, {t['rows']} строк {period}, предыдущая {c['previous'] or '—'}"
        )
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    cmd = argv[1] if len(argv) > 1 else "status"
    if cmd == "rollback":
        rollback()
    elif cmd == "activate" and len(argv) == 3:
        activate(argv[2])
    elif cmd != "status":
        sys.exit("использование: registry [status | rollback | activate vN]")
    print(status())


if __name__ == "__main__":
    main(sys.argv)

"""Загрузка сырых данных, исходной части и порций."""

from pathlib import Path

import pandas as pd

from bank_marketing.config import CSV_SEP, INCOMING_DIR, INITIAL_FILE, RAW_FILE, TARGET


def load_raw(path: Path = RAW_FILE) -> pd.DataFrame:
    """Читает bank-additional-full.csv; целевая y кодируется как 0/1."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} не найден: выполните `dvc pull` или scripts/download_data.py"
        )
    df = pd.read_csv(path, sep=CSV_SEP)
    df[TARGET] = (df[TARGET] == "yes").astype(int)
    return df


def batch_files() -> list[Path]:
    """Порции data/incoming/batch_*.csv по возрастанию номера."""
    return sorted(INCOMING_DIR.glob("batch_*.csv"))


def load_parts() -> dict[str, pd.DataFrame]:
    """Исходная часть и порции: {"initial": ..., "batch_01": ..., ...}; y — 0/1."""
    files = [INITIAL_FILE, *batch_files()]
    if not INITIAL_FILE.exists() or len(files) == 1:
        raise FileNotFoundError("нет data/initial или data/incoming: выполните `dvc repro`")
    return {path.stem: load_raw(path) for path in files}


def load_raw_parts() -> dict[str, pd.DataFrame]:
    """Исходная часть и порции как строки (для проверки качества, D-01 приводит типы)."""
    files = [INITIAL_FILE, *batch_files()]
    return {path.stem: pd.read_csv(path, sep=CSV_SEP, dtype=str) for path in files}

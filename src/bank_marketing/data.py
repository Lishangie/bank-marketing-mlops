"""Загрузка сырых данных."""

from pathlib import Path

import pandas as pd

from bank_marketing.config import CSV_SEP, RAW_FILE, TARGET


def load_raw(path: Path = RAW_FILE) -> pd.DataFrame:
    """Читает bank-additional-full.csv; целевая y кодируется как 0/1."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} не найден: выполните `dvc pull` или scripts/download_data.py"
        )
    df = pd.read_csv(path, sep=CSV_SEP)
    df[TARGET] = (df[TARGET] == "yes").astype(int)
    return df

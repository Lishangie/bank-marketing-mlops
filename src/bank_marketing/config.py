"""Пути проекта и параметры источника данных."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

ASSETS_DIR = ROOT / "report_assets"
ASSETS_IMG_DIR = ASSETS_DIR / "img"
ASSETS_CODE_DIR = ASSETS_DIR / "code"

RAW_FILE = RAW_DIR / "bank-additional-full.csv"
DATA_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
CSV_SEP = ";"
TARGET = "y"

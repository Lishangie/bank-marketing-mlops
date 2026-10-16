"""Пути проекта, параметры источника данных и состав признаков."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INITIAL_DIR = DATA_DIR / "initial"
INCOMING_DIR = DATA_DIR / "incoming"
INITIAL_FILE = INITIAL_DIR / "initial.csv"

MODELS_DIR = ROOT / "models"
REFERENCE_FILE = MODELS_DIR / "reference" / "data_reference.json"

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
VALIDATION_DIR = REPORTS_DIR / "validation"

ASSETS_DIR = ROOT / "report_assets"
ASSETS_IMG_DIR = ASSETS_DIR / "img"
ASSETS_CODE_DIR = ASSETS_DIR / "code"

# Материалы ПР2: рисунки дублируются в reports/figures/pr2.
PR2_DIR = ASSETS_DIR / "pr2"
PR2_IMG_DIR = PR2_DIR / "img"
PR2_CODE_DIR = PR2_DIR / "code"
PR2_FIGURES_DIR = FIGURES_DIR / "pr2"

PARAMS_FILE = ROOT / "params.yaml"

RAW_FILE = RAW_DIR / "bank-additional-full.csv"
DATA_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
CSV_SEP = ";"
TARGET = "y"

NUMERIC = [
    "age",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]
CATEGORICAL = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]
MACRO = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
# 21 столбец исходного файла в исходном порядке.
RAW_COLUMNS = [
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
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
    "y",
]
# Служебные столбцы порций: номер строки в исходном файле и восстановленный год-месяц.
ID_COLUMNS = ["row_id", "period"]

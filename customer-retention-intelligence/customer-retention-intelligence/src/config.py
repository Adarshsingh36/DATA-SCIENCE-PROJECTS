"""Central configuration: paths, constants and documented business assumptions."""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TELCO_FILE = DATA_RAW / "Telco_customer_churn.xlsx"
BANK_FILE = DATA_RAW / "Bank_Churn_Modelling.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# ---- Business assumptions (all configurable; documented in README / report) ----
CONTRIBUTION_MARGIN = float(os.getenv("CONTRIBUTION_MARGIN", 0.40))   # share of revenue that is margin
ANNUAL_DISCOUNT_RATE = float(os.getenv("ANNUAL_DISCOUNT_RATE", 0.10))  # for CLV discounting
CLV_HORIZON_MONTHS = int(os.getenv("CLV_HORIZON_MONTHS", 36))
HIGH_RISK_THRESHOLD = 0.50     # risk tier boundaries on calibrated churn probability
MEDIUM_RISK_THRESHOLD = 0.25

# ---- Simulated uplift experiment (NOT real experimental data) ----
SIM_TREATMENT_PROB = 0.50
SIM_REFERENCE_DISCOUNT_PCT = 10.0
SIM_MAX_RESPONSE = 0.45

for _d in (DATA_PROCESSED, MODELS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

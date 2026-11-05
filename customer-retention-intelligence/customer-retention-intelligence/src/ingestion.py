"""Automated ingestion, validation and loading into the database."""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text, inspect
from sqlalchemy.engine import Engine

from src import config
from src.preprocessing import RAW_RENAME, clean_raw, engineer_features
from database import models
from database.connection import get_engine

log = logging.getLogger("ingestion")

EXPECTED_CATEGORIES = {
    "gender": {"Male", "Female"},
    "contract": {"Month-to-month", "One year", "Two year"},
    "internet_service": {"DSL", "Fiber optic", "No"},
    "payment_method": {"Mailed check", "Electronic check", "Bank transfer (automatic)", "Credit card (automatic)"},
    "churn_label": {"Yes", "No"},
}


@dataclass
class DataQualityReport:
    n_rows: int = 0
    n_columns: int = 0
    duplicate_customer_ids: int = 0
    duplicate_rows: int = 0
    missing_by_column: dict = field(default_factory=dict)
    blank_total_charges: int = 0
    blank_total_charges_all_zero_tenure: bool = True
    invalid_categories: dict = field(default_factory=dict)
    range_violations: dict = field(default_factory=dict)
    churn_label_value_mismatch: int = 0
    total_charges_inconsistent_gt_50pct: int = 0
    churn_rate: float = 0.0
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors

    def to_dict(self):
        d = asdict(self); d["passed"] = self.passed
        return d


def load_raw_excel(path: Path | str = config.TELCO_FILE) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_excel(path)
    return df


def validate(raw: pd.DataFrame) -> DataQualityReport:
    """Schema, dtype, category, range, duplicate and integrity checks. Does not modify data."""
    r = DataQualityReport(n_rows=len(raw), n_columns=raw.shape[1])
    missing_cols = sorted(set(RAW_RENAME) - set(raw.columns))
    if missing_cols:
        r.errors.append(f"Missing columns: {missing_cols}")
        return r
    d = raw.rename(columns=RAW_RENAME)
    r.duplicate_customer_ids = int(d["customer_id"].duplicated().sum())
    r.duplicate_rows = int(d.duplicated().sum())
    if r.duplicate_customer_ids:
        r.errors.append(f"{r.duplicate_customer_ids} duplicate customer IDs")
    miss = d.isna().sum(); r.missing_by_column = {k: int(v) for k, v in miss.items() if v}
    blank = d["total_charges"].astype(str).str.strip().eq("")
    r.blank_total_charges = int(blank.sum())
    r.blank_total_charges_all_zero_tenure = bool(d.loc[blank, "tenure_months"].eq(0).all())
    if r.blank_total_charges:
        r.warnings.append(f"{r.blank_total_charges} blank Total Charges (all tenure=0: {r.blank_total_charges_all_zero_tenure}); imputed as 0.0")
    for col, allowed in EXPECTED_CATEGORIES.items():
        bad = set(d[col].dropna().unique()) - allowed
        if bad:
            r.invalid_categories[col] = sorted(map(str, bad)); r.errors.append(f"Unexpected values in {col}: {bad}")
    tc = pd.to_numeric(d["total_charges"], errors="coerce")
    checks = {"tenure_months<0": (d["tenure_months"] < 0).sum(), "tenure_months>120": (d["tenure_months"] > 120).sum(),
              "monthly_charges<=0": (d["monthly_charges"] <= 0).sum(), "total_charges<0": (tc < 0).sum(),
              "churn_value_not_0_1": (~d["churn_value"].isin([0, 1])).sum()}
    r.range_violations = {k: int(v) for k, v in checks.items() if v}
    if r.range_violations:
        r.errors.append(f"Range violations: {r.range_violations}")
    r.churn_label_value_mismatch = int((d["churn_label"].eq("Yes").astype(int) != d["churn_value"]).sum())
    if r.churn_label_value_mismatch:
        r.errors.append("Churn Label and Churn Value disagree")
    exp = d["tenure_months"] * d["monthly_charges"]
    ok = (tc.notna()) & (exp > 0)
    r.total_charges_inconsistent_gt_50pct = int((((tc - exp).abs() / exp)[ok] > 0.5).sum())
    if r.total_charges_inconsistent_gt_50pct:
        r.warnings.append(f"{r.total_charges_inconsistent_gt_50pct} rows where total_charges differs >50% from tenure*monthly "
                          "(plausible: plan changes over time); kept as observed.")
    r.churn_rate = float(d["churn_value"].mean())
    return r


def _raw_for_db(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw.rename(columns=RAW_RENAME).copy()
    d["total_charges"] = d["total_charges"].astype(str).str.strip()  # kept as text, exactly as supplied
    return d[[c.name for c in models.raw_customers.columns]]


def _chunked_insert(df: pd.DataFrame, table: str, engine: Engine, chunk: int = 2000):
    df.to_sql(table, engine, if_exists="append", index=False, chunksize=chunk, method="multi")


def ingest(path: Path | str = config.TELCO_FILE, engine: Engine | None = None, reset: bool = True) -> dict:
    """Validate and load RAW and PROCESSED layers. Returns a summary dict (also written to reports/)."""
    engine = engine or get_engine()
    raw = load_raw_excel(path)
    report = validate(raw)
    if not report.passed:
        raise ValueError(f"Data validation failed: {report.errors}")
    # Deterministic row order (sorted by ID) so that every downstream seeded procedure is reproducible
    processed = engineer_features(clean_raw(raw)).sort_values("customer_id").reset_index(drop=True)
    cols = [c.name for c in models.customers.columns]
    models.metadata.create_all(engine)
    if reset:
        with engine.begin() as conn:
            for t in models.TABLE_ORDER:
                conn.execute(text(f"DELETE FROM {t}"))
    _chunked_insert(_raw_for_db(raw), "raw_customers", engine)
    _chunked_insert(processed[cols], "customers", engine)
    processed.to_csv(config.DATA_PROCESSED / "customers_processed.csv", index=False)
    with engine.connect() as conn:
        n_raw = conn.execute(text("SELECT COUNT(*) FROM raw_customers")).scalar()
        n_proc = conn.execute(text("SELECT COUNT(*) FROM customers")).scalar()
    summary = {"rows_raw_table": int(n_raw), "rows_customers_table": int(n_proc), "backend": engine.dialect.name,
               "quality": report.to_dict()}
    assert n_raw == n_proc == len(raw), "Row-count integrity check failed"
    (config.REPORTS_DIR / "data_quality_report.json").write_text(json.dumps(summary, indent=2))
    log.info("Ingested %s rows into %s", n_proc, engine.dialect.name)
    return summary


def read_customers(engine: Engine | None = None) -> pd.DataFrame:
    df = pd.read_sql_table("customers", engine or get_engine())
    df.columns = [str(c) for c in df.columns]
    return df.sort_values("customer_id").reset_index(drop=True)


def bank_dataset_summary(path: Path | str = config.BANK_FILE) -> dict:
    """Inspection only; the bank dataset is the documented alternative, not used for modelling."""
    b = pd.read_csv(path)
    return {"rows": len(b), "columns": list(b.columns), "churn_rate": float(b["Exited"].mean()),
            "has_tenure_years_only": True, "has_monthly_charges": False}

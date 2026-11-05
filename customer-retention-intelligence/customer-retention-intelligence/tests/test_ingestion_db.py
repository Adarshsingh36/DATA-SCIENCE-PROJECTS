import os
import pandas as pd
import pytest
from sqlalchemy import create_engine, text, inspect
from src import ingestion, config
from src.preprocessing import RAW_RENAME
from database import models


def test_raw_file_preserved():
    assert config.TELCO_FILE.exists() and config.BANK_FILE.exists()


def test_validation_passes_on_real_data(raw):
    r = ingestion.validate(raw)
    assert r.passed and r.n_rows == 7043 and r.duplicate_customer_ids == 0
    assert r.blank_total_charges == 11 and r.blank_total_charges_all_zero_tenure


def test_validation_detects_duplicates_and_bad_values(raw):
    bad = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    assert not ingestion.validate(bad).passed
    bad = raw.copy(); bad.loc[0, "Contract"] = "Weekly"
    assert "contract" in ingestion.validate(bad).invalid_categories
    bad = raw.copy(); bad.loc[0, "Tenure Months"] = -3
    assert not ingestion.validate(bad).passed


def test_validation_missing_columns(raw):
    assert not ingestion.validate(raw.drop(columns=["Contract"])).passed


def test_ingest_into_sqlite_temp(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path/'t.db'}")
    s = ingestion.ingest(engine=eng)
    assert s["rows_raw_table"] == s["rows_customers_table"] == 7043 and s["backend"] == "sqlite"
    with eng.connect() as c:
        assert c.execute(text("SELECT COUNT(DISTINCT customer_id) FROM customers")).scalar() == 7043
        # leakage columns must not be in processed table
        cols = {x["name"] for x in inspect(eng).get_columns("customers")}
    assert not ({"churn_score", "churn_reason", "cltv", "churn_label"} & cols)


def test_ingest_idempotent(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path/'t.db'}")
    ingestion.ingest(engine=eng); s = ingestion.ingest(engine=eng)
    assert s["rows_customers_table"] == 7043


def test_all_required_tables_defined():
    required = {"customers", "churn_predictions", "survival_predictions", "uplift_predictions", "customer_segments",
                "retention_campaigns", "retention_results"}
    assert required <= set(models.metadata.tables)


def _pg_ok():
    try:
        from database.connection import get_engine, backend_name
        return backend_name() == "postgresql"
    except Exception:
        return False


pg = pytest.mark.skipif(not _pg_ok(), reason="PostgreSQL not reachable with current .env")


@pg
def test_postgres_connectivity_and_tables():
    from database.connection import check_connection, get_engine
    info = check_connection()
    assert info["ok"] and info["backend"] == "postgresql" and info["database"] == "customer_retention_db"
    tables = set(inspect(get_engine()).get_table_names())
    assert {"raw_customers", "customers", "churn_predictions", "survival_predictions", "uplift_predictions",
            "customer_segments", "retention_campaigns", "retention_results"} <= tables


@pg
def test_postgres_populated_and_consistent():
    from database.connection import get_engine
    with get_engine().connect() as c:
        for t in ["raw_customers", "customers", "churn_predictions", "survival_predictions", "uplift_predictions",
                  "customer_segments", "customer_clv", "churn_explanations"]:
            assert c.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar() == 7043, t
        assert c.execute(text("SELECT COUNT(*) FROM churn_predictions WHERE churn_probability<0 OR churn_probability>1")).scalar() == 0


def test_missing_credentials_raise(monkeypatch):
    from database import connection
    for k in ["DATABASE_URL", "POSTGRES_USER", "POSTGRES_PASSWORD", "FORCE_SQLITE", "ALLOW_SQLITE_FALLBACK"]:
        monkeypatch.delenv(k, raising=False)
    connection.reset_engine_cache()
    with pytest.raises(Exception):
        connection.get_engine()
    connection.reset_engine_cache()


def test_explicit_sqlite_fallback_only_when_enabled(monkeypatch, tmp_path):
    from database import connection
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg2://x:y@127.0.0.1:1/none")
    monkeypatch.delenv("FORCE_SQLITE", raising=False)
    monkeypatch.delenv("ALLOW_SQLITE_FALLBACK", raising=False)
    connection.reset_engine_cache()
    with pytest.raises(ConnectionError):
        connection.get_engine()
    monkeypatch.setenv("ALLOW_SQLITE_FALLBACK", "true"); connection.reset_engine_cache()
    assert connection.get_engine().dialect.name == "sqlite"
    connection.reset_engine_cache()


def test_processed_output_sorted_and_uplift_reproducible(tmp_path):
    import numpy as np
    from src import uplift_model as um
    eng = create_engine(f"sqlite:///{tmp_path/'t.db'}"); ingestion.ingest(engine=eng)
    a = ingestion.read_customers(eng)
    assert a.customer_id.is_monotonic_increasing
    p0 = np.full(len(a), 0.27)
    s1, s2 = um.simulate_experiment(a, p0), um.simulate_experiment(a.sample(frac=1, random_state=1).sort_values("customer_id").reset_index(drop=True), p0)
    assert s1.sim_treatment.equals(s2.sim_treatment)

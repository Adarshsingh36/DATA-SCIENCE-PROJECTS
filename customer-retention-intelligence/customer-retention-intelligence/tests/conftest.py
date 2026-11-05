import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src import config
from src.ingestion import load_raw_excel
from src.preprocessing import prepare_customers


@pytest.fixture(scope="session")
def raw():
    return load_raw_excel(config.TELCO_FILE)


@pytest.fixture(scope="session")
def df(raw):
    return prepare_customers(raw)


@pytest.fixture(scope="session")
def small(df):
    """Stratified ~1800-row subsample for fast model tests."""
    return df.groupby("churn", group_keys=False).apply(lambda g: g.sample(frac=0.26, random_state=0)).reset_index(drop=True)

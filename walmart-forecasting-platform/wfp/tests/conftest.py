import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def make_merged(n_stores=3, n_depts=4, weeks=120, seed=0, gaps=True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-02-05", periods=weeks, freq="7D")
    rows = []
    for s in range(1, n_stores + 1):
        for d in range(1, n_depts + 1):
            base = 1000 * s * d
            y = base * (1 + 0.3 * np.sin(2 * np.pi * np.arange(weeks) / 52)) + rng.normal(0, 50, weeks)
            for i, dt in enumerate(dates):
                if gaps and s == 2 and d == 3 and i in (10, 11):
                    continue
                rows.append((s, d, dt, y[i], dt.month in (11, 12) and dt.day > 24))
    df = pd.DataFrame(rows, columns=["Store", "Dept", "Date", "Weekly_Sales", "IsHoliday"])
    df["Type"] = df.Store.map({1: "A", 2: "B", 3: "C"})
    df["Size"] = df.Store.map({1: 150000, 2: 100000, 3: 40000}).astype(float)
    rng2 = np.random.default_rng(1)
    for c in ["Temperature", "Fuel_Price", "CPI", "Unemployment", "MarkDown_total", "markdown_available"]:
        df[c] = rng2.normal(50, 5, len(df)) if c != "markdown_available" else 0
    return df


@pytest.fixture(scope="session")
def merged():
    return make_merged()


@pytest.fixture(scope="session")
def panel(merged):
    from walmartfc.panel import build_bottom_panel
    return build_bottom_panel(merged)


@pytest.fixture(scope="session")
def cfg():
    from walmartfc.config import load_config
    return load_config()

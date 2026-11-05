import numpy as np
import pandas as pd
import pytest

from walmartfc.data.preprocess import PreprocessingLog, merge_tables, validate_raw


def _tables():
    d = pd.to_datetime(["2010-02-05", "2010-02-12", "2010-02-19"])
    train = pd.DataFrame(dict(Store=[1, 1, 1, 1], Dept=[1, 1, 2, 2], Date=[d[0], d[1], d[0], d[1]],
                              Weekly_Sales=[10.0, -5.0, 7.0, 8.0], IsHoliday=[False, True, False, True]))
    stores = pd.DataFrame(dict(Store=[1], Type=["A"], Size=[1000]))
    feats = pd.DataFrame(dict(Store=[1, 1, 1], Date=d, Temperature=[40, 41, 42], Fuel_Price=[2.5, 2.5, 2.6],
                              MarkDown1=[np.nan, 5.0, -3.0], MarkDown2=[np.nan] * 3, MarkDown3=[np.nan] * 3,
                              MarkDown4=[np.nan] * 3, MarkDown5=[np.nan] * 3, CPI=[211.0, np.nan, 212.0],
                              Unemployment=[8.0, np.nan, 8.1], IsHoliday=[False, True, False]))
    return train, stores, feats


def test_merge_preserves_rows_and_keeps_negatives():
    t, s, f = _tables()
    log = PreprocessingLog()
    m = merge_tables(t, s, f, log)
    assert len(m) == len(t)
    assert (m.Weekly_Sales < 0).sum() == 1            # negative sale retained
    assert m.Type.notna().all() and m.Temperature.notna().all()


def test_markdown_nan_not_median_filled_and_negatives_clipped():
    t, s, f = _tables()
    m = merge_tables(t, s, f)
    assert (m[[f"MarkDown{i}" for i in range(1, 6)]] >= 0).all().all()
    first = m[m.Date == "2010-02-05"].markdown_available.iloc[0]
    assert first == 0                                  # before programme start
    assert m[m.Date == "2010-02-12"].markdown_available.iloc[0] == 1


def test_cpi_forward_fill_is_past_only():
    t, s, f = _tables()
    m = merge_tables(t, s, f)
    assert m[m.Date == "2010-02-12"].CPI.iloc[0] == 211.0   # taken from previous week, not future/median


def test_features_rows_after_training_end_are_dropped():
    t, s, f = _tables()
    extra = f.iloc[[0]].copy(); extra["Date"] = pd.Timestamp("2011-01-07")
    log = PreprocessingLog()
    merge_tables(t, s, pd.concat([f, extra]), log)
    assert any("after last training date" in e["decision"] for e in log.entries)


def test_validate_rejects_duplicate_keys():
    t, s, f = _tables()
    with pytest.raises(ValueError):
        validate_raw(pd.concat([t, t.iloc[[0]]]), s, f)


def test_validate_reports_negative_sales():
    t, s, f = _tables()
    assert validate_raw(t, s, f)["negative_sales_rows"] == 1

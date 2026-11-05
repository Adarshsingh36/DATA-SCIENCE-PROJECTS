import numpy as np
import pandas as pd

from walmartfc.baselines import baseline_forecast
from walmartfc.intervals import (apply_cqr, conformal_q, cqr_calibrate, hbucket, residual_conformal_calibrate,
                                 apply_residual_conformal)


def test_naive_and_snaive_values(panel):
    t0, H = 90, 4
    nv = baseline_forecast(panel, [t0], H, "naive")
    sn = baseline_forecast(panel, [t0], H, "snaive")
    i = 0
    assert np.isclose(nv[(nv.sid == i) & (nv.h == 2)].y_hat.iloc[0], panel.Y[i, t0])
    assert np.isclose(sn[(sn.sid == i) & (sn.h == 2)].y_hat.iloc[0], panel.Y[i, t0 + 2 - 52])


def test_baselines_never_read_future(panel):
    t0 = 90
    a = baseline_forecast(panel, [t0], 6, "snaive")
    Y2 = panel.Y.copy(); Y2[:, t0 + 1:] = 1e9
    p2 = type(panel)(panel.dates, panel.series, Y2, panel.exog, panel.cal)
    b = baseline_forecast(p2, [t0], 6, "snaive")
    # snaive for h<=6 uses t0+h-52 <= t0 -> identical
    assert np.allclose(a.y_hat, b.y_hat, equal_nan=True)


def test_conformal_quantile_finite_sample_rule():
    s = np.arange(1, 10.0)                       # n=9, alpha=.2 -> k=ceil(10*.8)=8
    assert conformal_q(s, 0.2) == 8.0


def test_cqr_restores_coverage_on_exchangeable_data():
    rng = np.random.default_rng(0)
    def mk(n):
        y = rng.normal(0, 1, n)
        return pd.DataFrame(dict(y=y, q_lo=-0.5, q_hi=0.5, scale=1.0, h=rng.integers(1, 14, n), lv=0))
    cal, te = mk(5000), mk(5000)
    c = cqr_calibrate(cal, 0.2, "lv")
    lo, hi = apply_cqr(te, c, "lv")
    cov = ((te.y >= lo) & (te.y <= hi)).mean()
    assert 0.77 < cov < 0.83                     # nominal 0.8


def test_residual_conformal_coverage():
    rng = np.random.default_rng(1)
    def mk(n):
        yh = rng.uniform(50, 150, n)
        return pd.DataFrame(dict(y=yh + rng.normal(0, 10, n), y_hat=yh, scale=100.0, h=rng.integers(1, 14, n), level="a"))
    cal, te = mk(4000), mk(4000)
    c = residual_conformal_calibrate(cal, 0.2, "level")
    lo, hi = apply_residual_conformal(te, c, "level")
    assert 0.77 < ((te.y >= lo) & (te.y <= hi)).mean() < 0.83


def test_horizon_buckets():
    assert list(hbucket(np.array([1, 4, 5, 8, 9, 13]))) == [0, 0, 1, 1, 2, 2]

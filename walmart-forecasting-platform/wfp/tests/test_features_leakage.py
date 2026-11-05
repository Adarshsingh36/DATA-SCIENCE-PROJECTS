import numpy as np

from walmartfc.features import compute_base_stats, feature_columns, make_samples


def _samples(panel, cfg, t0, horizons=(1, 4, 13)):
    base = compute_base_stats(panel, cfg.forecast["rolling_windows"], cfg.forecast["vol_windows"])
    return make_samples(panel, base, [t0], list(horizons), cfg)


def test_features_unchanged_when_future_sales_are_perturbed(panel, cfg):
    """Core leakage test: nothing at or before the target week's *future* may influence features, except lag_52 which is <= t0."""
    t0 = 80
    a = _samples(panel, cfg, t0)
    p2 = type(panel)(panel.dates, panel.series, panel.Y.copy(), panel.exog, panel.cal)
    rng = np.random.default_rng(0)
    p2.Y[:, t0 + 1:] = rng.normal(1e6, 1e5, p2.Y[:, t0 + 1:].shape)      # destroy everything after the origin
    b = _samples(p2, cfg, t0)
    cols = feature_columns(a)
    for c in cols + ["scale"]:
        x, y = a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float)
        assert np.array_equal(np.nan_to_num(x, nan=-9.9), np.nan_to_num(y, nan=-9.9)), f"feature {c} leaks future data"
    assert not np.array_equal(a["y"].fillna(0), b["y"].fillna(0))         # only the target changes


def test_lag_alignment(panel, cfg):
    t0, h = 90, 5
    s = _samples(panel, cfg, t0, horizons=(h,))
    i = 0
    sc = s.scale.to_numpy()[i]
    assert np.isclose(s.lag_1.to_numpy()[i] * sc, panel.Y[i, t0])            # lag_1 = latest observed week
    assert np.isclose(s.lag_4.to_numpy()[i] * sc, panel.Y[i, t0 - 3], rtol=1e-4)
    assert np.isclose(s.lag_52.to_numpy()[i] * sc, panel.Y[i, t0 + h - 52], rtol=1e-4)   # target-aligned seasonal lag


def test_rolling_features_use_only_past_window(panel, cfg):
    t0 = 70
    s = _samples(panel, cfg, t0, horizons=(1,))
    i = 5
    sc = s.scale.to_numpy()[i]
    assert np.isclose(s.mean_8.to_numpy()[i] * sc, np.nanmean(panel.Y[i, t0 - 7: t0 + 1]), rtol=1e-4)
    assert np.isclose(s.std_8.to_numpy()[i] * sc, np.nanstd(panel.Y[i, t0 - 7: t0 + 1], ddof=1), rtol=1e-3)


def test_horizon_limit_enforced(panel, cfg):
    import pytest
    base = compute_base_stats(panel, cfg.forecast["rolling_windows"], cfg.forecast["vol_windows"])
    with pytest.raises(AssertionError):
        make_samples(panel, base, [60], [52], cfg)


def test_future_markdowns_off_by_default(panel, cfg):
    assert "markdown_tgt" not in _samples(panel, cfg, 80).columns

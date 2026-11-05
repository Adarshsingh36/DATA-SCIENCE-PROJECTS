import numpy as np
import pytest

from walmartfc import metrics as M
from walmartfc.splits import make_fold


def test_fold_is_chronological(panel):
    f = make_fold(panel, "x", str(panel.dates[100].date()), 13, 13)
    assert f.val_start < f.test_start and f.test_start + f.H <= panel.T
    for o in f.train_origins(5, f.val_start):
        assert o <= f.val_start - 2
    assert max(f.val_origins(4)) <= f.test_start - 2
    assert f.test_origin == f.test_start - 1


def test_fold_rejects_window_past_end(panel):
    with pytest.raises(ValueError):
        make_fold(panel, "x", str(panel.dates[-3].date()), 13, 13)


def test_wape_mae_rmse_known_values():
    y = np.array([100.0, 200.0, 300.0]); f = np.array([110.0, 180.0, 330.0])
    assert M.wape(y, f) == pytest.approx(60 / 600)
    assert M.mae(y, f) == pytest.approx(20.0)
    assert M.rmse(y, f) == pytest.approx(np.sqrt((100 + 400 + 900) / 3))
    assert M.bias(y, f) == pytest.approx(20 / 600)


def test_metrics_ignore_nan_actuals():
    y = np.array([100.0, np.nan]); f = np.array([90.0, 5.0])
    assert M.wape(y, f) == pytest.approx(0.1)


def test_rmsse_uses_training_naive_scale():
    train = np.array([[1.0, 2.0, 3.0, 4.0]])           # naive one-step squared error = 1
    out = M.rmsse(train, np.array([[5.0, 6.0]]), np.array([[4.0, 5.0]]))
    assert out["RMSSE"] == pytest.approx(1.0)


def test_interval_metrics_coverage():
    y = np.arange(10.0)
    r = M.interval_metrics(y, y - 1, y + 1)
    assert r["coverage"] == 1.0 and r["mean_width"] == 2.0

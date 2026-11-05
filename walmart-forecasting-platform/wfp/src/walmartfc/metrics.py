"""Forecast accuracy metrics. R-squared is deliberately absent: it is not a forecasting-accuracy measure."""
from __future__ import annotations

import numpy as np
import pandas as pd


def wape(y: np.ndarray, yhat: np.ndarray) -> float:
    m = np.isfinite(y) & np.isfinite(yhat)
    d = np.abs(y[m]).sum()
    return float(np.abs(y[m] - yhat[m]).sum() / d) if d > 0 else np.nan


def mae(y, yhat) -> float:
    m = np.isfinite(y) & np.isfinite(yhat)
    return float(np.abs(y[m] - yhat[m]).mean())


def rmse(y, yhat) -> float:
    m = np.isfinite(y) & np.isfinite(yhat)
    return float(np.sqrt(((y[m] - yhat[m]) ** 2).mean()))


def bias(y, yhat) -> float:
    """Signed % bias: sum(forecast - actual) / sum(|actual|)."""
    m = np.isfinite(y) & np.isfinite(yhat)
    return float((yhat[m] - y[m]).sum() / np.abs(y[m]).sum())


def rmsse_scale(Y_train: np.ndarray) -> np.ndarray:
    """Per-series denominator: mean squared one-step naive error on TRAINING columns only."""
    d = np.diff(Y_train, axis=1)
    with np.errstate(invalid="ignore"):
        return np.nanmean(d ** 2, axis=1)


def rmsse(Y_train: np.ndarray, y: np.ndarray, yhat: np.ndarray, weights: np.ndarray | None = None) -> dict:
    """y, yhat: (n_series, H). Returns unweighted mean RMSSE and (optionally) revenue-weighted RMSSE."""
    sc = rmsse_scale(Y_train)
    with np.errstate(invalid="ignore", divide="ignore"):
        err = np.nanmean(np.where(np.isfinite(y) & np.isfinite(yhat), (y - yhat) ** 2, np.nan), axis=1)
        r = np.sqrt(err / sc)
    ok = np.isfinite(r) & (sc > 0)
    out = {"RMSSE": float(np.mean(r[ok])) if ok.any() else np.nan}
    if weights is not None and ok.any():
        w = weights[ok] / weights[ok].sum()
        out["RMSSE_w"] = float(np.sum(w * r[ok]))
    return out


def evaluate_matrix(Y_train, y, yhat, weights=None) -> dict:
    out = dict(WAPE=wape(y.ravel(), yhat.ravel()), MAE=mae(y.ravel(), yhat.ravel()),
               RMSE=rmse(y.ravel(), yhat.ravel()), Bias=bias(y.ravel(), yhat.ravel()))
    out.update(rmsse(Y_train, y, yhat, weights))
    out["n_obs"] = int(np.isfinite(y).sum())
    return out


def forecasts_to_matrix(df: pd.DataFrame, n: int, H: int, col: str = "y_hat") -> np.ndarray:
    """Long forecast frame (sid, h) -> (n, H) matrix; missing = NaN."""
    M = np.full((n, H), np.nan)
    M[df.sid.to_numpy(), df.h.to_numpy() - 1] = df[col].to_numpy()
    return M


# ---- probabilistic -------------------------------------------------------------------------------------
def pinball(y, q, tau: float) -> float:
    m = np.isfinite(y) & np.isfinite(q)
    d = y[m] - q[m]
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


def interval_metrics(y, lo, hi, coverage: float = 0.8) -> dict:
    m = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    y, lo, hi = y[m], lo[m], hi[m]
    a = 1 - coverage
    inside = (y >= lo) & (y <= hi)
    width = hi - lo
    winkler = width + (2 / a) * np.maximum(lo - y, 0) + (2 / a) * np.maximum(y - hi, 0)
    return dict(coverage=float(inside.mean()), nominal=coverage, mean_width=float(width.mean()),
                rel_width=float(width.sum() / np.abs(y).sum()), interval_score=float(winkler.mean()), n=int(m.sum()))

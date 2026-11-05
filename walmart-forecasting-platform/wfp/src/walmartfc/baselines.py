"""Naive and seasonal-naive baselines (past-only)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .panel import Panel


def _last_observed(Y: np.ndarray, t0: int, max_age: int = 52) -> np.ndarray:
    """Most recent observed value at or before t0 (within max_age weeks), else NaN."""
    seg = Y[:, max(0, t0 - max_age + 1): t0 + 1]
    out = np.full(Y.shape[0], np.nan)
    for j in range(seg.shape[1] - 1, -1, -1):
        v = seg[:, j]
        take = np.isnan(out) & ~np.isnan(v)
        out[take] = v[take]
    return out


def baseline_forecast(panel: Panel, origins: list[int], H: int, kind: str) -> pd.DataFrame:
    """kind='naive': last observation carried forward. kind='snaive': value 52 weeks before the target week;
    if that week is unobserved the naive value is used (reported via ``fallback``)."""
    Y, T = panel.Y, panel.T
    rows = []
    for t0 in origins:
        last = _last_observed(Y, t0)
        for h in range(1, H + 1):
            tt = t0 + h
            if tt >= T:
                continue
            if kind == "naive":
                yh, fb = last, np.zeros(panel.n, bool)
            elif kind == "snaive":
                j = tt - 52
                base = Y[:, j] if j >= 0 else np.full(panel.n, np.nan)
                fb = np.isnan(base)
                yh = np.where(fb, last, base)
            else:
                raise ValueError(kind)
            rows.append(pd.DataFrame(dict(sid=np.arange(panel.n), origin=t0, h=h, target_idx=tt, y_hat=yh, fallback=fb)))
    return pd.concat(rows, ignore_index=True)

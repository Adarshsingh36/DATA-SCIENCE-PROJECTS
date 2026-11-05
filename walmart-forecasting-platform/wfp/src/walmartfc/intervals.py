"""Split-conformal prediction intervals calibrated on chronologically earlier validation data."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def hbucket(h: np.ndarray) -> np.ndarray:
    return np.minimum((np.asarray(h) - 1) // 4, 2)


def conformal_q(scores: np.ndarray, alpha: float) -> float:
    s = np.sort(scores[np.isfinite(scores)])
    n = len(s)
    if n == 0:
        return np.nan
    k = math.ceil((n + 1) * (1 - alpha))
    return float(s[min(k, n) - 1])


def cqr_calibrate(val: pd.DataFrame, alpha: float, group_col: str | None = None) -> dict:
    """Conformalised quantile regression: score = max(lo - y, y - hi) / scale, per horizon bucket (and group)."""
    sc = np.maximum(val.q_lo - val.y, val.y - val.q_hi) / val.scale
    out = {}
    keys = hbucket(val.h.to_numpy())
    grp = val[group_col].to_numpy() if group_col else np.zeros(len(val), int)
    for g in np.unique(grp):
        for b in range(3):
            m = (keys == b) & (grp == g) & np.isfinite(sc.to_numpy())
            out[(g, b)] = conformal_q(sc.to_numpy()[m], alpha)
    out["_all"] = conformal_q(sc.to_numpy(), alpha)
    return out


def apply_cqr(df: pd.DataFrame, calib: dict, group_col: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    b = hbucket(df.h.to_numpy())
    grp = df[group_col].to_numpy() if group_col else np.zeros(len(df), int)
    q = np.array([calib.get((g, bb), calib["_all"]) if np.isfinite(calib.get((g, bb), np.nan)) else calib["_all"]
                  for g, bb in zip(grp, b)])
    return (df.q_lo.to_numpy() - q * df.scale.to_numpy(), df.q_hi.to_numpy() + q * df.scale.to_numpy())


def residual_conformal_calibrate(val: pd.DataFrame, alpha: float, group_col: str) -> dict:
    """Symmetric intervals y_hat +- q*scale; q from |y - y_hat|/scale per (group, horizon bucket)."""
    sc = np.abs(val.y - val.y_hat) / val.scale
    keys = hbucket(val.h.to_numpy())
    out = {}
    for g in val[group_col].unique():
        for b in range(3):
            m = ((val[group_col] == g).to_numpy()) & (keys == b) & np.isfinite(sc.to_numpy())
            out[(g, b)] = conformal_q(sc.to_numpy()[m], alpha)
    out["_all"] = conformal_q(sc.to_numpy(), alpha)
    return out


def apply_residual_conformal(df: pd.DataFrame, calib: dict, group_col: str) -> tuple[np.ndarray, np.ndarray]:
    b = hbucket(df.h.to_numpy())
    q = np.array([calib.get((g, bb), np.nan) for g, bb in zip(df[group_col], b)])
    q = np.where(np.isfinite(q), q, calib["_all"])
    return df.y_hat.to_numpy() - q * df.scale.to_numpy(), df.y_hat.to_numpy() + q * df.scale.to_numpy()

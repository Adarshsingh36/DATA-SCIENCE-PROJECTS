"""Leakage-safe feature construction for direct multi-horizon weekly forecasting.

Forecast origin ``t0`` = index of the last observed week. A sample is (series i, origin t0, horizon h) with
target week ``tt = t0 + h``. Every feature uses only columns ``<= t0`` of the sales matrix, except:
  * calendar attributes of the target week (holidays, week-of-year, ...) -- known in advance;
  * ``lag_52`` and its neighbours, taken at ``tt - 52`` (always ``<= t0`` because h <= 51);
  * optionally the target week's markdown total, if ``known_future_markdowns`` is enabled (planned promotions).
Rolling statistics are trailing windows; nothing is computed over the full dataset.
"""
from __future__ import annotations

from dataclasses import dataclass

import warnings

import numpy as np
import pandas as pd

from .panel import Panel

warnings.filterwarnings("ignore", category=RuntimeWarning)  # all-NaN slices are expected and handled

LEVEL_CODE = {"store_dept": 0, "store": 1, "dept": 2, "group": 3, "total": 4}


@dataclass
class BaseStats:
    """Per-(series, week) trailing statistics; entry [:, t] only uses weeks <= t."""
    mean: dict[int, np.ndarray]
    std: dict[int, np.ndarray]
    absmean13: np.ndarray
    absmean52: np.ndarray
    nobs13: np.ndarray
    nobs_hist: np.ndarray
    last_obs_age: np.ndarray
    store_mom: np.ndarray
    dept_mom: np.ndarray
    store_yoy: np.ndarray
    dept_yoy: np.ndarray


def _roll(Y: np.ndarray, w: int, fn: str, minp: int) -> np.ndarray:
    df = pd.DataFrame(Y.T)
    r = df.rolling(w, min_periods=minp)
    return (r.mean() if fn == "mean" else r.std()).to_numpy().T


def compute_base_stats(panel: Panel, windows: list[int], vol_windows: list[int]) -> BaseStats:
    Y = panel.Y
    ws = sorted(set(windows) | set(vol_windows) | {13, 52})
    mean = {w: _roll(Y, w, "mean", max(2, w // 4)) for w in ws}
    std = {w: _roll(Y, w, "std", max(3, w // 2)) for w in sorted(set(vol_windows))}
    A = np.abs(Y)
    absmean13 = _roll(A, 13, "mean", 2)
    absmean52 = _roll(A, 52, "mean", 4)
    obs = (~np.isnan(Y)).astype(float)
    nobs13 = pd.DataFrame(obs.T).rolling(13, min_periods=1).sum().to_numpy().T
    nobs_hist = np.cumsum(obs, axis=1)
    T = Y.shape[1]
    idx = np.where(obs > 0, np.arange(T)[None, :], -1)
    last_idx = np.maximum.accumulate(idx, axis=1)
    age = np.where(last_idx >= 0, np.arange(T)[None, :] - last_idx, np.nan)

    # cross-sectional context: store totals and department totals (sums of Y with NaN->0), past-only windows
    def group_mom_yoy(keys: np.ndarray):
        out_mom = np.full(Y.shape, np.nan); out_yoy = np.full(Y.shape, np.nan)
        Z = np.nan_to_num(Y, nan=0.0)
        for k in np.unique(keys):
            m = keys == k
            tot = Z[m].sum(axis=0, keepdims=True)
            m4, m13 = _roll(tot, 4, "mean", 2), _roll(tot, 13, "mean", 4)
            with np.errstate(divide="ignore", invalid="ignore"):
                mom = m4 / m13
                yoy = np.full_like(m13, np.nan)
                yoy[:, 52:] = m13[:, 52:] / m13[:, :-52]
            out_mom[m] = mom; out_yoy[m] = yoy
        return out_mom, out_yoy

    s = panel.series
    if (s.level == "store_dept").all():
        smom, syoy = group_mom_yoy(s.Store.to_numpy())
        dmom, dyoy = group_mom_yoy(s.Dept.to_numpy())
    else:  # aggregate panel: context is the node itself
        smom = dmom = mean[4] / mean[13]
        syoy = dyoy = np.full_like(mean[13], np.nan)
        syoy[:, 52:] = mean[13][:, 52:] / mean[13][:, :-52]
    return BaseStats(mean, std, absmean13, absmean52, nobs13, nobs_hist, age, smom, dmom, syoy, dyoy)


def series_scale(base: BaseStats, t0: int, floor: float) -> np.ndarray:
    """Past-only scale (mean |sales| over the trailing 13 observed weeks, else 52). NaN if the series is dormant."""
    s = base.absmean13[:, t0].copy()
    fallback = base.absmean52[:, t0]
    s = np.where(np.isnan(s), fallback, s)
    return np.where(np.isnan(s), np.nan, np.maximum(s, floor))


def make_samples(panel: Panel, base: BaseStats, origins: list[int], horizons: list[int], cfg: dict,
                 with_target: bool = True) -> pd.DataFrame:
    fc = cfg["forecast"]
    lags = [k for k in fc["lags"] if k != 52]
    Y, T = panel.Y, panel.T
    assert max(horizons) <= 51, "lag-52 alignment requires h <= 51"
    frames = []
    cal = panel.cal
    ser = panel.series
    level_code = ser.level.map(LEVEL_CODE).to_numpy()
    static = dict(
        store=ser.Store.to_numpy(), dept=ser.Dept.to_numpy(), type_code=ser.type_code.to_numpy(),
        level_code=level_code, log_size=np.log(ser.Size.to_numpy().astype(float)),
    )
    n = panel.n
    for t0 in origins:
        s = series_scale(base, t0, fc["scale_floor"])
        elig = (~np.isnan(s)) & (base.nobs_hist[:, t0] >= fc["min_history_weeks"])
        with np.errstate(invalid="ignore", divide="ignore"):
            common = {}
            for k in lags:
                common[f"lag_{k}"] = Y[:, t0 - k + 1] / s if t0 - k + 1 >= 0 else np.full(n, np.nan)
            for w in fc["rolling_windows"]:
                common[f"mean_{w}"] = base.mean[w][:, t0] / s
            for w in fc["vol_windows"]:
                common[f"std_{w}"] = base.std[w][:, t0] / s
            common["cv_13"] = base.std[13][:, t0] / np.maximum(np.abs(base.mean[13][:, t0]), 1e-6 * s)
            common["mom_4_13"] = base.mean[4][:, t0] / base.mean[13][:, t0]
            m13_old = base.mean[13][:, t0 - 52] if t0 - 52 >= 0 else np.full(n, np.nan)
            common["yoy_13"] = base.mean[13][:, t0] / m13_old
            common["log_scale"] = np.log(s)
            common["nobs_13"] = base.nobs13[:, t0]
            common["hist_len"] = base.nobs_hist[:, t0]
            common["weeks_since_obs"] = base.last_obs_age[:, t0]
            common["store_mom"] = base.store_mom[:, t0]
            common["dept_mom"] = base.dept_mom[:, t0]
            common["store_yoy"] = base.store_yoy[:, t0]
            common["dept_yoy"] = base.dept_yoy[:, t0]
            for e in ("Temperature", "Fuel_Price", "CPI", "Unemployment"):
                common[f"{e.lower()}_t0"] = panel.exog[e][:, t0]
            md = panel.exog["MarkDown_total"]
            common["markdown_4wk_t0"] = np.nanmean(md[:, max(0, t0 - 3): t0 + 1], axis=1)
            common["markdown_avail_t0"] = panel.exog["markdown_available"][:, t0]
        for h in horizons:
            tt = t0 + h
            if tt >= T and with_target:
                continue
            d = {"sid": np.arange(n), "origin": np.full(n, t0), "h": np.full(n, h), "target_idx": np.full(n, tt),
                 "scale": s, "eligible": elig}
            d.update(static); d.update(common)
            with np.errstate(invalid="ignore", divide="ignore"):
                for off, name in ((0, "lag_52"), (-1, "lag_53"), (1, "lag_51")):
                    j = tt - 52 + off
                    d[name] = Y[:, j] / s if 0 <= j <= t0 else np.full(n, np.nan)
            j52 = [j for j in (tt - 53, tt - 52, tt - 51) if 0 <= j <= t0]
            d["lag52_avg3"] = (np.nanmean(Y[:, j52], axis=1) / s) if j52 else np.full(n, np.nan)
            row = cal.iloc[tt] if tt < T else None
            if row is None:  # beyond the grid (true future): calendar from dates
                raise ValueError("target beyond grid not supported in evaluation mode")
            for c in ("woy", "month", "is_holiday", "hol_type", "wk_to_xmas", "wk_to_thx", "easter_dist", "tgt_dom"):
                d[c] = np.full(n, row[c], dtype=float)
            if fc.get("known_future_markdowns"):
                d["markdown_tgt"] = md[:, tt]
            if with_target:
                d["y"] = Y[:, tt]
            frames.append(pd.DataFrame(d))
    df = pd.concat(frames, ignore_index=True)
    keep64 = {"y", "scale"}
    for c in df.columns:
        if df[c].dtype == np.float64 and c not in keep64:
            df[c] = df[c].astype(np.float32)
    if with_target:
        df["y_scaled"] = df["y"] / df["scale"]
    return df


FEATURE_EXCLUDE = {"sid", "origin", "target_idx", "scale", "eligible", "y", "y_scaled"}
CATEGORICAL = ["store", "dept", "type_code", "level_code", "hol_type", "month"]


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in FEATURE_EXCLUDE]

"""Exploratory analysis. Descriptive only: statistics computed here are never used as model features."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ROOT


def run_eda(cfg, prep) -> dict:
    bp, hier = prep["panel_bottom"], prep["hier"]
    figs = ROOT / "reports/figures"
    figs.mkdir(parents=True, exist_ok=True)
    Y, dates, ser, cal = bp.Y, bp.dates, bp.series, bp.cal
    out: dict = {}
    total = np.nansum(Y, axis=0)
    hol = cal.is_holiday.to_numpy().astype(bool)

    # 1. total weekly sales with holiday weeks
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(dates, total / 1e6, color="#1f4e79")
    ax.scatter(dates[hol], total[hol] / 1e6, color="#c0392b", zorder=3, label="Kaggle holiday week")
    ax.set_ylabel("Total weekly sales ($M)"); ax.set_title("Total weekly sales, all stores & departments"); ax.legend()
    fig.tight_layout(); fig.savefig(figs / "01_total_weekly_sales.png", dpi=130); plt.close(fig)

    # 2. year-over-year overlay by week-of-year
    fig, ax = plt.subplots(figsize=(11, 4))
    for y in (2010, 2011, 2012):
        m = dates.year == y
        ax.plot(cal.woy.to_numpy()[m], total[m] / 1e6, label=str(y))
    ax.set_xlabel("ISO week"); ax.set_ylabel("$M"); ax.set_title("Seasonality: same weeks across years"); ax.legend()
    fig.tight_layout(); fig.savefig(figs / "02_yoy_seasonality.png", dpi=130); plt.close(fig)

    # 3. holiday uplift measured against the trailing 4-week mean (descriptive)
    tot = pd.Series(total, index=dates)
    ratio = tot / tot.shift(1).rolling(4).mean()
    up = {}
    for t, nm in ((1, "Super Bowl"), (2, "Labor Day"), (3, "Thanksgiving"), (4, "Christmas wk")):
        m = (cal.hol_type.to_numpy() == t)
        up[nm] = float(ratio[m].mean() - 1) if m.any() else None
    pre = (cal.wk_to_xmas.to_numpy() >= 1) & (cal.wk_to_xmas.to_numpy() < 2) & (cal.month.to_numpy() == 12)
    up["Week before Christmas"] = float(ratio[pre].mean() - 1)
    out["holiday_uplift_vs_prev4wk_mean"] = up

    # 4. concentration: store, dept, type
    rev = np.nansum(Y, axis=1)
    by_dept = pd.Series(rev, index=ser.Dept).groupby(level=0).sum().sort_values(ascending=False)
    by_store = pd.Series(rev, index=ser.Store).groupby(level=0).sum().sort_values(ascending=False)
    out["top5_dept_share"] = float(by_dept.head(5).sum() / by_dept.sum())
    out["top10_store_share"] = float(by_store.head(10).sum() / by_store.sum())
    out["type_share"] = {k: float(v) for k, v in (pd.Series(rev, index=ser.Type).groupby(level=0).sum() / rev.sum()).items()}
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    (by_dept / 1e6).head(20).plot.bar(ax=axs[0], color="#1f4e79"); axs[0].set_title("Top-20 departments, total sales ($M)")
    (by_store / 1e6).plot.bar(ax=axs[1], color="#2e86c1"); axs[1].set_title("Store total sales ($M)"); axs[1].tick_params(labelsize=6)
    fig.tight_layout(); fig.savefig(figs / "03_concentration.png", dpi=130); plt.close(fig)

    # 5. spike investigation: are extreme weeks holiday-driven?
    med = np.nanmedian(Y, axis=1, keepdims=True)
    mad = np.nanmedian(np.abs(Y - med), axis=1, keepdims=True) * 1.4826
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (Y - med) / np.where(mad > 0, mad, np.nan)
    spike = z > 5
    tt = np.arange(Y.shape[1])
    near_hol = (cal.is_holiday.to_numpy().astype(bool)) | (cal.wk_to_xmas.to_numpy() <= 2) | (
        (cal.wk_to_thx.to_numpy() <= 1) & (cal.wk_to_thx.to_numpy() >= -1))
    n_sp = int(spike.sum())
    share_events = float(spike[:, near_hol].sum() / max(n_sp, 1))
    base_share = float(near_hol.mean())
    out["spikes"] = dict(definition="robust z > 5 vs series median/MAD", n=n_sp, share_in_holiday_or_xmas_weeks=share_events,
                         share_of_all_weeks_that_are_such=base_share,
                         lift=share_events / base_share if base_share else None)
    out["negative_sales"] = dict(n=int((Y < 0).sum()), share=float((Y < 0).sum() / np.isfinite(Y).sum()),
                                 total_value=float(Y[Y < 0].sum()))
    out["missing_share_of_grid"] = float(np.isnan(Y).mean())
    hist_len = np.isfinite(Y).sum(axis=1)
    out["series_with_full_history"] = int((hist_len == Y.shape[1]).sum())
    out["series_total"] = int(Y.shape[0])
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(dates, np.nansum(np.where(spike, 1, 0), axis=0), color="#c0392b")
    for t in dates[near_hol]:
        ax.axvline(t, color="grey", alpha=0.15)
    ax.set_title("Number of spike observations per week (grey = holiday / Thanksgiving / pre-Christmas weeks)")
    fig.tight_layout(); fig.savefig(figs / "04_spikes_vs_holidays.png", dpi=130); plt.close(fig)

    # 6. exogenous variables: pooled correlation with de-seasonalised store-level sales (descriptive only)
    store_tot = pd.DataFrame(np.vstack([np.nansum(Y[(ser.Store == s).to_numpy()], axis=0) for s in sorted(ser.Store.unique())]).T,
                             index=dates)
    ex = {}
    for c in ("Temperature", "Fuel_Price", "CPI", "Unemployment", "MarkDown_total"):
        E = pd.DataFrame(np.vstack([bp.exog[c][(ser.Store == s).to_numpy()][0] for s in sorted(ser.Store.unique())]).T, index=dates)
        a = store_tot.sub(store_tot.mean()).div(store_tot.std())
        b = E.sub(E.mean()).div(E.std().replace(0, np.nan))
        ex[c] = float(np.nanmean((a * b).to_numpy()))
    out["pooled_corr_store_total_vs_exog_(raw, confounded by season)"] = ex

    # 7. sample series
    fig, axs = plt.subplots(2, 2, figsize=(11, 6))
    top = np.argsort(-rev)[[0, 5, 50, 500]]
    for ax, i in zip(axs.ravel(), top):
        ax.plot(dates, Y[i]); ax.set_title(f"Store {ser.Store[i]} / Dept {ser.Dept[i]}", fontsize=9)
    fig.tight_layout(); fig.savefig(figs / "05_sample_series.png", dpi=130); plt.close(fig)

    Path(ROOT / "reports/results").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "reports/results/eda_summary.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    return out

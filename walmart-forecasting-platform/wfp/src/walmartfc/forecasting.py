"""Core forecasting run for one chronological fold: baselines, LightGBM, hierarchical reconciliation, intervals."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import metrics as M
from .baselines import baseline_forecast
from .features import compute_base_stats, make_samples, series_scale
from .intervals import apply_cqr, apply_residual_conformal, cqr_calibrate, residual_conformal_calibrate
from .models.lgbm import GlobalLGBM
from .panel import Hierarchy, Panel
from .reconcile import coherence_error, projection_matrix, reconcile, shrunk_covariance
from .splits import Fold

LEVELS = ["total", "group", "store", "dept", "store_dept"]


def subpanel(p: Panel, sl: slice) -> Panel:
    return Panel(p.dates, p.series.iloc[sl].reset_index(drop=True), p.Y[sl], {k: v[sl] for k, v in p.exog.items()}, p.cal)


@dataclass
class FoldResult:
    fold: Fold
    test: pd.DataFrame          # long: model, node, h, y, y_hat, lo, hi, ...
    metrics: pd.DataFrame
    intervals: pd.DataFrame
    importance: pd.DataFrame
    info: dict


def _frames(panel: Panel, base, fold: Fold, cfg, stage: str) -> pd.DataFrame:
    H = fold.H
    stride = cfg["forecast"]["origin_stride"]
    if stage == "train1":
        cut = fold.val_start
        origins = fold.train_origins(stride, cut)
    elif stage == "train2":
        cut = fold.test_start
        origins = fold.train_origins(stride, cut)
    elif stage == "val":
        df = make_samples(panel, base, fold.val_origins(4), list(range(1, H + 1)), cfg)
        return df[(df.target_idx >= fold.val_start) & (df.target_idx < fold.test_start)].reset_index(drop=True)
    elif stage == "test":
        return make_samples(panel, base, [fold.test_origin], list(range(1, H + 1)), cfg)
    df = make_samples(panel, base, origins, list(range(1, H + 1)), cfg)
    df = df[(df.target_idx < cut) & df.y.notna() & df.eligible]
    return df.reset_index(drop=True)


def _fit_predict_panel(panel: Panel, fold: Fold, cfg, tag: str, log=print):
    """Stage 1 (train < val_start, early stop on val) then refit on all data < test_start with fixed iterations."""
    fc = cfg["forecast"]
    base = compute_base_stats(panel, fc["rolling_windows"], fc["vol_windows"])
    t = time.time()
    tr1, va = _frames(panel, base, fold, cfg, "train1"), _frames(panel, base, fold, cfg, "val")
    log(f"[{fold.name}/{tag}] stage-1 rows train={len(tr1):,} val={len(va):,} ({time.time()-t:.0f}s)")
    m1 = GlobalLGBM(cfg).fit(tr1, va)
    val_pred = m1.predict(va)
    del tr1
    tr2 = _frames(panel, base, fold, cfg, "train2")
    m2 = GlobalLGBM(cfg).fit(tr2, fixed_iters=m1.best_iter)
    te = _frames(panel, base, fold, cfg, "test")
    test_pred = m2.predict(te)
    log(f"[{fold.name}/{tag}] best_iter={m1.best_iter} refit rows={len(tr2):,} done in {time.time()-t:.0f}s")
    return val_pred, test_pred, m2.importance(), m1.best_iter, base


def _node_matrix(df: pd.DataFrame, n_nodes: int, col: str, offset_map=None) -> pd.DataFrame:
    pass


def run_fold(bottom: Panel, full: Panel, hier: Hierarchy, fold: Fold, cfg, log=print) -> FoldResult:
    fc = cfg["forecast"]
    H = fold.H
    alpha = 1 - fc["interval_coverage"]
    n_all, m = hier.S.shape
    off = n_all - m
    agg = subpanel(full, slice(0, off))

    # ---- LightGBM on bottom and on aggregate nodes --------------------------------------------------------
    vb, tb, imp_b, it_b, base_b = _fit_predict_panel(bottom, fold, cfg, "bottom", log)
    va_, ta, imp_a, it_a, base_a = _fit_predict_panel(agg, fold, cfg, "aggregates", log)
    for d, o in ((vb, off), (tb, off), (va_, 0), (ta, 0)):
        d["node"] = d.sid + o
    val_base = pd.concat([va_, vb], ignore_index=True)
    test_base = pd.concat([ta, tb], ignore_index=True)
    test_base["level"] = hier.rows.level.to_numpy()[test_base.node]
    val_base["level"] = hier.rows.level.to_numpy()[val_base.node]
    val_base["level_code"] = val_base.level.map({"store_dept": 0, "store": 1, "dept": 2, "group": 3, "total": 4})
    test_base["level_code"] = test_base.level.map({"store_dept": 0, "store": 1, "dept": 2, "group": 3, "total": 4})

    Yf = full.Y
    tgt = np.arange(fold.test_start, fold.test_start + H)
    Y_test = Yf[:, tgt]                      # (n_all, H)
    Y_hist = Yf[:, :fold.test_start]         # training columns only -> RMSSE scale, weights

    def mat(df, col="y_hat"):
        out = np.full((n_all, H), np.nan)
        out[df.node.to_numpy(), df.h.to_numpy() - 1] = df[col].to_numpy()
        return out

    rows = []                                 # long test forecasts for every model
    def add_model(name, F, lo=None, hi=None, lo2=None, hi2=None):
        d = pd.DataFrame(dict(model=name, node=np.repeat(np.arange(n_all), H), h=np.tile(np.arange(1, H + 1), n_all),
                              y=Y_test.ravel(), y_hat=F.ravel()))
        for k, v in (("lo", lo), ("hi", hi), ("lo_raw", lo2), ("hi_raw", hi2)):
            d[k] = np.nan if v is None else v.ravel()
        rows.append(d)

    # ---- baselines on every node (aggregates computed on aggregated series) -------------------------------
    for kind in ("naive", "snaive"):
        bf = baseline_forecast(full, [fold.test_origin], H, kind)
        F = np.full((n_all, H), np.nan)
        F[bf.sid.to_numpy(), bf.h.to_numpy() - 1] = bf.y_hat.to_numpy()
        add_model(kind, np.nan_to_num(F, nan=0.0))

    # ---- LightGBM base forecasts + CQR intervals ------------------------------------------------------------
    cal = cqr_calibrate(val_base.assign(y=val_base.y), alpha, "level_code")
    lo_c, hi_c = apply_cqr(test_base, cal, "level_code")
    test_base["lo_cqr"], test_base["hi_cqr"] = lo_c, hi_c
    scale_mat = mat(test_base, "scale")
    F_base = mat(test_base)
    F_base0 = np.nan_to_num(F_base, nan=0.0)
    add_model("lgbm", F_base0, mat(test_base, "lo_cqr"), mat(test_base, "hi_cqr"), mat(test_base, "q_lo"), mat(test_base, "q_hi"))

    # ---- reconciliation ---------------------------------------------------------------------------------------
    # Two covariance estimates for MinT (both use only data before the test window):
    #   mint_shrink          : shrunk covariance of hold-out (validation-window) errors of the base forecasts
    #   mint_shrink_insample : shrunk covariance of in-sample one-step naive residuals (training columns)
    keys = val_base[["origin", "h"]].drop_duplicates().reset_index(drop=True)
    kidx = {(o, h): i for i, (o, h) in enumerate(zip(keys.origin, keys.h))}
    K = len(keys)
    Vf = np.full((n_all, K), np.nan); Vs = np.full((n_all, K), np.nan); Vh = np.zeros(K, int)
    ci = np.array([kidx[(o, h)] for o, h in zip(val_base.origin, val_base.h)])
    Vf[val_base.node.to_numpy(), ci] = val_base.y_hat.to_numpy()
    Vs[val_base.node.to_numpy(), ci] = val_base.scale.to_numpy()
    Vh[ci] = val_base.h.to_numpy()
    Vy = np.stack([Yf[:, o + h] for o, h in zip(keys.origin, keys.h)], axis=1)
    lvl = hier.rows.level.to_numpy()
    W_val, lam_val = shrunk_covariance((Vy - Vf).T)
    W_ins, lam_ins = shrunk_covariance(np.diff(Yf[:, :fold.val_start], axis=1).T)
    log(f"[{fold.name}] MinT shrinkage lambda: hold-out errors={lam_val:.3f} (n={K}), in-sample naive residuals={lam_ins:.3f}")
    Wmap = {"mint_shrink": W_val, "mint_shrink_insample": W_ins}

    coh = {"base_unreconciled": coherence_error(hier, F_base0)}
    val_scores = {}
    ga = (keys.origin == keys.origin.min()).to_numpy()          # cross-fitting groups over validation samples
    for method in cfg["hierarchy"]["reconciliation"]:
        G = projection_matrix(hier, method.replace("_insample", ""), Wmap.get(method))
        F = reconcile(hier, F_base, G)
        coh[method] = coherence_error(hier, F)
        if method == "mint_shrink":
            # W is estimated from validation errors, so scoring/calibrating on the same errors would be optimistic.
            # Cross-fit: G estimated on one group of validation origins is applied to the other group (and vice versa).
            Fv = np.zeros_like(Vf)
            for fit_m, app_m in ((ga, ~ga), (~ga, ga)):
                Wc, _ = shrunk_covariance((Vy[:, fit_m] - Vf[:, fit_m]).T)
                Gc = projection_matrix(hier, "mint_shrink", Wc)
                Fv[:, app_m] = reconcile(hier, Vf[:, app_m], Gc)
                del Gc
        else:
            Fv = reconcile(hier, Vf, G)
        vd = pd.DataFrame(dict(node=np.repeat(np.arange(n_all), K), h=np.tile(Vh, n_all), y=Vy.ravel(), y_hat=Fv.ravel(),
                               scale=Vs.ravel()))
        vd["level"] = lvl[vd.node]
        vd = vd[np.isfinite(vd.y) & np.isfinite(vd.scale)]
        # validation WAPE (mean over the 5 hierarchy levels) -> used to pick the default method WITHOUT looking at test data
        val_scores[method] = float(np.mean([M.wape(g.y.to_numpy(), g.y_hat.to_numpy()) for _, g in vd.groupby("level")]))
        calr = residual_conformal_calibrate(vd, alpha, "level")
        td = pd.DataFrame(dict(node=np.repeat(np.arange(n_all), H), h=np.tile(np.arange(1, H + 1), n_all),
                               y_hat=F.ravel(), scale=np.where(np.isfinite(scale_mat), scale_mat, fc["scale_floor"]).ravel()))
        td["level"] = lvl[td.node]
        lo, hi = apply_residual_conformal(td, calr, "level")
        add_model(f"lgbm_{method}", F, lo.reshape(n_all, H), hi.reshape(n_all, H))
        del G
    selected = min(val_scores, key=val_scores.get)
    log(f"[{fold.name}] validation-selected reconciliation: {selected} {({k: round(v, 4) for k, v in val_scores.items()})}")
    lam = lam_val

    test = pd.concat(rows, ignore_index=True)
    test["level"] = lvl[test.node]
    test["target_idx"] = fold.test_start + test.h - 1
    test["fold"] = fold.name

    # ---- metrics by model x level ---------------------------------------------------------------------------------
    met = []
    w_all = np.nan_to_num(np.nanmean(np.abs(Y_hist[:, -13:]), axis=1), nan=0.0)
    for model, d in test.groupby("model"):
        F = d.sort_values(["node", "h"]).y_hat.to_numpy().reshape(n_all, H)
        for L in LEVELS:
            sl = hier.level_slices[L]
            r = M.evaluate_matrix(Y_hist[sl], Y_test[sl], F[sl], w_all[sl])
            met.append(dict(fold=fold.name, model=model, level=L, **r))
        for h in range(H):  # accuracy by horizon on the bottom level
            sl = hier.level_slices["store_dept"]
            met.append(dict(fold=fold.name, model=model, level=f"store_dept@h{h+1}",
                            WAPE=M.wape(Y_test[sl, h], F[sl, h]), MAE=M.mae(Y_test[sl, h], F[sl, h]),
                            RMSE=M.rmse(Y_test[sl, h], F[sl, h]), n_obs=int(np.isfinite(Y_test[sl, h]).sum())))
    metrics = pd.DataFrame(met)

    # ---- interval metrics -----------------------------------------------------------------------------------------
    im = []
    for model, d in test.groupby("model"):
        for (a, b, kind) in (("lo", "hi", "calibrated"), ("lo_raw", "hi_raw", "raw_quantile")):
            if d[a].notna().any():
                for L in LEVELS:
                    dd = d[d.level == L]
                    r = M.interval_metrics(dd.y.to_numpy(), dd[a].to_numpy(), dd[b].to_numpy(), fc["interval_coverage"])
                    im.append(dict(fold=fold.name, model=model, level=L, interval=kind, **r))
    intervals = pd.DataFrame(im)

    info = dict(coherence=coh, mint_lambda=lam_val, mint_lambda_insample=lam_ins, val_reconciliation_wape=val_scores,
                selected_reconciliation=f"lgbm_{selected}", best_iter_bottom=it_b, best_iter_agg=it_a,
                ineligible_bottom_test_obs=int(((~tb.eligible.to_numpy()) & np.isfinite(tb.y.to_numpy())).sum()),
                val_origins=[int(x) for x in fold.val_origins(4)])
    imp = imp_b.assign(panel="bottom", fold=fold.name)
    return FoldResult(fold, test, metrics, intervals, imp, info)

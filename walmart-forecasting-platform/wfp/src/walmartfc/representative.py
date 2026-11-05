"""Prophet and Temporal Fusion Transformer on a representative subset of store-department series.

Why a subset: Prophet needs one model per series and TFT is trained on CPU in this project's reference environment
(1 core, 3 GB RAM). The subset is stratified by *past-only* revenue decile and restricted to series with a complete,
gap-free history (TFT requires contiguous time indices). All other models are re-scored on the SAME subset so the
comparison is like-for-like.
"""
from __future__ import annotations

import logging
import tempfile
import time
import warnings

import numpy as np
import pandas as pd

from . import metrics as M
from .config import ROOT
from .pipeline import artifacts_dir, get_folds, selected_model

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


def select_representative(panel, n_series: int, sel_end: int, seed: int = 42) -> np.ndarray:
    """Series with complete history, stratified by revenue decile measured on columns < sel_end only."""
    Y = panel.Y
    complete = np.isfinite(Y).all(axis=1)
    rev = np.nansum(Y[:, :sel_end], axis=1)
    idx = np.where(complete & (rev > 0))[0]
    dec = pd.qcut(pd.Series(rev[idx]).rank(method="first"), 10, labels=False).to_numpy()
    rng = np.random.default_rng(seed)
    per = max(1, n_series // 10)
    chosen = []
    for d in range(10):
        pool = idx[dec == d]
        chosen.extend(rng.choice(pool, size=min(per, len(pool)), replace=False).tolist())
    return np.array(sorted(chosen))


def _holidays_df(panel) -> pd.DataFrame:
    cal = panel.cal
    names = {1: "superbowl", 2: "laborday", 3: "thanksgiving", 4: "christmas"}
    rows = [dict(holiday=names[int(t)], ds=d, lower_window=0, upper_window=0)
            for d, t in zip(cal.index, cal.hol_type) if t > 0]
    return pd.DataFrame(rows)


def prophet_forecast(panel, sids, fold, cfg) -> pd.DataFrame:
    from prophet import Prophet
    hol = _holidays_df(panel)
    dates = panel.dates
    fut = pd.DataFrame(dict(ds=dates[fold.test_start: fold.test_start + fold.H]))
    out = []
    t = time.time()
    for k, sid in enumerate(sids):
        y = panel.Y[sid, : fold.test_start]
        d = pd.DataFrame(dict(ds=dates[: fold.test_start], y=y)).dropna()
        m = Prophet(yearly_seasonality=6, weekly_seasonality=False, daily_seasonality=False, holidays=hol,
                    seasonality_mode="multiplicative", interval_width=cfg["forecast"]["interval_coverage"],
                    uncertainty_samples=200, changepoint_prior_scale=0.05)
        m.fit(d)
        f = m.predict(fut)
        out.append(pd.DataFrame(dict(sid=sid, h=np.arange(1, fold.H + 1), y_hat=f.yhat.to_numpy(),
                                     lo=f.yhat_lower.to_numpy(), hi=f.yhat_upper.to_numpy())))
    print(f"  prophet: {len(sids)} series in {time.time() - t:.0f}s", flush=True)
    return pd.concat(out, ignore_index=True)


def tft_forecast(panel, sids, fold, cfg) -> pd.DataFrame:
    import lightning.pytorch as pl
    import torch
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss

    tc = cfg["representative"]["tft"]
    torch.manual_seed(cfg["lightgbm"]["seed"]); np.random.seed(cfg["lightgbm"]["seed"])
    torch.set_num_threads(1)
    cal, ser = panel.cal, panel.series
    T = fold.test_start + fold.H
    rows = []
    for sid in sids:
        d = pd.DataFrame(dict(
            series_id=str(sid), time_idx=np.arange(T), y=panel.Y[sid, :T],
            store_id=str(ser.Store[sid]), dept_id=str(ser.Dept[sid]), store_type=str(ser.Type[sid]), size=float(ser.Size[sid]),
            hol_type=cal.hol_type.to_numpy()[:T].astype(str), is_holiday=cal.is_holiday.to_numpy()[:T].astype(float),
            woy=cal.woy.to_numpy()[:T].astype(float), wk_to_xmas=cal.wk_to_xmas.to_numpy()[:T],
            wk_to_thx=cal.wk_to_thx.to_numpy()[:T],
            temp=panel.exog["Temperature"][sid, :T], fuel=panel.exog["Fuel_Price"][sid, :T], cpi=panel.exog["CPI"][sid, :T],
            unemp=panel.exog["Unemployment"][sid, :T], markdown=panel.exog["MarkDown_total"][sid, :T]))
        rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    enc, H = tc["encoder_length"], fold.H
    train_df = df[df.time_idx < fold.val_start]
    training = TimeSeriesDataSet(
        train_df, time_idx="time_idx", target="y", group_ids=["series_id"], max_encoder_length=enc, max_prediction_length=H,
        static_categoricals=["series_id", "store_id", "dept_id", "store_type"], static_reals=["size"],
        time_varying_known_categoricals=["hol_type"],
        time_varying_known_reals=["time_idx", "is_holiday", "woy", "wk_to_xmas", "wk_to_thx"],
        time_varying_unknown_reals=["y", "temp", "fuel", "cpi", "unemp", "markdown"],
        target_normalizer=GroupNormalizer(groups=["series_id"], method="standard"),
        add_relative_time_idx=True, add_target_scales=True, add_encoder_length=False)
    val_ds = TimeSeriesDataSet.from_dataset(training, df[df.time_idx < fold.test_start], min_prediction_idx=fold.val_start,
                                            stop_randomization=True)
    tl = training.to_dataloader(train=True, batch_size=tc["batch_size"], num_workers=0)
    vl = val_ds.to_dataloader(train=False, batch_size=tc["batch_size"] * 2, num_workers=0)
    q = cfg["forecast"]["quantiles"]
    model = TemporalFusionTransformer.from_dataset(
        training, learning_rate=tc["learning_rate"], hidden_size=tc["hidden_size"], attention_head_size=tc["attention_heads"],
        dropout=tc["dropout"], hidden_continuous_size=max(4, tc["hidden_size"] // 2), loss=QuantileLoss(quantiles=q),
        log_interval=-1, reduce_on_plateau_patience=2)
    ck = tempfile.mkdtemp()
    ckpt = ModelCheckpoint(dirpath=ck, monitor="val_loss", save_top_k=1, mode="min")
    trainer = pl.Trainer(max_epochs=tc["max_epochs"], accelerator="cpu", devices=1, gradient_clip_val=0.1,
                         callbacks=[EarlyStopping(monitor="val_loss", patience=3, mode="min"), ckpt],
                         enable_progress_bar=False, logger=False, enable_model_summary=False)
    t = time.time()
    trainer.fit(model, tl, vl)
    best = TemporalFusionTransformer.load_from_checkpoint(ckpt.best_model_path)
    print(f"  TFT trained in {time.time() - t:.0f}s, best val_loss={float(ckpt.best_model_score):.4f}, "
          f"params={sum(p.numel() for p in best.parameters()):,}", flush=True)
    pred_ds = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
    pl_ = pred_ds.to_dataloader(train=False, batch_size=256, num_workers=0)
    res = best.predict(pl_, mode="quantiles", return_index=True)
    qs = res.output.numpy() if hasattr(res, "output") else res[0].numpy()
    idx = res.index if hasattr(res, "index") else res[1]
    out = []
    for r in range(qs.shape[0]):
        sid = int(idx.iloc[r]["series_id"])
        out.append(pd.DataFrame(dict(sid=sid, h=np.arange(1, H + 1), y_hat=qs[r, :, 1], lo=qs[r, :, 0], hi=qs[r, :, 2])))
    return pd.concat(out, ignore_index=True)


def run_representative(cfg, prep, only=None):
    bottom, hier = prep["panel_bottom"], prep["hier"]
    ad = artifacts_dir(cfg)
    folds = get_folds(cfg, bottom)
    rc = cfg["representative"]
    sel_end = min(f.val_start for f in folds.values())
    sids = select_representative(bottom, rc["n_series"], sel_end, cfg["lightgbm"]["seed"])
    print(f"representative series: {len(sids)} (revenue deciles from weeks < {sel_end})", flush=True)
    off = hier.S.shape[0] - hier.S.shape[1]
    all_metrics, all_int = [], []
    for name, fold in folds.items():
        if only and name not in only:
            continue
        print(f"=== representative fold {name}", flush=True)
        H = fold.H
        Y_test = bottom.Y[sids][:, fold.test_start: fold.test_start + H]
        Y_hist = bottom.Y[sids][:, : fold.test_start]
        w = np.nan_to_num(np.nanmean(np.abs(Y_hist[:, -13:]), axis=1))
        sidpos = {int(s): i for i, s in enumerate(sids)}
        fc = {}
        base = pd.read_parquet(ad / f"fold_{name}" / "test_forecasts.parquet")
        for model in dict.fromkeys(["naive", "snaive", "lgbm", "lgbm_bottom_up", selected_model(cfg, name)]):
            d = base[(base.model == model) & (base.node >= off) & (base.node - off).isin(sids)].copy()
            d["sid"] = d.node - off
            fc[model] = d[["sid", "h", "y_hat", "lo", "hi"]]
        # per-model cache so an interrupted run resumes instead of restarting
        for nm, enabled, fn in (("prophet", rc["prophet"], prophet_forecast), ("tft", rc["tft"]["enabled"], tft_forecast)):
            if not enabled:
                continue
            cp = ad / f"fold_{name}" / f"rep_cache_{nm}.parquet"
            if cp.exists():
                fc[nm] = pd.read_parquet(cp)
                print(f"  {nm}: loaded cached forecasts", flush=True)
            else:
                fc[nm] = fn(bottom, sids, fold, cfg)
                fc[nm].to_parquet(cp)
        stack = []
        for model, d in fc.items():
            F = np.full((len(sids), H), np.nan); LO = F.copy(); HI = F.copy()
            pos = d.sid.map(sidpos).to_numpy()
            F[pos, d.h.to_numpy() - 1] = d.y_hat.to_numpy()
            LO[pos, d.h.to_numpy() - 1] = d.lo.to_numpy(); HI[pos, d.h.to_numpy() - 1] = d.hi.to_numpy()
            r = M.evaluate_matrix(Y_hist, Y_test, F, w)
            all_metrics.append(dict(fold=name, model=model, n_series=len(sids), **r))
            if np.isfinite(LO).any():
                all_int.append(dict(fold=name, model=model, **M.interval_metrics(Y_test.ravel(), LO.ravel(), HI.ravel(),
                                                                               cfg["forecast"]["interval_coverage"])))
            dd = pd.DataFrame(dict(fold=name, model=model, sid=np.repeat(sids, H), h=np.tile(np.arange(1, H + 1), len(sids)),
                                   y=Y_test.ravel(), y_hat=F.ravel(), lo=LO.ravel(), hi=HI.ravel()))
            stack.append(dd)
        pd.concat(stack).to_parquet(ad / f"fold_{name}" / "representative_forecasts.parquet")
    res = ROOT / "reports/results"
    pd.DataFrame(all_metrics).to_csv(res / "representative_metrics.csv", index=False)
    pd.DataFrame(all_int).to_csv(res / "representative_interval_metrics.csv", index=False)
    pd.Series(sids).to_csv(res / "representative_series.csv", index=False, header=["sid"])
    print(pd.DataFrame(all_metrics).round(4).to_string(), flush=True)

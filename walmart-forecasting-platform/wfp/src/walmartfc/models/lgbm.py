"""Global LightGBM: one point model (weighted L1 => median / WAPE-aligned) + q10/q90 models, direct multi-horizon.

Target = y / scale where scale is a past-only per-series level; predictions are multiplied back, so outputs
are dollars. Sample weight = scale, so the training loss equals dollar-denominated error (not unit-free error).
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..features import CATEGORICAL, feature_columns


def _prep(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    X = df[cols].copy()
    for c in CATEGORICAL:
        if c in X:
            X[c] = X[c].fillna(0).astype(int)
    for c in X.columns:
        if c not in CATEGORICAL:
            X[c] = X[c].astype("float32")
    return X


def _params(cfg, objective: str, alpha: float | None = None) -> dict:
    p = cfg["lightgbm"]
    d = dict(objective=objective, learning_rate=p["learning_rate"], num_leaves=p["num_leaves"],
             min_child_samples=p["min_child_samples"], feature_fraction=p["feature_fraction"],
             bagging_fraction=p["bagging_fraction"], bagging_freq=p["bagging_freq"], lambda_l2=p["lambda_l2"],
             seed=p["seed"], verbose=-1, num_threads=1, max_bin=63)
    if alpha is not None:
        d["alpha"] = alpha
    return d


class GlobalLGBM:
    def __init__(self, cfg):
        self.cfg = cfg
        self.models: dict[str, lgb.Booster] = {}
        self.best_iter: dict[str, int] = {}
        self.cols: list[str] = []

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None, fixed_iters: dict | None = None):
        q = self.cfg["forecast"]["quantiles"]
        specs = {"point": ("l1", None), "q_lo": ("quantile", q[0]), "q_hi": ("quantile", q[-1])}
        train = train[train.eligible & train.y_scaled.notna() & np.isfinite(train.y_scaled)]
        self.cols = feature_columns(train)
        Xt = _prep(train, self.cols)
        for name, (obj, alpha) in specs.items():
            ds = lgb.Dataset(Xt, train.y_scaled.astype("float32"), weight=train.scale.astype("float32"),
                             categorical_feature=[c for c in CATEGORICAL if c in self.cols], free_raw_data=True)
            p = _params(self.cfg, obj, alpha)
            if fixed_iters is not None:
                self.models[name] = lgb.train(p, ds, num_boost_round=fixed_iters[name])
                self.best_iter[name] = fixed_iters[name]
            else:
                v = val[val.eligible & val.y_scaled.notna() & np.isfinite(val.y_scaled)]
                dv = lgb.Dataset(_prep(v, self.cols), v.y_scaled.astype("float32"), weight=v.scale.astype("float32"),
                                 reference=ds)
                m = lgb.train(p, ds, num_boost_round=self.cfg["lightgbm"]["n_estimators"], valid_sets=[dv],
                              callbacks=[lgb.early_stopping(self.cfg["lightgbm"]["early_stopping_rounds"], verbose=False)])
                self.models[name] = m
                self.best_iter[name] = max(int(m.best_iteration), 20)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = _prep(df, self.cols)
        out = df[["sid", "origin", "h", "target_idx", "scale", "eligible"]].copy()
        s = df.scale.to_numpy()
        raw = {k: m.predict(X) * s for k, m in self.models.items()}
        out["y_hat"] = raw["point"]
        lo, hi = np.minimum(raw["q_lo"], raw["q_hi"]), np.maximum(raw["q_lo"], raw["q_hi"])
        out["q_lo"], out["q_hi"] = lo, hi
        for c in ("y_hat", "q_lo", "q_hi"):
            out.loc[~df.eligible.to_numpy(), c] = np.nan
        if "y" in df:
            out["y"] = df["y"].to_numpy()
        return out

    def importance(self) -> pd.DataFrame:
        m = self.models["point"]
        return (pd.DataFrame(dict(feature=self.cols, gain=m.feature_importance("gain")))
                .sort_values("gain", ascending=False).assign(gain_share=lambda d: d.gain / d.gain.sum()))

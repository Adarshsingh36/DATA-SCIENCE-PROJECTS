"""Read-only data access for the dashboard (parquet/CSV artifacts produced by the pipeline)."""
from __future__ import annotations

import json
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROOT, load_config, resolve

LEVEL_LABEL = {"total": "Total", "group": "Group (Type; state proxy)", "store": "Store", "dept": "Department",
               "store_dept": "Store x Department"}


class Store:
    def __init__(self, artifacts: Path | None = None, results: Path | None = None):
        cfg = load_config()
        self.cfg = cfg
        self.art = Path(artifacts) if artifacts else resolve(cfg, "artifacts_dir")
        self.res = Path(results) if results else ROOT / "reports/results"
        with open(self.art / "prepared.pkl", "rb") as f:
            prep = pickle.load(f)
        self.bottom, self.hier = prep["panel_bottom"], prep["hier"]
        self.rows = self.hier.rows.reset_index(drop=True)
        self.dates = self.bottom.dates
        self.folds = sorted(p.name.replace("fold_", "") for p in self.art.glob("fold_*") if (p / "metrics.csv").exists())
        self._fc, self._met, self._int = {}, {}, {}

    def forecasts(self, fold: str) -> pd.DataFrame:
        if fold not in self._fc:
            d = pd.read_parquet(self.art / f"fold_{fold}" / "test_forecasts.parquet")
            self._fc[fold] = d
        return self._fc[fold]

    def selected(self, fold: str) -> str:
        dm = self.cfg["hierarchy"]["default_method"]
        if dm != "validation_selected":
            return f"lgbm_{dm}"
        p = self.art / f"fold_{fold}" / "info.json"
        return json.loads(p.read_text()).get("selected_reconciliation", "lgbm_bottom_up") if p.exists() else "lgbm_bottom_up"

    def metrics(self, fold: str) -> pd.DataFrame:
        if fold not in self._met:
            self._met[fold] = pd.read_csv(self.art / f"fold_{fold}" / "metrics.csv")
        return self._met[fold]

    def interval_metrics(self, fold: str) -> pd.DataFrame:
        if fold not in self._int:
            self._int[fold] = pd.read_csv(self.art / f"fold_{fold}" / "interval_metrics.csv")
        return self._int[fold]

    def history(self, node: int) -> pd.Series:
        Y = self.hier.aggregate(self.bottom.Y)[node]
        obs = np.asarray(self.hier.S[node] @ (~np.isnan(self.bottom.Y)).astype(float)).ravel() > 0
        return pd.Series(np.where(obs, Y, np.nan), index=self.dates)

    def test_dates(self, fold: str) -> pd.DatetimeIndex:
        f = self.forecasts(fold)
        idx = np.sort(f.target_idx.unique())
        return self.dates[idx]

    def node_options(self, level: str) -> list[dict]:
        r = self.rows[self.rows.level == level]
        return [{"label": n, "value": int(i)} for i, n in zip(r.index, r.name)]

    def rep_metrics(self) -> pd.DataFrame | None:
        p = self.res / "representative_metrics.csv"
        return pd.read_csv(p) if p.exists() else None

    def rep_intervals(self) -> pd.DataFrame | None:
        p = self.res / "representative_interval_metrics.csv"
        return pd.read_csv(p) if p.exists() else None

    def rep_forecasts(self, fold: str) -> pd.DataFrame | None:
        p = self.art / f"fold_{fold}" / "representative_forecasts.parquet"
        return pd.read_parquet(p) if p.exists() else None

    def inventory(self) -> pd.DataFrame | None:
        p = self.art / "inventory_plan.parquet"
        return pd.read_parquet(p) if p.exists() else None

    def inventory_summary(self) -> pd.DataFrame | None:
        p = self.res / "inventory_policy_comparison.csv"
        return pd.read_csv(p) if p.exists() else None

    def inventory_solver(self) -> pd.DataFrame | None:
        p = self.res / "inventory_solver_summary.csv"
        return pd.read_csv(p) if p.exists() else None

"""End-to-end orchestration. Usage: python -m walmartfc.pipeline <stage> [--config path]

Stages: prepare | eda | forecast | representative | inventory | all
Each stage reads/writes ``artifacts/`` so stages can be re-run independently.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from .config import ROOT, load_config, resolve
from .data.preprocess import PreprocessingLog, load_raw, merge_tables, validate_raw
from .panel import build_bottom_panel, build_full_panel, build_hierarchy, load_state_mapping
from .splits import make_fold


def _log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def artifacts_dir(cfg) -> Path:
    p = resolve(cfg, "artifacts_dir")
    p.mkdir(parents=True, exist_ok=True)
    return p


def prepare(cfg):
    """Load -> validate -> merge -> panels -> hierarchy. Cached to artifacts/prepared.pkl."""
    train, stores, features = load_raw(resolve(cfg, "raw_dir"))
    rep = validate_raw(train, stores, features)
    log = PreprocessingLog()
    df = merge_tables(train, stores, features, log)
    bottom = build_bottom_panel(df)
    smap = load_state_mapping(ROOT / cfg["hierarchy"]["state_mapping_csv"])
    hier = build_hierarchy(bottom.series, smap)
    full = build_full_panel(bottom, hier)
    miss = float(np.isnan(bottom.Y).mean())
    log.add("grid completeness", f"series x week grid {bottom.Y.shape[0]} x {bottom.Y.shape[1]}; unobserved cells kept as NaN "
            "(not zero) for features/evaluation; treated as 0 only when summing to aggregates", int(np.isnan(bottom.Y).sum()),
            f"{miss:.1%} of cells are unobserved (series start late, end early or have gaps).")
    log.add("geographic level", f"hierarchy group level = {hier.group_label}", int((hier.rows.level == 'group').sum()),
            "stores.csv has no state/region. Supply config/store_state_mapping.csv (Store,State) to use real states; "
            "no mapping is invented.")
    out = dict(panel_bottom=bottom, panel_full=full, hier=hier, validation=rep, merged_rows=len(df))
    with open(artifacts_dir(cfg) / "prepared.pkl", "wb") as f:
        pickle.dump(out, f)
    (ROOT / "reports/results").mkdir(parents=True, exist_ok=True)
    log.to_json(ROOT / "reports/results/preprocessing_log.json")
    (ROOT / "reports/results/preprocessing_log.md").write_text(log.to_markdown())
    with open(ROOT / "reports/results/data_validation.json", "w") as f:
        json.dump(rep, f, indent=2, default=str)
    _log(f"prepared: {bottom.Y.shape} bottom series, hierarchy {hier.S.shape}")
    return out


def load_prepared(cfg):
    p = artifacts_dir(cfg) / "prepared.pkl"
    if not p.exists():
        return prepare(cfg)
    with open(p, "rb") as f:
        return pickle.load(f)


def selected_model(cfg, fold: str) -> str:
    """Default model for downstream use: config hierarchy.default_method, or the validation-selected method."""
    dm = cfg["hierarchy"]["default_method"]
    if dm != "validation_selected":
        return f"lgbm_{dm}"
    p = artifacts_dir(cfg) / f"fold_{fold}" / "info.json"
    if p.exists():
        return json.loads(p.read_text()).get("selected_reconciliation", "lgbm_bottom_up")
    return "lgbm_bottom_up"


def get_folds(cfg, panel):
    fc = cfg["forecast"]
    return {n: make_fold(panel, n, d, fc["horizon"], fc["val_weeks"]) for n, d in fc["folds"].items()}


def forecast(cfg, only=None):
    from .forecasting import run_fold
    prep = load_prepared(cfg)
    bottom, full, hier = prep["panel_bottom"], prep["panel_full"], prep["hier"]
    ad = artifacts_dir(cfg)
    for name, fold in get_folds(cfg, bottom).items():
        if only and name not in only:
            continue
        _log(f"=== fold {name}: val_start={bottom.dates[fold.val_start].date()} test={bottom.dates[fold.test_start].date()}.."
             f"{bottom.dates[fold.test_start + fold.H - 1].date()}")
        r = run_fold(bottom, full, hier, fold, cfg, log=_log)
        d = ad / f"fold_{name}"
        d.mkdir(exist_ok=True)
        r.test.to_parquet(d / "test_forecasts.parquet")
        r.metrics.to_csv(d / "metrics.csv", index=False)
        r.intervals.to_csv(d / "interval_metrics.csv", index=False)
        r.importance.to_csv(d / "feature_importance.csv", index=False)
        with open(d / "info.json", "w") as f:
            json.dump(r.info, f, indent=2, default=str)
        _log(f"fold {name} saved")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "eda", "forecast", "representative", "inventory", "all"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--fold", action="append", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    stages = ["prepare", "eda", "forecast", "representative", "inventory"] if a.stage == "all" else [a.stage]
    for s in stages:
        if s == "prepare":
            prepare(cfg)
        elif s == "forecast":
            forecast(cfg, a.fold)
        elif s == "eda":
            from .eda import run_eda
            run_eda(cfg, load_prepared(cfg))
        elif s == "representative":
            from .representative import run_representative
            run_representative(cfg, load_prepared(cfg), only=a.fold)
        elif s == "inventory":
            from .inventory import run_inventory
            run_inventory(cfg, load_prepared(cfg))


if __name__ == "__main__":
    main()

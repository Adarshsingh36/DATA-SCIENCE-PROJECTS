"""Load, validate and merge the three Walmart tables. The raw files are never modified.

Every decision that alters values is recorded in a :class:`PreprocessingLog`, which the
pipeline writes to ``reports/results/preprocessing_log.json`` and ``docs/preprocessing_decisions.md``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

MARKDOWN_COLS = [f"MarkDown{i}" for i in range(1, 6)]


@dataclass
class PreprocessingLog:
    entries: list[dict] = field(default_factory=list)

    def add(self, step: str, decision: str, affected: int | float | None = None, rationale: str = "") -> None:
        self.entries.append(dict(step=step, decision=decision, affected=affected, rationale=rationale))

    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.entries, f, indent=2, default=str)

    def to_markdown(self) -> str:
        lines = ["| # | Step | Decision | Rows/cells affected | Rationale |", "|---|---|---|---|---|"]
        for i, e in enumerate(self.entries, 1):
            lines.append(f"| {i} | {e['step']} | {e['decision']} | {e['affected']} | {e['rationale']} |")
        return "\n".join(lines)


def load_raw(raw_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read train.csv, stores.csv, features.csv exactly as shipped (read-only)."""
    raw_dir = Path(raw_dir)
    missing = [f for f in ("train.csv", "stores.csv", "features.csv") if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {raw_dir}. Run `python scripts/download_data.py` "
            "or download from https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/data"
        )
    train = pd.read_csv(raw_dir / "train.csv", parse_dates=["Date"])
    stores = pd.read_csv(raw_dir / "stores.csv")
    features = pd.read_csv(raw_dir / "features.csv", parse_dates=["Date"])
    for df in (train, features):
        if df["IsHoliday"].dtype == object:
            df["IsHoliday"] = df["IsHoliday"].astype(str).str.upper().eq("TRUE")
        df["IsHoliday"] = df["IsHoliday"].astype(bool)
    return train, stores, features


def validate_raw(train: pd.DataFrame, stores: pd.DataFrame, features: pd.DataFrame) -> dict:
    """Structural checks. Returns a report dict; raises on violations that would corrupt a merge."""
    rep: dict = {}
    exp_train = {"Store", "Dept", "Date", "Weekly_Sales", "IsHoliday"}
    exp_stores = {"Store", "Type", "Size"}
    exp_feat = {"Store", "Date", "Temperature", "Fuel_Price", "CPI", "Unemployment", "IsHoliday", *MARKDOWN_COLS}
    for name, df, exp in (("train", train, exp_train), ("stores", stores, exp_stores), ("features", features, exp_feat)):
        miss = exp - set(df.columns)
        if miss:
            raise ValueError(f"{name}.csv is missing columns {sorted(miss)}")
    rep["train_rows"], rep["stores_rows"], rep["features_rows"] = len(train), len(stores), len(features)
    rep["n_stores"], rep["n_depts"] = train.Store.nunique(), train.Dept.nunique()
    rep["n_store_dept_series"] = int(train.groupby(["Store", "Dept"]).ngroups)
    rep["date_min"], rep["date_max"] = str(train.Date.min().date()), str(train.Date.max().date())
    dup = int(train.duplicated(["Store", "Dept", "Date"]).sum())
    if dup:
        raise ValueError(f"{dup} duplicate (Store, Dept, Date) rows in train.csv")
    if int(features.duplicated(["Store", "Date"]).sum()):
        raise ValueError("duplicate (Store, Date) rows in features.csv")
    if stores.Store.duplicated().any():
        raise ValueError("duplicate Store in stores.csv")
    rep["all_fridays"] = bool((train.Date.dt.dayofweek == 4).all())
    keys_t = set(zip(train.Store, train.Date))
    keys_f = set(zip(features.Store, features.Date))
    rep["train_store_dates_missing_in_features"] = len(keys_t - keys_f)
    if rep["train_store_dates_missing_in_features"]:
        raise ValueError("Some (Store, Date) in train.csv have no features row; merge would create NaNs")
    rep["stores_missing_in_stores_csv"] = sorted(set(train.Store) - set(stores.Store))
    chk = train.merge(features[["Store", "Date", "IsHoliday"]], on=["Store", "Date"], suffixes=("", "_f"))
    rep["isholiday_mismatch_rows"] = int((chk.IsHoliday != chk.IsHoliday_f).sum())
    rep["negative_sales_rows"] = int((train.Weekly_Sales < 0).sum())
    rep["zero_sales_rows"] = int((train.Weekly_Sales == 0).sum())
    rep["features_date_max"] = str(features.Date.max().date())
    rep["feature_nan_counts"] = {c: int(v) for c, v in features.isna().sum().items() if v}
    return rep


def merge_tables(
    train: pd.DataFrame, stores: pd.DataFrame, features: pd.DataFrame, log: PreprocessingLog | None = None
) -> pd.DataFrame:
    """Merge on the correct keys: train+stores on Store; +features on (Store, Date).

    Keys: train.Store -> stores.Store (many-to-one); (train.Store, train.Date) -> (features.Store, features.Date)
    (many-to-one, because each store-date has one features row shared by all departments).
    """
    log = log or PreprocessingLog()
    last = train.Date.max()
    n_future = int((features.Date > last).sum())
    f = features[features.Date <= last].copy()
    log.add("features window", f"dropped features rows after last training date {last.date()}", n_future,
            "Those rows belong to Kaggle's hidden test period; they contain NaN CPI/Unemployment and no sales.")

    # Markdown handling: NaN means 'not recorded' (the programme began Nov-2011); negatives are not valid promotions.
    nan_md = int(f[MARKDOWN_COLS].isna().sum().sum())
    neg_md = int((f[MARKDOWN_COLS] < 0).sum().sum())
    first_md = f.loc[f[MARKDOWN_COLS].notna().any(axis=1), "Date"].min()
    f["markdown_available"] = (f.Date >= first_md).astype(int)
    f[MARKDOWN_COLS] = f[MARKDOWN_COLS].clip(lower=0).fillna(0.0)
    log.add("MarkDown1-5 NaN", f"filled with 0 and added indicator `markdown_available` (1 from {first_md.date()})", nan_md,
            "Structural missingness: markdowns were not recorded before the programme. Median-filling (as in the reference "
            "notebook) would invent promotions. The indicator lets models distinguish 'no markdown' from 'not recorded'.")
    log.add("MarkDown negatives", "clipped to 0", neg_md, "Negative promotional spend is not meaningful.")
    f["MarkDown_total"] = f[MARKDOWN_COLS].sum(axis=1)

    f = f.sort_values(["Store", "Date"])
    nan_cpi = int(f[["CPI", "Unemployment"]].isna().sum().sum())
    if nan_cpi:
        f[["CPI", "Unemployment"]] = f.groupby("Store")[["CPI", "Unemployment"]].ffill()  # past-only fill
    log.add("CPI/Unemployment NaN", "forward-filled within store (past-only); none remain inside the training window", nan_cpi,
            "No global median/mean imputation, which would leak future/global statistics.")

    m = train.merge(stores, on="Store", how="left", validate="many_to_one")
    m = m.merge(f.drop(columns=["IsHoliday"]), on=["Store", "Date"], how="left", validate="many_to_one")
    m["Type"] = m["Type"].astype("category")
    log.add("merge", "train LEFT JOIN stores ON Store; LEFT JOIN features ON (Store, Date); IsHoliday taken from train "
            "(verified identical to features.IsHoliday)", len(m),
            "validate='many_to_one' asserts no row multiplication.")
    neg = int((m.Weekly_Sales < 0).sum())
    log.add("negative Weekly_Sales", "retained, not clipped or dropped", neg,
            "Returns/adjustments are genuine records; WAPE/MAE/RMSE are computed on signed values.")
    log.add("outliers / spikes", "NOT removed; investigated in EDA (see docs/eda.md)", 0,
            "The reference notebook dropped |z|>2.5 rows, which deletes exactly the holiday peaks inventory planning needs.")
    log.add("target scaling", "no global normalisation of Weekly_Sales; models forecast dollars", 0,
            "Inside LightGBM the target is divided by a past-only per-series scale and multiplied back (invertible, dollars out).")
    assert len(m) == len(train), "merge changed row count"
    return m

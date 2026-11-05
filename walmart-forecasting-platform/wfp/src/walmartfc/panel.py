"""Panel (series x weeks) data structure and hierarchy construction."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .calendar import calendar_frame

EXOG_COLS = ["Temperature", "Fuel_Price", "CPI", "Unemployment", "MarkDown_total", "markdown_available"]


@dataclass
class Panel:
    dates: pd.DatetimeIndex
    series: pd.DataFrame            # one row per series: name, level, Store, Dept, type_code, Size
    Y: np.ndarray                   # (n_series, T) weekly sales, NaN = not observed
    exog: dict[str, np.ndarray]     # name -> (n_series, T)
    cal: pd.DataFrame               # (T, calendar features)

    @property
    def n(self) -> int:
        return self.Y.shape[0]

    @property
    def T(self) -> int:
        return self.Y.shape[1]

    def week_index(self, date) -> int:
        return int(self.dates.get_loc(pd.Timestamp(date)))


def build_bottom_panel(df: pd.DataFrame) -> Panel:
    """Merged long table -> bottom-level (Store, Dept) panel on a complete weekly grid."""
    dates = pd.DatetimeIndex(sorted(df.Date.unique()))
    gaps = np.unique(np.diff(dates.values).astype("timedelta64[D]").astype(int))
    if len(gaps) != 1 or gaps[0] != 7:
        raise ValueError(f"weekly grid is not contiguous: gaps={gaps}")
    keys = df[["Store", "Dept"]].drop_duplicates().sort_values(["Store", "Dept"]).reset_index(drop=True)
    sid = {(s, d): i for i, (s, d) in enumerate(zip(keys.Store, keys.Dept))}
    ti = pd.Series(np.arange(len(dates)), index=dates)
    rows = np.fromiter((sid[(s, d)] for s, d in zip(df.Store, df.Dept)), dtype=np.int64, count=len(df))
    cols = ti.loc[df.Date].to_numpy()
    Y = np.full((len(keys), len(dates)), np.nan)
    Y[rows, cols] = df.Weekly_Sales.to_numpy()

    stores = df[["Store", "Type", "Size"]].drop_duplicates("Store").set_index("Store").sort_index()
    series = keys.copy()
    series["Type"] = series.Store.map(stores["Type"].astype(str))
    series["Size"] = series.Store.map(stores["Size"]).astype(float)
    series["type_code"] = series.Type.map({"A": 0, "B": 1, "C": 2}).astype(int)
    series["level"] = "store_dept"
    series["name"] = series.Store.astype(str).str.zfill(2) + "-" + series.Dept.astype(str).str.zfill(2)

    # store-level exogenous variables on the same grid, then broadcast to series
    f = df.drop_duplicates(["Store", "Date"])
    exog = {}
    for c in EXOG_COLS:
        piv = f.pivot(index="Store", columns="Date", values=c).reindex(index=stores.index, columns=dates)
        exog[c] = piv.loc[series.Store.to_numpy()].to_numpy(dtype=float)
    hol = df.drop_duplicates("Date").set_index("Date").reindex(dates)["IsHoliday"].to_numpy()
    return Panel(dates, series, Y, exog, calendar_frame(dates, hol))


@dataclass
class Hierarchy:
    """Grouped hierarchy: Total > Group(State|Type) > Store > Store-Dept, plus Dept (across stores)."""
    S: sp.csr_matrix                # (n_all, n_bottom) summing matrix
    rows: pd.DataFrame              # one row per hierarchy node: level, name, Store, Dept, group
    group_label: str                # 'State' or 'Type (state proxy)'
    n_bottom: int
    level_slices: dict[str, slice] = field(default_factory=dict)

    def aggregate(self, Yb: np.ndarray) -> np.ndarray:
        """Aggregate bottom values to all nodes. NaN (unobserved) counts as 0 sales for sums."""
        return np.asarray(self.S @ np.nan_to_num(Yb, nan=0.0))


def load_state_mapping(path: str | Path | None) -> pd.DataFrame | None:
    if path and Path(path).exists():
        m = pd.read_csv(path)
        if {"Store", "State"} <= set(m.columns) and m.State.notna().all() and len(m):
            return m[["Store", "State"]]
    return None


def build_hierarchy(bottom_series: pd.DataFrame, state_map: pd.DataFrame | None = None) -> Hierarchy:
    b = bottom_series.reset_index(drop=True).copy()
    if state_map is not None:
        b["group"] = b.Store.map(state_map.set_index("Store")["State"])
        if b.group.isna().any():
            raise ValueError("state mapping does not cover every store")
        label = "State"
    else:
        b["group"] = "Type " + b.Type
        label = "Type (state proxy; no state in Walmart files)"
    m = len(b)
    blocks, meta = [], []

    def add(level: str, keycols: list[str], namefn):
        keys = b[keycols].drop_duplicates().sort_values(keycols).reset_index(drop=True) if keycols else pd.DataFrame(index=[0])
        data, ri, ci = [], [], []
        for r, (_, k) in enumerate(keys.iterrows()):
            mask = np.ones(m, bool)
            for c in keycols:
                mask &= (b[c] == k[c]).to_numpy()
            idx = np.where(mask)[0]
            ri.extend([r] * len(idx)); ci.extend(idx.tolist()); data.extend([1.0] * len(idx))
            meta.append(dict(level=level, name=namefn(k),
                             Store=int(k["Store"]) if "Store" in keycols else 0,
                             Dept=int(k["Dept"]) if "Dept" in keycols and level == "dept" else 0,
                             group=k["group"] if "group" in keycols else ""))
        blocks.append(sp.csr_matrix((data, (ri, ci)), shape=(len(keys), m)))

    add("total", [], lambda k: "Total")
    add("group", ["group"], lambda k: str(k["group"]))
    add("store", ["Store"], lambda k: f"Store {int(k['Store']):02d}")
    add("dept", ["Dept"], lambda k: f"Dept {int(k['Dept']):02d}")
    bottom_meta = pd.DataFrame(dict(level="store_dept", name=b.name, Store=b.Store, Dept=b.Dept, group=b.group))
    blocks.append(sp.identity(m, format="csr"))
    rows = pd.concat([pd.DataFrame(meta), bottom_meta], ignore_index=True)
    S = sp.vstack(blocks, format="csr")
    sl, start = {}, 0
    for lvl in ["total", "group", "store", "dept", "store_dept"]:
        n = int((rows.level == lvl).sum())
        sl[lvl] = slice(start, start + n); start += n
    return Hierarchy(S=S, rows=rows, group_label=label, n_bottom=m, level_slices=sl)


def build_full_panel(bottom: Panel, hier: Hierarchy) -> Panel:
    """Panel for *all* hierarchy nodes (aggregates by summation, exogenous by store-weighted mean)."""
    Y_all = hier.aggregate(bottom.Y)
    # an aggregate series is 'observed' whenever any bottom series contributes that week
    obs = hier.S @ (~np.isnan(bottom.Y)).astype(float)
    Y_all = np.where(np.asarray(obs) > 0, Y_all, np.nan)
    Sb = hier.S.copy().astype(float)
    wsum = np.asarray(Sb.sum(axis=1)).ravel()
    exog = {k: np.asarray(Sb @ np.nan_to_num(v)) / wsum[:, None] for k, v in bottom.exog.items()}
    r = hier.rows
    size_all = np.asarray(Sb @ bottom.series.Size.to_numpy()) / wsum
    dom_type = np.zeros(len(r), int)
    ser = pd.DataFrame(dict(name=r.name.to_numpy(), level=r.level.to_numpy(), Store=r.Store.to_numpy(),
                            Dept=r.Dept.to_numpy(), Type=r.group.to_numpy(), Size=size_all, type_code=dom_type))
    bt = bottom.series
    ser.loc[hier.level_slices["store_dept"], "type_code"] = bt.type_code.to_numpy()
    ser.loc[hier.level_slices["store_dept"], "Type"] = bt.Type.to_numpy()
    return Panel(bottom.dates, ser, Y_all, exog, bottom.cal)

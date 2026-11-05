"""Pure figure builders (no Dash state) so they can be unit-tested."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .data import LEVEL_LABEL, Store

MODEL_LABEL = {"naive": "Naive", "snaive": "Seasonal naive (52w)", "lgbm": "LightGBM (base)",
               "lgbm_bottom_up": "LightGBM + bottom-up", "lgbm_ols": "LightGBM + OLS",
               "lgbm_wls_struct": "LightGBM + WLS-struct", "lgbm_mint_shrink": "LightGBM + MinT-shrink (hold-out errors)", "lgbm_mint_shrink_insample": "LightGBM + MinT-shrink (in-sample naive)",
               "prophet": "Prophet", "tft": "Temporal Fusion Transformer"}
PALETTE = ["#1f4e79", "#c0392b", "#2e8b57", "#8e44ad", "#d68910", "#17a589", "#7f8c8d", "#e67e22", "#2980b9"]


def label(m: str) -> str:
    return MODEL_LABEL.get(m, m)


def forecast_figure(st: Store, fold: str, node: int, model: str, history_weeks: int = 52) -> go.Figure:
    fc = st.forecasts(fold)
    off = st.hier.S.shape[0] - st.hier.S.shape[1]
    name = st.rows.name[node]
    d = None
    if model in ("prophet", "tft"):
        rf = st.rep_forecasts(fold)
        if rf is not None and node >= off:
            d = rf[(rf.model == model) & (rf.sid == node - off)].sort_values("h")
    else:
        d = fc[(fc.model == model) & (fc.node == node)].sort_values("h")
    hist = st.history(node)
    tdates = st.test_dates(fold)
    first = tdates[0]
    h = hist[hist.index < first].iloc[-history_weeks:]
    fig = go.Figure()
    fig.add_scatter(x=h.index, y=h.values, name="History", line=dict(color="#555"))
    act = hist[hist.index.isin(tdates)]
    fig.add_scatter(x=act.index, y=act.values, name="Actual (held out)", line=dict(color="#111", width=3))
    if d is None or d.empty:
        fig.add_annotation(text="No forecast for this model/series (Prophet and TFT cover a representative store-department subset)",
                           showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)
    else:
        x = tdates[d.h.to_numpy() - 1]
        if d.lo.notna().any():
            fig.add_scatter(x=x, y=d.hi, line=dict(width=0), showlegend=False, hoverinfo="skip")
            fig.add_scatter(x=x, y=d.lo, fill="tonexty", fillcolor="rgba(31,78,121,0.18)", line=dict(width=0),
                            name="80% prediction interval")
        fig.add_scatter(x=x, y=d.y_hat, name=label(model), line=dict(color="#1f4e79", width=3, dash="dot"))
    fig.update_layout(title=f"{name}: weekly sales forecast vs actual ({fold} fold)", yaxis_title="Weekly sales ($)",
                      template="plotly_white", hovermode="x unified", legend=dict(orientation="h", y=-0.2))
    return fig


def metric_table(st: Store, fold: str, level: str) -> pd.DataFrame:
    m = st.metrics(fold)
    m = m[m.level == level][["model", "WAPE", "RMSSE", "RMSSE_w", "MAE", "RMSE", "Bias", "n_obs"]].copy()
    m["model"] = m.model.map(label)
    return m.sort_values("WAPE").reset_index(drop=True)


def wape_bar(st: Store, fold: str, level: str) -> go.Figure:
    t = metric_table(st, fold, level)
    fig = go.Figure(go.Bar(x=t.model, y=t.WAPE * 100, marker_color=PALETTE[0], text=(t.WAPE * 100).round(2), textposition="outside"))
    fig.update_layout(title=f"WAPE (%) by model - {LEVEL_LABEL[level]} level, {fold} fold", yaxis_title="WAPE %",
                      template="plotly_white")
    return fig


def horizon_curve(st: Store, fold: str) -> go.Figure:
    m = st.metrics(fold)
    m = m[m.level.str.startswith("store_dept@h")].copy()
    m["h"] = m.level.str.extract(r"@h(\d+)").astype(int)
    fig = go.Figure()
    for i, (mod, g) in enumerate(m.groupby("model")):
        fig.add_scatter(x=g.sort_values("h").h, y=g.sort_values("h").WAPE * 100, name=label(mod), mode="lines+markers")
    fig.update_layout(title="Store x Department WAPE by forecast horizon", xaxis_title="Weeks ahead", yaxis_title="WAPE %",
                      template="plotly_white")
    return fig


def hierarchy_levels_figure(st: Store, fold: str, models: list[str]) -> go.Figure:
    m = st.metrics(fold)
    m = m[m.level.isin(["total", "group", "store", "dept", "store_dept"]) & m.model.isin(models)]
    fig = go.Figure()
    order = ["total", "group", "store", "dept", "store_dept"]
    for mod in models:
        g = m[m.model == mod].set_index("level").reindex(order)
        fig.add_bar(x=[LEVEL_LABEL[x] for x in order], y=g.WAPE * 100, name=label(mod))
    fig.update_layout(barmode="group", title=f"WAPE % by hierarchy level: base vs reconciled ({fold})", template="plotly_white",
                      yaxis_title="WAPE %")
    return fig


def contribution_treemap(st: Store, fold: str, model: str | None = None) -> go.Figure:
    model = model or st.selected(fold)
    fc = st.forecasts(fold)
    off = st.hier.S.shape[0] - st.hier.S.shape[1]
    d = fc[(fc.model == model) & (fc.node >= off)].copy()
    d["abs_err"] = (d.y - d.y_hat).abs()
    meta = st.rows.loc[d.node, ["Store", "Dept", "group"]].reset_index(drop=True)
    d = d.reset_index(drop=True).join(meta)
    g = d.groupby(["group", "Store"]).agg(sales=("y", "sum"), err=("abs_err", "sum")).reset_index()
    g["Store"] = "Store " + g.Store.astype(str)
    g["wape"] = g.err / g.sales.abs()
    fig = go.Figure(go.Treemap(
        labels=g.Store, parents=g.group, values=g.sales.clip(lower=1), marker=dict(colors=g.wape * 100, colorscale="RdYlGn_r",
        colorbar=dict(title="WAPE %"), cmin=0, cmax=max(20, float((g.wape * 100).quantile(0.95)))), textinfo="label",
        hovertemplate="%{label}<br>sales $%{value:,.0f}<br>WAPE %{color:.1f}%<extra></extra>"))
    fig.update_layout(title=f"Revenue (size) and forecast error (colour) by {st.hier.group_label.split(' (')[0]} > Store",
                      template="plotly_white")
    return fig


def dept_error_figure(st: Store, fold: str, model: str | None = None) -> go.Figure:
    model = model or st.selected(fold)
    fc = st.forecasts(fold)
    sl = st.hier.level_slices["dept"]
    d = fc[(fc.model == model) & (fc.node >= sl.start) & (fc.node < sl.stop)]
    g = d.groupby("node").agg(sales=("y", "sum"), err=("y_hat", lambda s: 0)).reset_index()
    e = d.assign(ae=(d.y - d.y_hat).abs()).groupby("node").ae.sum()
    g["wape"] = g.node.map(e) / g.sales.abs()
    g["dept"] = st.rows.name[g.node].to_numpy()
    g = g.sort_values("sales", ascending=False).head(25)
    fig = go.Figure(go.Bar(x=g.dept, y=g.wape * 100, marker_color=PALETTE[1]))
    fig.update_layout(title="WAPE % for the 25 largest departments (all stores aggregated)", template="plotly_white",
                      yaxis_title="WAPE %")
    return fig


def inventory_figure(st: Store, store: int, dept: int) -> go.Figure:
    inv = st.inventory()
    fig = go.Figure()
    if inv is None:
        return fig
    d = inv[(inv.Store == store) & (inv.Dept == dept)].sort_values("h")
    if d.empty:
        fig.add_annotation(text="No plan for this store-department", showarrow=False, xref="paper", yref="paper", x=.5, y=.5)
        return fig
    fig.add_scatter(x=d.week, y=d.forecast_hi, line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=d.week, y=d.forecast_lo, fill="tonexty", fillcolor="rgba(31,78,121,0.18)", line=dict(width=0),
                    name="Demand p10-p90")
    fig.add_scatter(x=d.week, y=d.forecast_p50, name="Forecast p50", line=dict(color="#1f4e79", width=3))
    fig.add_scatter(x=d.week, y=d.actual_sales, name="Actual sales", line=dict(color="#111", dash="dot"))
    fig.add_bar(x=d.week, y=d.order_qty, name="Recommended order (placed)", marker_color="rgba(214,137,16,0.7)")
    fig.update_layout(title=f"Store {store} / Dept {dept}: forecast demand and recommended orders (scenario assumptions)",
                      yaxis_title="$ (retail sales value)", template="plotly_white", hovermode="x unified",
                      legend=dict(orientation="h", y=-0.2))
    return fig

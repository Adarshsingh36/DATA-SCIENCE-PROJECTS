"""Plotly Dash dashboard. Run: python -m walmartfc.dashboard.app  (after the pipeline has produced artifacts/)."""
from __future__ import annotations

import pandas as pd
from dash import Dash, Input, Output, dash_table, dcc, html

from . import figures as F
from .data import LEVEL_LABEL, Store


def _table(df: pd.DataFrame, pct_cols=(), **kw):
    d = df.copy()
    for c in pct_cols:
        if c in d:
            d[c] = (d[c] * 100).round(2)
    for c in d.columns:
        if d[c].dtype.kind == "f" and c not in pct_cols:
            d[c] = d[c].round(3) if d[c].abs().max() < 100 else d[c].round(0)
    return dash_table.DataTable(data=d.to_dict("records"), columns=[{"name": c, "id": c} for c in d.columns],
                                style_cell={"fontFamily": "Inter, Arial", "fontSize": 13, "padding": "6px", "textAlign": "right"},
                                style_header={"fontWeight": "bold", "backgroundColor": "#eef3f8"}, page_size=kw.get("page_size", 12),
                                style_table={"overflowX": "auto"}, sort_action="native")


def create_app(store: Store | None = None) -> Dash:
    st = store or Store()
    app = Dash(__name__, title="Walmart Demand Forecasting & Inventory", suppress_callback_exceptions=True)
    folds = st.folds
    fold_opts = [{"label": f"{f} fold", "value": f} for f in folds]
    models = sorted(st.forecasts(folds[0]).model.unique(), key=lambda m: list(F.MODEL_LABEL).index(m) if m in F.MODEL_LABEL else 99)
    model_opts = [{"label": F.label(m), "value": m} for m in models]
    if st.rep_metrics() is not None:
        model_opts += [{"label": F.label(m) + " (representative series only)", "value": m} for m in ("prophet", "tft")
                       if m in set(st.rep_metrics().model)]
    level_opts = [{"label": v, "value": k} for k, v in LEVEL_LABEL.items()]
    inv = st.inventory()
    store_opts = [{"label": f"Store {s}", "value": int(s)} for s in sorted(inv.Store.unique())] if inv is not None else []

    app.layout = html.Div(style={"maxWidth": "1300px", "margin": "0 auto", "fontFamily": "Inter, Arial"}, children=[
        html.H2("Walmart weekly demand forecasting, hierarchy and inventory planning"),
        html.P("Chronological backtests on Kaggle's Walmart Store Sales data. Holdout weeks are never used in training. "
               "Inventory costs, capacities and lead times are scenario assumptions, not Walmart data.", style={"color": "#555"}),
        dcc.Tabs(id="tabs", value="forecast", children=[
            dcc.Tab(label="Weekly forecasts", value="forecast"), dcc.Tab(label="Forecast accuracy", value="accuracy"),
            dcc.Tab(label="Hierarchical analysis", value="hier"), dcc.Tab(label="Inventory recommendations", value="inv")]),
        html.Div(id="tab-body", style={"paddingTop": "12px"}),
        # forecast controls
        html.Div(id="forecast-pane", children=[
            html.Div(style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}, children=[
                dcc.Dropdown(id="f-fold", options=fold_opts, value=folds[-1], clearable=False, style={"width": 180}),
                dcc.Dropdown(id="f-level", options=level_opts, value="total", clearable=False, style={"width": 260}),
                dcc.Dropdown(id="f-node", clearable=False, style={"width": 260}),
                dcc.Dropdown(id="f-model", options=model_opts, value=st.selected(folds[-1]), clearable=False, style={"width": 320})]),
            dcc.Graph(id="f-graph")]),
        html.Div(id="accuracy-pane", children=[
            html.Div(style={"display": "flex", "gap": "12px"}, children=[
                dcc.Dropdown(id="a-fold", options=fold_opts, value=folds[-1], clearable=False, style={"width": 180}),
                dcc.Dropdown(id="a-level", options=level_opts, value="store_dept", clearable=False, style={"width": 260})]),
            dcc.Graph(id="a-bar"), html.Div(id="a-table"), dcc.Graph(id="a-horizon"),
            html.H4("Prediction-interval quality (nominal 80%)"), html.Div(id="a-int"),
            html.H4("Representative-series comparison (all models incl. Prophet and TFT, same series)"), html.Div(id="a-rep")]),
        html.Div(id="hier-pane", children=[
            dcc.Dropdown(id="h-fold", options=fold_opts, value=folds[-1], clearable=False, style={"width": 180}),
            dcc.Graph(id="h-levels"), dcc.Graph(id="h-tree"), dcc.Graph(id="h-dept")]),
        html.Div(id="inv-pane", children=[
            html.Div(style={"display": "flex", "gap": "12px"}, children=[
                dcc.Dropdown(id="i-store", options=store_opts, value=store_opts[0]["value"] if store_opts else None,
                             clearable=False, style={"width": 180}),
                dcc.Dropdown(id="i-dept", clearable=False, style={"width": 180})]),
            dcc.Graph(id="i-graph"), html.H4("Policy comparison on realised demand (scenario assumptions)"),
            html.Div(id="i-summary"), html.H4("Order recommendations for the selected store"), html.Div(id="i-table")]),
    ])

    @app.callback(Output("forecast-pane", "style"), Output("accuracy-pane", "style"), Output("hier-pane", "style"),
                  Output("inv-pane", "style"), Input("tabs", "value"))
    def show(tab):
        s = lambda k: {} if tab == k else {"display": "none"}  # noqa: E731
        return s("forecast"), s("accuracy"), s("hier"), s("inv")

    @app.callback(Output("f-node", "options"), Output("f-node", "value"), Input("f-level", "value"))
    def nodes(level):
        o = st.node_options(level)
        return o, o[0]["value"]

    @app.callback(Output("f-graph", "figure"), Input("f-fold", "value"), Input("f-node", "value"), Input("f-model", "value"))
    def fgraph(fold, node, model):
        return F.forecast_figure(st, fold, int(node), model)

    @app.callback(Output("a-bar", "figure"), Output("a-table", "children"), Output("a-horizon", "figure"),
                  Output("a-int", "children"), Output("a-rep", "children"), Input("a-fold", "value"), Input("a-level", "value"))
    def acc(fold, level):
        t = F.metric_table(st, fold, level)
        im = st.interval_metrics(fold)
        im = im[(im.level == level)].copy()
        im["model"] = im.model.map(F.label)
        im = im[["model", "interval", "coverage", "nominal", "rel_width", "interval_score", "n"]]
        rep = st.rep_metrics()
        rep_t = html.P("Run `python -m walmartfc.pipeline representative` to populate.")
        if rep is not None:
            r = rep[rep.fold == fold].copy()
            r["model"] = r.model.map(F.label)
            rep_t = _table(r[["model", "n_series", "WAPE", "RMSSE", "RMSSE_w", "MAE", "RMSE", "Bias"]].sort_values("WAPE"),
                           pct_cols=("WAPE", "Bias", "coverage"))
        return (F.wape_bar(st, fold, level), _table(t, pct_cols=("WAPE", "Bias")), F.horizon_curve(st, fold),
                _table(im, pct_cols=("coverage", "nominal", "rel_width")), rep_t)

    @app.callback(Output("h-levels", "figure"), Output("h-tree", "figure"), Output("h-dept", "figure"), Input("h-fold", "value"))
    def hier(fold):
        ms = [m for m in ("snaive", "lgbm", "lgbm_bottom_up", "lgbm_wls_struct", "lgbm_mint_shrink", "lgbm_mint_shrink_insample") if m in set(st.metrics(fold).model)]
        return F.hierarchy_levels_figure(st, fold, ms), F.contribution_treemap(st, fold), F.dept_error_figure(st, fold)

    @app.callback(Output("i-dept", "options"), Output("i-dept", "value"), Input("i-store", "value"))
    def depts(store):
        if inv is None or store is None:
            return [], None
        ds = sorted(inv[inv.Store == store].Dept.unique())
        return [{"label": f"Dept {d}", "value": int(d)} for d in ds], int(ds[0])

    @app.callback(Output("i-graph", "figure"), Output("i-summary", "children"), Output("i-table", "children"),
                  Input("i-store", "value"), Input("i-dept", "value"))
    def invcb(store, dept):
        if inv is None:
            return F.go.Figure(), html.P("Run the inventory stage first."), html.Div()
        s = st.inventory_summary()
        sub = inv[inv.Store == store]
        tab = (sub.groupby("Dept").agg(total_forecast=("forecast_p50", "sum"), total_order=("order_qty", "sum"),
                                         first_order_week=("week", lambda w: w[sub.loc[w.index, "order_qty"] > 1e-6].min()),
                                         n_orders=("order_qty", lambda q: int((q > 1e-6).sum())),
                                         opening=("opening_inventory", "max")).reset_index())
        tab["first_order_week"] = tab.first_order_week.dt.strftime("%Y-%m-%d").fillna("-")
        return (F.inventory_figure(st, store, dept), _table(s.round(3)) if s is not None else html.Div(),
                _table(tab.sort_values("total_order", ascending=False), page_size=10))

    return app


def main():  # pragma: no cover
    create_app().run(debug=False, host="0.0.0.0", port=8050)


if __name__ == "__main__":  # pragma: no cover
    main()

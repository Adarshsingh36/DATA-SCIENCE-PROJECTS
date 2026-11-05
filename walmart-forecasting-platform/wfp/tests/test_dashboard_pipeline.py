from pathlib import Path

import pytest

ART = Path(__file__).resolve().parents[1] / "artifacts"
needs = pytest.mark.skipif(not (ART / "prepared.pkl").exists() or not list(ART.glob("fold_*/metrics.csv")),
                           reason="run the pipeline first")


@needs
def test_dashboard_builds_and_figures_render():
    from walmartfc.dashboard import figures as F
    from walmartfc.dashboard.app import create_app
    from walmartfc.dashboard.data import Store
    st = Store()
    app = create_app(st)
    assert app.layout is not None
    fold = st.folds[-1]
    for lvl in ("total", "store", "dept", "store_dept"):
        node = st.node_options(lvl)[0]["value"]
        fig = F.forecast_figure(st, fold, node, st.selected(fold))
        assert len(fig.data) >= 3
        assert not F.wape_bar(st, fold, lvl).data == ()
    assert len(F.horizon_curve(st, fold).data) >= 2
    assert len(F.hierarchy_levels_figure(st, fold, ["lgbm", "lgbm_bottom_up"]).data) == 2
    F.contribution_treemap(st, fold); F.dept_error_figure(st, fold)


@needs
def test_server_serves_layout_and_dependencies():
    from walmartfc.dashboard.app import create_app
    c = create_app().server.test_client()
    assert c.get("/").status_code == 200
    assert b"_dash-layout" in c.get("/_dash-layout").data or c.get("/_dash-layout").status_code == 200
    assert c.get("/_dash-dependencies").status_code == 200


@needs
def test_reconciled_forecasts_are_coherent_on_saved_results():
    import numpy as np
    import pandas as pd
    from walmartfc.dashboard.data import Store
    st = Store()
    fold = st.folds[-1]
    fc = st.forecasts(fold)
    sel = st.selected(fold)
    d = fc[fc.model == sel].pivot(index="node", columns="h", values="y_hat")
    S = st.hier.S
    off = S.shape[0] - S.shape[1]
    assert np.allclose(S @ d.iloc[off:].to_numpy(), d.to_numpy(), atol=1e-6)


@needs
def test_saved_test_window_is_strictly_after_validation_origins():
    import json
    from walmartfc.config import load_config
    from walmartfc.dashboard.data import Store
    cfg = load_config()
    st = Store()
    for fold in st.folds:
        f = st.forecasts(fold)
        expected_start = st.bottom.week_index(cfg["forecast"]["folds"][fold])
        assert f.target_idx.min() == expected_start
        assert f.target_idx.max() == expected_start + cfg["forecast"]["horizon"] - 1
        info = json.loads((ART / f"fold_{fold}" / "info.json").read_text())
        assert max(info["val_origins"]) < expected_start      # calibration origins precede the test window


@needs
def test_every_dashboard_callback_responds():
    from walmartfc.dashboard.app import create_app
    c = create_app().server.test_client()
    deps = c.get("/_dash-dependencies").get_json()
    vals = {"f-fold": "latest", "a-fold": "latest", "h-fold": "latest", "f-level": "store", "f-node": 3, "f-model": "lgbm_bottom_up",
            "a-level": "store_dept", "tabs": "forecast", "i-store": 1, "i-dept": 1}
    for d in deps:
        out = d["output"]
        outs = ([{"id": o.split(".")[0], "property": o.split(".")[1]} for o in out.strip(".").split("...")]
                if out.startswith("..") else {"id": out.split(".")[0], "property": out.split(".")[1]})
        body = {"output": out, "outputs": outs, "changedPropIds": [],
                "inputs": [{"id": i["id"], "property": i["property"], "value": vals[i["id"]]} for i in d["inputs"]]}
        assert c.post("/_dash-update-component", json=body).status_code == 200, out

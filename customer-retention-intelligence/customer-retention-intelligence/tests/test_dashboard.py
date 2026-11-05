import pytest
from pathlib import Path
from streamlit.testing.v1 import AppTest

D = Path(__file__).resolve().parents[1] / "dashboard"
PAGES = sorted((D / "pages").glob("*.py"))


def _db_ready():
    try:
        from dashboard.utils import load_master
        return len(load_master()) == 7043
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _db_ready(), reason="database not populated: run `python -m src.pipeline`")


def run(path, timeout=120):
    at = AppTest.from_file(str(path), default_timeout=timeout).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_app_entrypoint_starts():
    at = AppTest.from_file(str(D / "app.py"), default_timeout=120).run()
    assert not at.exception


@pytest.mark.parametrize("p", PAGES, ids=lambda p: p.stem)
def test_every_page_loads(p):
    at = run(p)
    assert len(at.title) == 1


def test_executive_metrics():
    at = run(D / "pages/1_Executive_Dashboard.py")
    labels = [m.label for m in at.metric]
    assert {"Total customers", "Actual churn rate", "Predicted churn rate"} <= set(labels)
    assert at.metric[0].value == "7,043"


def test_churn_search_controls():
    at = run(D / "pages/2_Churn_Intelligence.py")
    cid = at.selectbox[0].options[3]
    at.selectbox[0].select(cid).run(); assert not at.exception
    at.text_input[0].set_value("3668-QPYBK").run(); assert not at.exception
    assert at.selectbox[0].value == "3668-QPYBK"
    at.text_input[0].set_value("zzzz-nomatch").run(); assert at.warning


def test_survival_controls():
    at = run(D / "pages/3_Survival_Analysis.py")
    at.selectbox[0].select("contract").run(); assert not at.exception
    at.radio[0].set_value(at.radio[0].options[1]).run(); assert not at.exception
    at.selectbox[1].select(at.selectbox[1].options[5]).run(); assert not at.exception


def test_retention_controls():
    at = run(D / "pages/4_Retention_Intelligence.py")
    assert any("SIMULATED" in m.value for m in at.markdown)
    at.checkbox[0].set_value(False).run(); assert not at.exception


def test_segmentation_page():
    at = run(D / "pages/5_Customer_Segmentation.py"); assert len(at.dataframe) >= 1


def test_roi_simulator_controls_change_results():
    at = run(D / "pages/6_ROI_Simulator.py")
    def metric(name): return next(m.value for m in at.metric if m.label == name)
    base = metric("Net incremental benefit"); base_n = metric("Customers targeted")
    at.sidebar.slider[0].set_value(20.0).run(); assert not at.exception          # discount
    assert metric("Net incremental benefit") != base
    at.sidebar.number_input[1].set_value(50).run(); assert not at.exception      # n targeted
    assert int(metric("Customers targeted").replace(",", "")) <= 50
    at.sidebar.number_input[0].set_value(1000.0).run(); assert not at.exception  # budget
    for sb in at.sidebar.selectbox[0].options:
        at.sidebar.selectbox[0].select(sb).run(); assert not at.exception
    assert any("ROI" == m.label for m in at.metric)

"""Shared dashboard helpers: data loading (from the database), styling, disclosures."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import text

from database.connection import get_engine, backend_name
from src import config

QUERY = """
SELECT c.*, p.churn_probability, p.risk_tier, p.predicted_label,
       s.surv_prob_12m, s.surv_prob_24m, s.surv_prob_36m, s.expected_months_next_24, s.extrapolated, s.partial_hazard,
       v.clv, v.revenue_at_risk_annual,
       u.sim_treatment, u.sim_retained, u.ite_selected, u.ite_s_learner, u.ite_t_learner, u.selected_learner,
       u.reference_effect, u.quadrant, u.positive_uplift, u.expected_net_value, u.economically_worthwhile,
       g.segment_label, g.cluster_id, g.pc1, g.pc2
FROM customers c
JOIN churn_predictions p ON p.customer_id = c.customer_id
JOIN survival_predictions s ON s.customer_id = c.customer_id
JOIN customer_clv v ON v.customer_id = c.customer_id
JOIN uplift_predictions u ON u.customer_id = c.customer_id
JOIN customer_segments g ON g.customer_id = c.customer_id
"""

CSS = """
<style>
.block-container {padding-top: 1.4rem; max-width: 1400px;}
div[data-testid="stMetric"] {background: linear-gradient(135deg,#141b34,#1b2447); border: 1px solid #2a3566;
  border-radius: 14px; padding: 14px 16px;}
div[data-testid="stMetricValue"] {font-size: 1.7rem;}
h1, h2, h3 {letter-spacing: -0.01em;}
.sim-banner {background:#3b2a05; border:1px solid #b7791f; color:#fbd38d; padding:10px 14px; border-radius:10px; margin-bottom:12px;}
.note {background:#10233f; border:1px solid #2b4c7e; color:#bcd3f5; padding:10px 14px; border-radius:10px; margin:8px 0;}
</style>
"""


def setup(title: str, icon: str = "📊"):
    try:
        st.set_page_config(page_title=f"{title} | Retention Intelligence", page_icon=icon, layout="wide")
    except Exception:  # already set by app.py
        pass
    st.markdown(CSS, unsafe_allow_html=True)
    st.title(f"{icon} {title}")


def sim_banner():
    st.markdown('<div class="sim-banner"><b>Simulated uplift data.</b> The Telco dataset contains no real retention-offer '
                'assignment or outcomes. Treatment/control groups and offer responses are SIMULATED with documented assumptions; '
                'uplift and ROI figures are scenario estimates, not experimentally validated causal evidence.</div>',
                unsafe_allow_html=True)


def note(msg: str):
    st.markdown(f'<div class="note">{msg}</div>', unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading data from the database…", ttl=600)
def load_master() -> pd.DataFrame:
    df = pd.read_sql(text(QUERY), get_engine())
    df.columns = [str(c) for c in df.columns]
    return df


def require_data() -> pd.DataFrame:
    try:
        df = load_master()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load data from the database ({exc}). Run `python -m src.pipeline` after configuring PostgreSQL (see README).")
        st.stop()
    if df.empty:
        st.error("The analytical tables are empty. Run `python -m src.pipeline` first.")
        st.stop()
    return df


@st.cache_data(ttl=600)
def load_explanation(customer_id: str):
    with get_engine().connect() as c:
        r = c.execute(text("SELECT base_value, shap_json, feature_json FROM churn_explanations WHERE customer_id=:i"), {"i": customer_id}).fetchone()
    if r is None:
        return None
    return float(r[0]), json.loads(r[1]), json.loads(r[2])


@st.cache_data(ttl=600)
def load_json(name: str) -> dict:
    return json.loads((config.REPORTS_DIR / name).read_text())


@st.cache_data(ttl=600)
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(config.REPORTS_DIR / name)


@st.cache_resource
def load_cox():
    from src.survival_model import load_cox as _lc
    return _lc()


def money(x: float) -> str:
    return f"${x:,.0f}"


def sidebar_footer():
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Database backend: **{backend_name()}**")

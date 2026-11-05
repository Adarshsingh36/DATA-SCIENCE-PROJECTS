import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dashboard.utils import setup, require_data, load_explanation, money, note
from src import config

setup("Churn Intelligence", "🔎")
df = require_data()

col1, col2 = st.columns([1, 2])
q = col1.text_input("Search customer ID", value="", placeholder="e.g. 3668-QPYBK")
matches = df[df.customer_id.str.contains(q.strip(), case=False, regex=False)] if q.strip() else df.sort_values("churn_probability", ascending=False)
if matches.empty:
    st.warning("No customer matches that search."); st.stop()
cid = col2.selectbox("Customer", matches.customer_id.head(500).tolist(),
                     help="With no search text, customers are listed from highest to lowest churn probability.")
r = df[df.customer_id == cid].iloc[0]

g1, g2, g3 = st.columns([1.2, 1, 1])
with g1:
    fig = go.Figure(go.Indicator(mode="gauge+number", value=float(r.churn_probability) * 100, number={"suffix": "%"},
        title={"text": "Churn probability"}, gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#6366f1"},
        "steps": [{"range": [0, 25], "color": "#14532d"}, {"range": [25, 50], "color": "#78350f"}, {"range": [50, 100], "color": "#7f1d1d"}]}))
    fig.update_layout(height=260, margin=dict(t=50, b=10)); st.plotly_chart(fig, use_container_width=True)
with g2:
    st.metric("Risk category", r.risk_tier); st.metric("Actual status (label)", "Churned" if r.churn == 1 else "Retained")
    st.metric("Tenure", f"{int(r.tenure_months)} months")
with g3:
    st.metric("Monthly charges", f"${r.monthly_charges:,.2f}"); st.metric("Contract", r.contract)
    st.metric("Annual revenue at risk", money(r.revenue_at_risk_annual))

exp = load_explanation(cid)
st.subheader("Why this probability? (SHAP explanation)")
if exp is None:
    st.info("No explanation stored for this customer.")
else:
    base, shap_d, feat_d = exp
    s = pd.Series(shap_d); top = s.reindex(s.abs().sort_values(ascending=False).index).head(10)
    t = pd.DataFrame({"feature": top.index, "value": [feat_d[k] for k in top.index], "shap": top.values})
    t["effect"] = np.where(t.shap > 0, "raises churn risk", "lowers churn risk")
    fig = go.Figure(go.Bar(x=t.shap[::-1], y=[f"{f} = {v:g}" for f, v in zip(t.feature, t.value)][::-1], orientation="h",
                           marker_color=["#ef4444" if v > 0 else "#3b82f6" for v in t.shap[::-1]]))
    fig.update_layout(height=420, xaxis_title="contribution to churn log-odds (SHAP)", margin=dict(l=10, r=10))
    a, b = st.columns([3, 2])
    a.plotly_chart(fig, use_container_width=True)
    with b:
        st.markdown("**Top contributing factors**")
        st.dataframe(t.assign(shap=t.shap.round(3)), hide_index=True, use_container_width=True)
        p_expl = 1 / (1 + np.exp(-(base + s.sum())))
        st.caption(f"Explainer model: baseline log-odds {base:.2f} + Σ SHAP {s.sum():.2f} → {p_expl:.1%}. "
                   f"The headline probability ({r.churn_probability:.1%}) is the calibrated out-of-fold prediction, so the two differ slightly.")
note("SHAP shows which features the model <b>associates</b> with higher or lower predicted churn for this customer. "
    "It is not evidence that changing a feature would <b>cause</b> the customer to stay or leave.")

with st.expander("Global explanations"):
    for f in ["shap_global_importance.png", "shap_summary.png"]:
        p = config.FIGURES_DIR / f
        if p.exists(): st.image(str(p))
st.subheader("Highest-risk customers")
st.dataframe(df.sort_values("churn_probability", ascending=False)[["customer_id", "churn_probability", "risk_tier", "contract", "tenure_months",
             "monthly_charges", "revenue_at_risk_annual"]].head(25).round(3), hide_index=True, use_container_width=True)

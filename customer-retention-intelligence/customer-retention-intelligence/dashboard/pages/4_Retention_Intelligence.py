import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dashboard.utils import setup, require_data, load_json, sim_banner, note, money

setup("Retention Intelligence", "🎯")
sim_banner()
df = require_data()
up = load_json("uplift_metrics.json")
ev = up["evaluation_holdout"]

c = st.columns(5)
c[0].metric("Selected uplift learner", up["selected_learner"].replace("_", "-"))
c[1].metric("Customers with positive uplift", f"{up['share_positive_uplift']:.0%}")
c[2].metric("Mean estimated ITE", f"{up['mean_estimated_ite_selected']:+.3f}")
c[3].metric("Simulated ATE (treated − control)", f"{up['simulated_average_treatment_effect']:+.3f}")
c[4].metric("Hold-out Qini (selected)", f"{ev['qini'][up['selected_learner']]:.3f}")

st.subheader("Three different questions")
a, b, d = st.columns(3)
hi = df.risk_tier == "High"
a.metric("1. Likely to churn (High risk)", f"{hi.sum():,}")
b.metric("2. Likely to respond (positive uplift)", f"{int(df.positive_uplift.sum()):,}")
d.metric("3. Economically worthwhile (default offer)", f"{int(df.economically_worthwhile.sum()):,}",
         help="Default scenario: 10% discount, $10 offer cost per customer, 12-month benefit window, 40% margin.")
st.caption(f"High-risk customers who are NOT worthwhile targets: {int((hi & ~df.economically_worthwhile).sum()):,} "
           f"(lost causes, sleeping dogs or negative expected value). Discounting every high-risk customer would waste budget.")

tab1, tab2, tab3, tab4 = st.tabs(["Uplift vs risk", "Treatment / control", "Model validation", "Targeting list"])
with tab1:
    s = df.sample(min(3000, len(df)), random_state=1)
    fig = px.scatter(s, x="churn_probability", y="ite_selected", color="quadrant", opacity=0.6,
                     color_discrete_map={"Persuadable": "#10b981", "Sure thing": "#3b82f6", "Lost cause": "#9ca3af", "Sleeping dog": "#ef4444"},
                     labels={"churn_probability": "Baseline churn probability", "ite_selected": "Estimated incremental retention (ITE)"})
    fig.add_hline(y=0, line_dash="dash"); fig.update_layout(height=470); st.plotly_chart(fig, use_container_width=True)
    q = df.quadrant.value_counts().reset_index(); q.columns = ["quadrant", "customers"]
    st.dataframe(q, hide_index=True)
with tab2:
    t = df.groupby("sim_treatment").agg(customers=("sim_retained", "size"), retention_rate=("sim_retained", "mean")).reset_index()
    t["arm"] = t.sim_treatment.map({0: "Control", 1: "Treated (simulated offer)"})
    a, b = st.columns(2)
    a.plotly_chart(px.bar(t, x="arm", y="retention_rate", text=t.retention_rate.round(3), title="Retention rate by arm (simulated experiment)"), use_container_width=True)
    dim = b.selectbox("Break down by", ["contract", "monthly_charge_category", "tenure_group", "quadrant", "segment_label"])
    g = df.groupby([dim, "sim_treatment"]).sim_retained.mean().unstack()
    g["observed_uplift"] = g[1] - g[0]; g["mean_predicted_ite"] = df.groupby(dim).ite_selected.mean()
    b.dataframe(g.rename(columns={0: "control", 1: "treated"}).round(3), use_container_width=True)
    note("Control outcomes are the REAL observed retention labels; treated outcomes are simulated. See README for the exact assumptions.")
with tab3:
    qd = pd.DataFrame({"model": list(ev["qini"]), "Qini": list(ev["qini"].values()), "AUUC": list(ev["auuc"].values())})
    if "spearman_vs_reference_effect" in ev: qd["Spearman vs known simulated effect"] = list(ev["spearman_vs_reference_effect"].values())
    st.dataframe(qd.round(3), hide_index=True, use_container_width=True)
    dec = pd.DataFrame(up["uplift_deciles_holdout_selected"])
    fig = go.Figure()
    fig.add_bar(x=dec.decile, y=dec.observed_uplift, name="Observed uplift (hold-out, simulated)")
    fig.add_scatter(x=dec.decile, y=dec.mean_predicted_ite, name="Mean predicted ITE", mode="lines+markers")
    fig.update_layout(title="Uplift by predicted-ITE decile (1 = highest predicted uplift)", xaxis_title="decile"); st.plotly_chart(fig, use_container_width=True)
    note("Because the simulation makes would-be churners the only ones who can be saved, a plain churn-risk ranking also scores well here. "
         "The simulated truth was written by us, so recovering it demonstrates the pipeline works; it does not prove real customers behave this way.")
with tab4:
    f1, f2, f3 = st.columns(3)
    tiers = f1.multiselect("Risk tier", ["High", "Medium", "Low"], default=["High", "Medium"])
    quads = f2.multiselect("Quadrant", sorted(df.quadrant.unique()), default=["Persuadable"])
    only = f3.checkbox("Economically worthwhile only", value=True)
    m = df[df.risk_tier.isin(tiers) & df.quadrant.isin(quads)]
    if only: m = m[m.economically_worthwhile]
    st.write(f"**{len(m):,} customers** · expected net value at default offer: {money(m.expected_net_value.sum())}")
    st.dataframe(m.sort_values("expected_net_value", ascending=False)[["customer_id", "churn_probability", "risk_tier", "ite_selected", "quadrant",
                 "expected_net_value", "monthly_charges", "contract", "segment_label"]].head(300).round(3), hide_index=True, use_container_width=True)

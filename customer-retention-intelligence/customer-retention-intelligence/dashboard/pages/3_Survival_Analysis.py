import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dashboard.utils import setup, require_data, load_csv, load_json, load_cox, note
from src import survival_model as sm

setup("Survival Analysis", "⏳")
df = require_data()
note("Duration = tenure in months; event = churn. Customers still active are <b>right-censored</b>. "
     "The data are a single snapshot, so survival is expressed in months of tenure; calendar departure dates cannot be predicted.")

st.subheader("Kaplan-Meier survival curves")
group = st.selectbox("Compare by", ["All customers", "contract", "internet_service", "payment_method", "tenure_group", "senior_citizen", "paperless_billing"])
fig = go.Figure()
if group == "All customers":
    groups = {"All customers": sm.fit_kaplan_meier(df)}
else:
    groups = sm.km_by_group(df, group)
for name, km in groups.items():
    sf, ci = km.survival_function_, km.confidence_interval_
    x = sf.index.values
    fig.add_trace(go.Scatter(x=x, y=sf.iloc[:, 0], mode="lines", name=name, line_shape="hv"))
    fig.add_trace(go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[ci.iloc[:, 1], ci.iloc[::-1, 0]], fill="toself", opacity=0.12,
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
fig.update_layout(xaxis_title="Tenure (months)", yaxis_title="P(still a customer)", yaxis_range=[0, 1.02], height=430)
st.plotly_chart(fig, use_container_width=True)
if group != "All customers":
    lr = sm.logrank_by_group(df, group)
    st.caption(f"Multivariate log-rank test: χ² = {lr['test_statistic']:.1f}, df = {lr['df']}, p = {lr['p_value']:.2g}")
st.dataframe(sm.km_summary(groups[list(groups)[0]]).round(3).rename(columns={"months": "tenure (months)"}), hide_index=True)

st.subheader("Cox proportional hazards: hazard ratios")
which = st.radio("Model", ["Stratified by contract (used for predictions)", "Contract as covariate (shows contract hazard ratios)"], horizontal=True)
fn = "cox_hazard_ratios_stratified.csv" if which.startswith("Strat") else "cox_hazard_ratios_with_contract.csv"
pf = "cox_ph_test_stratified.csv" if which.startswith("Strat") else "cox_ph_test_with_contract.csv"
hr, ph = load_csv(fn), load_csv(pf)
h = hr.sort_values("hazard_ratio")
fig = go.Figure(go.Scatter(x=h.hazard_ratio, y=h.covariate, mode="markers", marker=dict(size=9, color="#6366f1"),
    error_x=dict(type="data", symmetric=False, array=h.hr_upper_95 - h.hazard_ratio, arrayminus=h.hazard_ratio - h.hr_lower_95)))
fig.add_vline(x=1, line_dash="dash", line_color="#9ca3af"); fig.update_xaxes(type="log", title="Hazard ratio (log scale; >1 = higher churn hazard)")
fig.update_layout(height=560); st.plotly_chart(fig, use_container_width=True)
sj = load_json("survival_metrics.json")
c = st.columns(3)
c[0].metric("Events (churned)", f"{sj['n_events']:,}"); c[1].metric("Censored (active)", f"{sj['n_censored']:,}")
c[2].metric("Concordance (selected model)", f"{(sj['cox_concordance'] if which.startswith('Strat') else sj['cox_concordance_with_contract_dummies']):.3f}")
viol = ph[ph.violates_ph_at_5pct]
if len(viol):
    st.warning(f"Proportional-hazards assumption is questionable for {len(viol)} covariate(s) at the 5% level: "
               f"{', '.join(viol.covariate)}. Hazard ratios for these are time-averaged effects; interpret with care.")
with st.expander("Proportional-hazards test (Schoenfeld residuals, rank time)"):
    st.dataframe(ph.round(4), hide_index=True, use_container_width=True)

st.subheader("Individual survival and expected retention")
cid = st.selectbox("Customer", df.sort_values("churn_probability", ascending=False).customer_id.head(500).tolist(), key="surv_c")
r = df[df.customer_id == cid]
cox = load_cox()
curve = sm.customer_curve(cox, r, stratified=True, horizon=36)
fig = px.line(curve, x="months_from_now", y="survival_probability", title="Probability customer is still active, by months from now")
fig.update_layout(yaxis_range=[0, 1.02], height=360); st.plotly_chart(fig, use_container_width=True)
x = r.iloc[0]
c = st.columns(4)
c[0].metric("Survives next 12 mo", f"{x.surv_prob_12m:.1%}"); c[1].metric("Survives next 24 mo", f"{x.surv_prob_24m:.1%}")
c[2].metric("Survives next 36 mo", f"{x.surv_prob_36m:.1%}"); c[3].metric("Expected active months (next 24)", f"{x.expected_months_next_24:.1f}")
if x.extrapolated:
    st.info("This customer's tenure + 24 months exceeds the longest observed tenure (72 months); the baseline hazard is held flat beyond that point, so long-horizon estimates are optimistic.")
st.caption("Expected active months = restricted mean survival over the next 24 months (Σ of monthly conditional survival probabilities).")

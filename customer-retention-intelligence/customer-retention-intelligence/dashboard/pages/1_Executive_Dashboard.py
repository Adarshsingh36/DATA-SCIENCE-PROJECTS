import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import plotly.express as px
import streamlit as st
from dashboard.utils import setup, require_data, money, note

setup("Executive Dashboard", "📊")
df = require_data()

c = st.columns(6)
c[0].metric("Total customers", f"{len(df):,}")
c[1].metric("Actual churn rate", f"{df.churn.mean():.1%}")
c[2].metric("Predicted churn rate", f"{df.churn_probability.mean():.1%}", help="Mean calibrated out-of-fold churn probability.")
c[3].metric("High-risk customers", f"{(df.risk_tier == 'High').sum():,}", help="Churn probability ≥ 50%.")
c[4].metric("Annual revenue at risk", money(df.revenue_at_risk_annual.sum()), help="Σ churn probability × monthly charges × 12.")
c[5].metric("Mean CLV (36 mo)", money(df.clv.mean()), help="Survival-based contribution-margin CLV; see README for assumptions.")
note("Predicted probabilities are <b>out-of-fold</b>: each customer is scored by a model that never saw their label.")

a, b, d = st.columns(3)
with a:
    ch = df.churn.map({0: "Retained", 1: "Churned"}).value_counts().reset_index()
    ch.columns = ["status", "customers"]
    st.plotly_chart(px.pie(ch, names="status", values="customers", hole=0.55, title="Actual churn distribution",
                           color="status", color_discrete_map={"Retained": "#3b82f6", "Churned": "#ef4444"}), use_container_width=True)
with b:
    rt = df.risk_tier.value_counts().reindex(["Low", "Medium", "High"]).reset_index(); rt.columns = ["risk_tier", "customers"]
    st.plotly_chart(px.bar(rt, x="risk_tier", y="customers", color="risk_tier", title="Customers by predicted risk tier",
                           color_discrete_map={"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"}), use_container_width=True)
with d:
    st.plotly_chart(px.histogram(df, x="churn_probability", nbins=40, title="Distribution of churn probability",
                                 color_discrete_sequence=["#6366f1"]), use_container_width=True)

a, b = st.columns(2)
with a:
    g = df.groupby("contract").agg(actual=("churn", "mean"), predicted=("churn_probability", "mean")).reset_index().melt("contract")
    st.plotly_chart(px.bar(g, x="contract", y="value", color="variable", barmode="group", title="Actual vs predicted churn by contract",
                           labels={"value": "churn rate"}), use_container_width=True)
with b:
    st.plotly_chart(px.histogram(df, x="clv", nbins=40, title="Customer lifetime value distribution (36-month, contribution margin)",
                                 color_discrete_sequence=["#10b981"]), use_container_width=True)

a, b = st.columns(2)
with a:
    t = df.groupby("tenure_group").agg(actual=("churn", "mean"), customers=("churn", "size")).reset_index()
    st.plotly_chart(px.bar(t, x="tenure_group", y="actual", text=t.customers, title="Churn rate by tenure group",
                           category_orders={"tenure_group": ["0-12m", "13-24m", "25-48m", "49-72m"]}), use_container_width=True)
with b:
    p = df.groupby("payment_method").churn.mean().sort_values().reset_index()
    st.plotly_chart(px.bar(p, x="churn", y="payment_method", orientation="h", title="Churn rate by payment method"), use_container_width=True)

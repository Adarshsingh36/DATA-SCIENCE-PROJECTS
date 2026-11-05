import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from dashboard.utils import setup, require_data, load_csv, load_json, note

setup("Customer Segmentation", "🧩")
df = require_data()
sj = load_json("segmentation_metrics.json")
st.caption(f"K-Means with k = {sj['k']} (silhouette = {sj['silhouette']:.3f}). Segment names are derived from measured cluster profiles relative to the whole base.")

ks = load_csv("segment_k_scores.csv")
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=ks.k, y=ks.inertia, name="Inertia (elbow)", mode="lines+markers"), secondary_y=False)
fig.add_trace(go.Scatter(x=ks.k, y=ks.silhouette, name="Silhouette", mode="lines+markers"), secondary_y=True)
fig.update_layout(height=300, title="Choosing k", xaxis_title="k"); st.plotly_chart(fig, use_container_width=True)

a, b = st.columns([3, 2])
s = df.sample(min(4000, len(df)), random_state=2)
a.plotly_chart(px.scatter(s, x="pc1", y="pc2", color="segment_label", opacity=0.6, title="Segments in PCA space").update_layout(height=470), use_container_width=True)
sz = df.segment_label.value_counts().reset_index(); sz.columns = ["segment", "customers"]
b.plotly_chart(px.pie(sz, names="segment", values="customers", hole=0.5, title="Segment sizes").update_layout(height=470), use_container_width=True)

st.subheader("Segment profiles")
prof = df.groupby("segment_label").agg(customers=("churn", "size"), tenure_months=("tenure_months", "mean"), monthly_charges=("monthly_charges", "mean"),
        mean_clv=("clv", "mean"), total_clv=("clv", "sum"), services=("service_count", "mean"), predicted_churn=("churn_probability", "mean"),
        actual_churn=("churn", "mean"), survival_12m=("surv_prob_12m", "mean"), revenue_at_risk=("revenue_at_risk_annual", "sum"))
st.dataframe(prof.round(2), use_container_width=True)
a, b = st.columns(2)
a.plotly_chart(px.box(df, x="segment_label", y="clv", color="segment_label", title="CLV distribution by segment").update_layout(showlegend=False), use_container_width=True)
c = df.groupby("segment_label").agg(predicted=("churn_probability", "mean"), actual=("churn", "mean")).reset_index().melt("segment_label")
b.plotly_chart(px.bar(c, x="segment_label", y="value", color="variable", barmode="group", title="Churn risk by segment", labels={"value": "churn rate"}), use_container_width=True)
note("Labels are interpretive summaries of cluster averages, not fixed customer types; individual members vary.")

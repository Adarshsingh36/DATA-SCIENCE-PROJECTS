import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import text
from dashboard.utils import setup, require_data, sim_banner, money, note
from database.connection import get_engine
from src import roi, config

setup("ROI Simulator", "💰")
sim_banner()
df = require_data()
frame = df[["customer_id", "monthly_charges", "churn_probability", "ite_selected"]]

st.sidebar.header("Campaign parameters")
strategy = st.sidebar.selectbox("Targeting strategy", roi.STRATEGIES)
discount = st.sidebar.slider("Discount percentage", 0.0, 40.0, 10.0, 1.0, help="Effect is calibrated at 10%; other depths use a diminishing-returns assumption.")
budget = st.sidebar.number_input("Campaign budget ($)", 1000.0, 5_000_000.0, 20000.0, 1000.0)
n_target = st.sidebar.number_input("Number of customers to target (max)", 1, len(df), 1000, 50)
offer_cost = st.sidebar.number_input("Retention offer cost per customer ($)", 0.0, 500.0, 10.0, 1.0, help="Contact/handling cost, excluding the discount itself.")
duration = st.sidebar.slider("Expected retention duration (months)", 1, 36, 12, help="Months of margin an additionally retained customer is assumed to generate.")
margin = st.sidebar.slider("Contribution margin", 0.05, 1.0, float(config.CONTRIBUTION_MARGIN), 0.05)

r = roi.simulate_campaign(frame, discount, budget, int(n_target), offer_cost, float(duration), strategy, margin)
if r["n_targeted"] == 0:
    st.warning("No customers qualify under these settings (budget too small, or no positive-value customers for this strategy)."); 
c = st.columns(5)
c[0].metric("Customers targeted", f"{r['n_targeted']:,}"); c[1].metric("Baseline churn prob. (targeted)", f"{r['baseline_churn_prob']:.1%}")
c[2].metric("Incremental retention prob.", f"{r['incremental_retention_prob']:+.1%}"); c[3].metric("Expected customers retained", f"{r['expected_customers_retained']:.1f}")
c[4].metric("Expected retained revenue", money(r["expected_retained_revenue"]))
c = st.columns(5)
c[0].metric("Incremental benefit (margin)", money(r["incremental_benefit"])); c[1].metric("Discount cost", money(r["discount_cost"]))
c[2].metric("Campaign cost (offer + discount)", money(r["campaign_cost"])); c[3].metric("Net incremental benefit", money(r["net_incremental_benefit"]))
c[4].metric("ROI", "n/a" if np.isnan(r["roi_pct"]) else f"{r['roi_pct']:.1f}%")
st.latex(r"ROI = \frac{\text{Incremental Benefit} - \text{Campaign Cost}}{\text{Campaign Cost}} \times 100")
note("<b>Scenario estimates only.</b> Incremental retention comes from uplift models trained on SIMULATED treatment/control data, so these numbers "
     "illustrate the decision framework and are not forecasts. Discount cost is paid to every targeted customer who stays, including those who would have stayed anyway.")

a, b = st.columns(2)
fig = go.Figure(go.Waterfall(x=["Incremental benefit", "Discount cost", "Fixed offer cost", "Net benefit"], measure=["relative", "relative", "relative", "total"],
                             y=[r["incremental_benefit"], -r["discount_cost"], -r["fixed_offer_cost"], 0]))
fig.update_layout(title="Benefit-to-net waterfall", height=380); a.plotly_chart(fig, use_container_width=True)
rows = []
for s in roi.STRATEGIES:
    x = roi.simulate_campaign(frame, discount, budget, int(n_target), offer_cost, float(duration), s, margin)
    rows.append({"strategy": s, "targeted": x["n_targeted"], "retained (expected)": round(x["expected_customers_retained"], 1),
                 "campaign cost": round(x["campaign_cost"]), "net benefit": round(x["net_incremental_benefit"]), "ROI %": None if np.isnan(x["roi_pct"]) else round(x["roi_pct"], 1)})
b.markdown("**Strategy comparison (same parameters)**"); b.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

st.subheader("Sensitivity: ROI and net benefit vs discount depth")
sens = []
for d in np.arange(2, 41, 2):
    x = roi.simulate_campaign(frame, float(d), budget, int(n_target), offer_cost, float(duration), strategy, margin)
    sens.append({"discount %": d, "ROI %": x["roi_pct"], "Net benefit": x["net_incremental_benefit"], "Targeted": x["n_targeted"]})
sens = pd.DataFrame(sens)
fig = px.line(sens, x="discount %", y=["ROI %", "Net benefit"], markers=True); st.plotly_chart(fig, use_container_width=True)

with st.expander("Targeted customers"):
    st.dataframe(r["_selected_econ"].round(3), hide_index=True, use_container_width=True)
if st.button("Save this scenario to the database"):
    eng = get_engine()
    with eng.begin() as conn:
        cid = conn.execute(text("INSERT INTO retention_campaigns (name,strategy,discount_pct,budget,n_targeted_requested,offer_cost_per_customer,retention_duration_months) "
                                "VALUES (:n,:s,:d,:b,:t,:o,:du) RETURNING campaign_id"),
                           {"n": "Dashboard scenario", "s": strategy, "d": discount, "b": budget, "t": int(n_target), "o": offer_cost, "du": float(duration)}).scalar()
        conn.execute(text("INSERT INTO retention_results (campaign_id,n_targeted,baseline_churn_prob,incremental_retention_prob,expected_customers_retained,expected_retained_revenue,"
                          "incremental_benefit,discount_cost,campaign_cost,net_incremental_benefit,roi_pct) VALUES (:c,:n,:b,:i,:e,:rr,:ib,:dc,:cc,:nb,:roi)"),
                     {"c": cid, "n": r["n_targeted"], "b": r["baseline_churn_prob"], "i": r["incremental_retention_prob"], "e": r["expected_customers_retained"],
                      "rr": r["expected_retained_revenue"], "ib": r["incremental_benefit"], "dc": r["discount_cost"], "cc": r["campaign_cost"],
                      "nb": r["net_incremental_benefit"], "roi": None if np.isnan(r["roi_pct"]) else r["roi_pct"]})
    st.success(f"Saved as campaign #{cid} (flagged as simulated-uplift scenario).")

"""Streamlit entry point:  streamlit run dashboard/app.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st
from dashboard.utils import sidebar_footer

st.set_page_config(page_title="Customer Retention Intelligence", page_icon="📡", layout="wide")
pages = [
    st.Page("pages/1_Executive_Dashboard.py", title="Executive Dashboard", icon="📊", default=True),
    st.Page("pages/2_Churn_Intelligence.py", title="Churn Intelligence", icon="🔎"),
    st.Page("pages/3_Survival_Analysis.py", title="Survival Analysis", icon="⏳"),
    st.Page("pages/4_Retention_Intelligence.py", title="Retention Intelligence", icon="🎯"),
    st.Page("pages/5_Customer_Segmentation.py", title="Customer Segmentation", icon="🧩"),
    st.Page("pages/6_ROI_Simulator.py", title="ROI Simulator", icon="💰"),
]
st.navigation(pages).run()
sidebar_footer()

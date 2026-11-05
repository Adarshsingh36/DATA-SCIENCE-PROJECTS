"""Exploratory data analysis: saves figures and a summary JSON."""
from __future__ import annotations
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src import config

sns.set_theme(style="whitegrid")
F = config.FIGURES_DIR


def _save(name):
    plt.tight_layout(); plt.savefig(F / f"eda_{name}.png", dpi=130); plt.close()


def run_eda(df: pd.DataFrame) -> dict:
    lab = df.assign(Churn=df["churn"].map({0: "Retained", 1: "Churned"}))
    pal = {"Retained": "#3b82f6", "Churned": "#ef4444"}
    lab["Churn"].value_counts().plot.bar(color=["#3b82f6", "#ef4444"]); plt.title("Churn distribution"); plt.ylabel("customers"); _save("churn_distribution")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, c, t in zip(ax, ["tenure_months", "monthly_charges", "total_charges"], ["Tenure (months)", "Monthly charges ($)", "Total charges ($)"]):
        sns.histplot(data=lab, x=c, hue="Churn", palette=pal, bins=30, element="step", ax=a); a.set_title(t)
    _save("numeric_distributions")
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    for a, c in zip(ax.ravel(), ["contract", "payment_method", "internet_service", "tenure_group", "monthly_charge_category", "gender"]):
        r = df.groupby(c)["churn"].mean().sort_values(ascending=False)
        sns.barplot(x=r.values, y=r.index, ax=a, color="#6366f1"); a.set_title(f"Churn rate by {c}"); a.set_xlabel("churn rate"); a.set_ylabel("")
    _save("churn_rate_by_category")
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
    for a, c in zip(ax, ["senior_citizen", "partner", "dependents", "paperless_billing"]):
        r = df.groupby(c)["churn"].mean(); sns.barplot(x=["No", "Yes"], y=r.values, ax=a, color="#10b981"); a.set_title(c); a.set_ylabel("churn rate")
    _save("demographics")
    addons = ["online_security", "online_backup", "device_protection", "tech_support", "streaming_tv", "streaming_movies"]
    r = pd.DataFrame({a: df[df.has_internet == 1].groupby(a)["churn"].mean() for a in addons}).T
    r.columns = ["Not subscribed", "Subscribed"]; r.plot.barh(figsize=(8, 4.5)); plt.title("Churn rate by add-on (internet customers)"); _save("addons")
    plt.figure(figsize=(8, 5)); sns.boxplot(data=lab, x="contract", y="monthly_charges", hue="Churn", palette=pal); plt.title("Monthly charges by contract and churn"); _save("charges_by_contract")
    num = ["tenure_months", "monthly_charges", "total_charges", "service_count", "avg_monthly_spend", "engagement_proxy", "churn"]
    plt.figure(figsize=(7, 6)); sns.heatmap(df[num].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0); plt.title("Correlation (numeric + proxies)"); _save("correlation")
    s = {"n": len(df), "churn_rate": float(df.churn.mean()), "mean_tenure": float(df.tenure_months.mean()),
         "mean_monthly_charges": float(df.monthly_charges.mean()), "total_monthly_revenue": float(df.monthly_charges.sum()),
         "churn_by_contract": df.groupby("contract")["churn"].mean().round(4).to_dict(),
         "churn_by_payment": df.groupby("payment_method")["churn"].mean().round(4).to_dict(),
         "churn_by_internet": df.groupby("internet_service")["churn"].mean().round(4).to_dict(),
         "churn_by_tenure_group": df.groupby("tenure_group")["churn"].mean().round(4).to_dict(),
         "churned_monthly_revenue": float(df.loc[df.churn == 1, "monthly_charges"].sum())}
    (config.REPORTS_DIR / "eda_summary.json").write_text(json.dumps(s, indent=2))
    return s

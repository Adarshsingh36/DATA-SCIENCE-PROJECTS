"""Transparent Customer Lifetime Value.

CLV_i = sum_{k=1..H} S_i(k) * (monthly_charges_i * margin) / (1 + r_m)^k

 * S_i(k): Cox-model probability that customer i is still active k months from now, given current tenure.
 * margin: contribution-margin share of revenue (ASSUMPTION, default 40%, config.CONTRIBUTION_MARGIN).
 * r_m   : monthly discount rate from the annual rate (ASSUMPTION, default 10%).
 * H     : horizon in months (ASSUMPTION, default 36), kept short because tenure is observed only up to 72 months.
Assumed: constant monthly charges over the horizon (no price changes, upsell, or usage growth).
IBM's precomputed CLTV column is NOT used.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from src import config


def monthly_rate(annual_rate: float) -> float:
    return (1.0 + annual_rate) ** (1.0 / 12.0) - 1.0


def compute_clv(monthly_charges, surv_matrix, margin: float = config.CONTRIBUTION_MARGIN,
                annual_rate: float = config.ANNUAL_DISCOUNT_RATE) -> np.ndarray:
    mc = np.asarray(monthly_charges, dtype=float)
    S = np.asarray(surv_matrix, dtype=float)
    H = S.shape[1]
    disc = (1.0 + monthly_rate(annual_rate)) ** -np.arange(1, H + 1)
    return (S * disc).sum(axis=1) * mc * margin


def clv_constant_hazard(monthly_margin: float, monthly_churn: float, horizon: int, annual_rate: float) -> float:
    """Closed-form reference with constant monthly churn: used to sanity-check compute_clv in tests."""
    d = 1.0 / (1.0 + monthly_rate(annual_rate))
    return float(sum(monthly_margin * ((1 - monthly_churn) ** k) * d ** k for k in range(1, horizon + 1)))


def clv_table(df: pd.DataFrame, surv_matrix, churn_prob) -> pd.DataFrame:
    clv = compute_clv(df["monthly_charges"], surv_matrix)
    return pd.DataFrame({
        "customer_id": df["customer_id"].values, "clv": clv,
        "monthly_margin": df["monthly_charges"].values * config.CONTRIBUTION_MARGIN,
        "revenue_at_risk_annual": np.asarray(churn_prob) * df["monthly_charges"].values * 12.0,
    })

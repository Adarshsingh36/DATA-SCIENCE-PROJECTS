"""Retention-campaign economics. Pure functions (used by the dashboard simulator, pipeline and tests).

Per targeted customer i, with baseline churn p0, predicted incremental retention ITE (at the chosen discount),
monthly charge c, benefit window D months, contribution margin m, discount d (fraction), fixed offer cost f:

  incremental retained revenue = ITE * c * D
  incremental benefit          = incremental retained revenue * m
  discount cost                = d * c * D * P(stay | treated)    (discount is paid to EVERY customer who stays,
                                                                   including those who would have stayed anyway)
  campaign cost                = f + discount cost
  net incremental benefit      = incremental benefit - campaign cost
  ROI %                        = (incremental benefit - campaign cost) / campaign cost * 100

All uplift inputs come from a SIMULATED experiment -> results are scenario estimates, not validated outcomes.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from src import config

STRATEGIES = ["Uplift-ranked, economically worthwhile only", "Uplift-ranked (ignore economics)",
              "Highest churn risk", "Random"]


def response_multiplier(discount_pct: float, ref_pct: float = config.SIM_REFERENCE_DISCOUNT_PCT) -> float:
    """Diminishing-returns response to discount depth (ASSUMPTION): 0 at 0%, 1 at the reference discount."""
    return float((1 - np.exp(-discount_pct / 10.0)) / (1 - np.exp(-ref_pct / 10.0)))


def customer_economics(monthly_charges, p0, ite_ref, discount_pct, offer_cost, duration_months,
                       margin: float = config.CONTRIBUTION_MARGIN) -> pd.DataFrame:
    c = np.asarray(monthly_charges, float); p0 = np.asarray(p0, float)
    ite = np.asarray(ite_ref, float) * response_multiplier(discount_pct)
    ite = np.clip(ite, -(1 - p0), p0)                      # incremental retention can't exceed baseline churn
    d = discount_pct / 100.0
    p_stay_treated = 1 - p0 + ite
    inc_rev = ite * c * duration_months
    benefit = inc_rev * margin
    disc_cost = d * c * duration_months * p_stay_treated
    camp_cost = offer_cost + disc_cost
    return pd.DataFrame({"ite": ite, "incremental_revenue": inc_rev, "incremental_benefit": benefit,
                         "discount_cost": disc_cost, "campaign_cost": camp_cost,
                         "net_benefit": benefit - camp_cost, "baseline_churn_prob": p0})


def select_customers(econ: pd.DataFrame, strategy: str, n_target: int, budget: float, seed: int = 0) -> np.ndarray:
    n = len(econ)
    if strategy == STRATEGIES[0]:
        order = np.argsort(-econ["net_benefit"].values); order = order[econ["net_benefit"].values[order] > 0]
    elif strategy == STRATEGIES[1]:
        order = np.argsort(-econ["ite"].values); order = order[econ["ite"].values[order] > 0]
    elif strategy == STRATEGIES[2]:
        order = np.argsort(-econ["baseline_churn_prob"].values)
    elif strategy == STRATEGIES[3]:
        order = np.random.default_rng(seed).permutation(n)
    else:
        raise ValueError(f"Unknown strategy {strategy}")
    order = order[:max(int(n_target), 0)]
    cum = np.cumsum(econ["campaign_cost"].values[order])
    return order[cum <= budget]


def simulate_campaign(frame: pd.DataFrame, discount_pct: float, budget: float, n_target: int, offer_cost: float,
                      duration_months: float, strategy: str = STRATEGIES[0],
                      margin: float = config.CONTRIBUTION_MARGIN, seed: int = 0) -> dict:
    """frame needs: customer_id, monthly_charges, churn_probability, ite_selected (effect at the reference discount)."""
    econ = customer_economics(frame["monthly_charges"], frame["churn_probability"], frame["ite_selected"],
                              discount_pct, offer_cost, duration_months, margin)
    idx = select_customers(econ, strategy, n_target, budget, seed)
    sel = econ.iloc[idx]
    n = len(sel)
    camp_cost = float(sel["campaign_cost"].sum())
    benefit = float(sel["incremental_benefit"].sum())
    res = {
        "strategy": strategy, "n_targeted": int(n),
        "baseline_churn_prob": float(sel["baseline_churn_prob"].mean()) if n else 0.0,
        "incremental_retention_prob": float(sel["ite"].mean()) if n else 0.0,
        "expected_customers_retained": float(sel["ite"].sum()),
        "expected_retained_revenue": float(sel["incremental_revenue"].sum()),
        "incremental_benefit": benefit, "discount_cost": float(sel["discount_cost"].sum()),
        "campaign_cost": camp_cost, "fixed_offer_cost": float(offer_cost * n),
        "net_incremental_benefit": benefit - camp_cost,
        "roi_pct": float((benefit - camp_cost) / camp_cost * 100.0) if camp_cost > 0 else float("nan"),
        "is_simulated_uplift": True,
    }
    res["_selected_ids"] = frame["customer_id"].values[idx]
    res["_selected_econ"] = sel.assign(customer_id=frame["customer_id"].values[idx])
    return res

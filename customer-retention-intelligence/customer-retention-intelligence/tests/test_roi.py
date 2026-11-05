import numpy as np
import pandas as pd
import pytest
from src import roi


def frame(n=500, seed=0):
    r = np.random.default_rng(seed)
    p0 = r.uniform(0.05, 0.9, n)
    return pd.DataFrame({"customer_id": [f"c{i}" for i in range(n)], "monthly_charges": r.uniform(20, 110, n),
                         "churn_probability": p0, "ite_selected": p0 * r.uniform(0, 0.4, n)})


def test_roi_formula_hand_calc():
    f = pd.DataFrame({"customer_id": ["a"], "monthly_charges": [100.0], "churn_probability": [0.5], "ite_selected": [0.2]})
    r = roi.simulate_campaign(f, 10.0, 1e6, 10, 5.0, 12.0, roi.STRATEGIES[1], margin=0.4)
    benefit = 0.2 * 100 * 12 * 0.4                      # 96
    disc = 0.10 * 100 * 12 * (1 - 0.5 + 0.2)            # 84
    cost = 5 + disc
    assert r["incremental_benefit"] == pytest.approx(benefit) and r["discount_cost"] == pytest.approx(disc)
    assert r["campaign_cost"] == pytest.approx(cost) and r["net_incremental_benefit"] == pytest.approx(benefit - cost)
    assert r["roi_pct"] == pytest.approx((benefit - cost) / cost * 100)
    assert r["expected_customers_retained"] == pytest.approx(0.2)


def test_budget_and_target_limits():
    f = frame()
    for s in roi.STRATEGIES:
        r = roi.simulate_campaign(f, 10, 3000, 40, 5, 12, s)
        assert r["campaign_cost"] <= 3000 + 1e-6 and r["n_targeted"] <= 40


def test_zero_budget_no_targets_nan_roi():
    r = roi.simulate_campaign(frame(), 10, 0, 100, 5, 12)
    assert r["n_targeted"] == 0 and np.isnan(r["roi_pct"])


def test_worthwhile_strategy_never_negative_net_per_customer():
    r = roi.simulate_campaign(frame(), 10, 1e9, 10000, 5, 12, roi.STRATEGIES[0])
    assert (r["_selected_econ"].net_benefit > 0).all() and r["net_incremental_benefit"] >= 0


def test_not_all_high_risk_customers_are_saved():
    f = frame(); r = roi.simulate_campaign(f, 10, 1e9, 10000, 5, 12, roi.STRATEGIES[2])
    assert r["expected_customers_retained"] < r["n_targeted"] * r["baseline_churn_prob"]


def test_response_multiplier():
    assert roi.response_multiplier(0) == 0 and roi.response_multiplier(10) == pytest.approx(1)
    assert roi.response_multiplier(20) > 1 and roi.response_multiplier(20) < 2        # diminishing returns


def test_zero_discount_has_no_discount_cost_or_effect():
    r = roi.simulate_campaign(frame(), 0, 1e9, 100, 5, 12, roi.STRATEGIES[1])
    assert r["discount_cost"] == 0 and r["expected_customers_retained"] == pytest.approx(0)


def test_ite_clipped_to_baseline():
    e = roi.customer_economics([50], [0.1], [0.9], 10, 5, 12)
    assert e.ite.iloc[0] == pytest.approx(0.1)


def test_uplift_ranking_beats_random():
    f = frame(2000, 3)
    u = roi.simulate_campaign(f, 10, 1e9, 300, 5, 12, roi.STRATEGIES[1])
    rnd = roi.simulate_campaign(f, 10, 1e9, 300, 5, 12, roi.STRATEGIES[3])
    assert u["expected_customers_retained"] > rnd["expected_customers_retained"]

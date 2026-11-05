import numpy as np

from walmartfc.inventory import base_stock_plan, simulate, solve_store


def _problem(n=3, H=6, K=3, level=10.0):
    mid = np.full((n, H), level)
    D = np.stack([mid * 0.8, mid, mid * 1.2], axis=2)
    return D, np.array([0.25, 0.5, 0.25])


def test_solver_orders_enough_to_avoid_stockouts_when_penalty_high():
    D, w = _problem()
    opening = np.full(3, 10.0)
    Q, Z, st, obj, _ = solve_store(D, w, opening, cap=1e6, L=1, order_cost=0.1, hold=0.001, penalty=10.0, time_limit_s=10)
    assert st in ("OPTIMAL", "FEASIBLE")
    r = simulate(Q * 1000, D[:, :, 1] * 1000, opening * 1000, 1, 100.0, 0.001, 10.0)
    # lead time 1 and opening = 1 week: only the final week's order cannot arrive; fill rate should be high
    assert r["fill_rate"] > 0.95


def test_capacity_constraint_respected():
    D, w = _problem(level=20.0)
    opening = np.full(3, 5.0)
    cap = 40.0
    Q, Z, st, *_ = solve_store(D, w, opening, cap=cap, L=1, order_cost=0.0, hold=0.01, penalty=1.0, time_limit_s=10)
    inv = opening.copy()
    for t in range(D.shape[1]):
        inv = inv + (Q[:, t - 1] if t >= 1 else 0)
        inv = np.maximum(inv - D[:, t, 2], 0)            # worst-case (high) path
        assert inv.sum() <= cap + 1e-6


def test_high_ordering_cost_reduces_number_of_orders():
    D, w = _problem()
    opening = np.full(3, 10.0)
    cheap = solve_store(D, w, opening, 1e6, 1, 0.0, 0.05, 5.0, 10)[1].sum()
    dear = solve_store(D, w, opening, 1e6, 1, 50.0, 0.05, 5.0, 10)[1].sum()
    assert dear <= cheap


def test_no_orders_after_horizon_minus_lead_time():
    D, w = _problem(H=5)
    Q, *_ = solve_store(D, w, np.full(3, 10.0), 1e6, 2, 0.1, 0.01, 5.0, 10)
    assert Q[:, 3:].sum() == 0


def test_simulation_accounting():
    r = simulate(np.zeros((1, 2)), np.array([[10.0, 10.0]]), np.array([15.0]), 1, 0.0, 0.1, 1.0)
    assert r["stockout"] == 5.0 and abs(r["holding"] - 0.5) < 1e-9 and r["fill_rate"] == 0.75


def test_base_stock_plan_nonnegative():
    mid = np.full((2, 6), 10.0)
    assert (base_stock_plan(mid, mid * 0.3, np.full(2, 10.0), 1) >= 0).all()

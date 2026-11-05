"""Store-department inventory optimisation with OR-Tools (SCIP mixed-integer programme).

IMPORTANT - what is and is not data here
  * Demand forecasts (p50 and the calibrated 80% interval ends) come from the reconciled forecasting pipeline.
  * Everything else (opening inventory, lead time, ordering cost, holding cost, stockout penalty, storage capacity)
    is a *configurable scenario assumption* in ``config/default.yaml``. The Walmart files contain no unit costs, stock
    levels, supplier data or capacities, so none is invented as fact. Quantities are in retail-sales dollars.

Model (per store, one MILP over all its departments and the forecast horizon, units: $ thousands)
  decisions  q[i,t] >= 0  order placed in week t (arrives in week t+L), z[i,t] in {0,1} order indicator
  scenarios  k in {low, mid, high} demand paths (interval ends / median), weights w_k; orders are here-and-now
  state      I[i,t,k] >= 0 end inventory, u[i,t,k] >= 0 unmet demand (lost sales)
  balance    I[i,t,k] = I[i,t-1,k] + q[i,t-L] - d[i,t,k] + u[i,t,k],   I[i,-1,k] = opening inventory
  capacity   sum_i I[i,t,k] <= capacity_store                              for all t, k
  objective  min  sum ordering_cost*z + sum_k w_k * sum (holding_rate*I + stockout_penalty*u)
"""
from __future__ import annotations

import pickle
import time

import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

from .config import ROOT
from .pipeline import artifacts_dir, selected_model


def solve_store(D: np.ndarray, w: np.ndarray, opening: np.ndarray, cap: float, L: int, order_cost: float,
                hold: float, penalty: float, time_limit_s: float, min_order: float = 0.0):
    """D: (n_items, H, K) demand scenarios (>=0). Returns q (n,H), z, status, objective, bound."""
    n, H, K = D.shape
    s = pywraplp.Solver.CreateSolver("SCIP")
    if s is None:
        s = pywraplp.Solver.CreateSolver("CBC")
    s.SetTimeLimit(int(time_limit_s * 1000))
    s.SetNumThreads(1)
    inf = s.infinity()
    last_order = H - 1 - L                          # orders later than this arrive after the horizon
    q = [[s.NumVar(0, inf, "") if t <= last_order else None for t in range(H)] for _ in range(n)]
    z = [[s.BoolVar("") if t <= last_order else None for t in range(H)] for _ in range(n)]
    I = [[[s.NumVar(0, inf, "") for _ in range(K)] for _ in range(H)] for _ in range(n)]
    u = [[[s.NumVar(0, inf, "") for _ in range(K)] for _ in range(H)] for _ in range(n)]
    obj = s.Objective()
    for i in range(n):
        bigM = float(D[i].sum(axis=0).max() * 1.5 + 1.0)
        for t in range(H):
            if q[i][t] is not None:
                s.Add(q[i][t] <= bigM * z[i][t])
                if min_order > 0:
                    s.Add(q[i][t] >= min_order * z[i][t])
                obj.SetCoefficient(z[i][t], order_cost / 1000.0)
            for k in range(K):
                prev = opening[i] if t == 0 else I[i][t - 1][k]
                arr = q[i][t - L] if (t - L >= 0 and q[i][t - L] is not None) else 0
                s.Add(I[i][t][k] == prev + arr - D[i, t, k] + u[i][t][k])
                obj.SetCoefficient(I[i][t][k], w[k] * hold)
                obj.SetCoefficient(u[i][t][k], w[k] * penalty)
    for t in range(H):
        for k in range(K):
            s.Add(sum(I[i][t][k] for i in range(n)) <= cap)
    obj.SetMinimization()
    st = s.Solve()
    ok = st in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE)
    Q = np.zeros((n, H)); Z = np.zeros((n, H))
    if ok:
        for i in range(n):
            for t in range(H):
                if q[i][t] is not None:
                    Q[i, t] = q[i][t].solution_value(); Z[i, t] = z[i][t].solution_value()
    status = {pywraplp.Solver.OPTIMAL: "OPTIMAL", pywraplp.Solver.FEASIBLE: "FEASIBLE"}.get(st, f"CODE_{st}")
    return Q, Z, status, (s.Objective().Value() if ok else np.nan), (s.Objective().BestBound() if ok else np.nan)


def simulate(plan_q: np.ndarray, demand: np.ndarray, opening: np.ndarray, L: int, order_cost: float, hold: float,
             penalty: float) -> dict:
    """Execute an order plan against realised demand (shape (n,H)); returns costs ($) and fill rate."""
    n, H = demand.shape
    inv = opening.copy(); hold_c = pen_c = unmet = 0.0; ords = int((plan_q > 1e-9).sum())
    for t in range(H):
        arr = plan_q[:, t - L] if t - L >= 0 else 0.0
        inv = inv + arr
        sold = np.minimum(inv, demand[:, t]); short = demand[:, t] - sold
        inv = inv - sold
        hold_c += hold * inv.sum(); pen_c += penalty * short.sum(); unmet += short.sum()
    tot_d = demand.sum()
    return dict(ordering=ords * order_cost, holding=hold_c, stockout=pen_c, total=ords * order_cost + hold_c + pen_c,
                fill_rate=1 - unmet / tot_d if tot_d > 0 else np.nan, n_orders=ords)


def base_stock_plan(mid: np.ndarray, spread: np.ndarray, opening: np.ndarray, L: int, z: float = 1.28) -> np.ndarray:
    """Reference policy: every week order up to (forecast over lead time + review week) + z * sigma, sigma from interval width."""
    n, H = mid.shape
    sigma = spread / (2 * 1.2816)
    plan = np.zeros((n, H)); inv = opening.copy(); pipeline = np.zeros((n, H + L + 1))
    for t in range(H):
        inv = inv + (plan[:, t - L] if t - L >= 0 else 0)
        if t <= H - 1 - L:
            cover = mid[:, t + 1: t + L + 2].sum(axis=1) if t + 1 < H else 0.0
            sig = sigma[:, t + 1: t + L + 2].mean(axis=1) * np.sqrt(L + 1) if t + 1 < H else 0.0
            target = cover + z * sig
            on_order = plan[:, max(0, t - L + 1): t].sum(axis=1) if L > 1 else 0.0
            plan[:, t] = np.maximum(target - inv - on_order, 0)
        inv = np.maximum(inv - mid[:, t], 0)   # planning-time projection with the mid forecast
    return plan


def run_inventory(cfg, prep, fold_name: str = "latest", method: str | None = None):
    method = method or selected_model(cfg, fold_name)
    ic = cfg["inventory"]
    bottom, hier = prep["panel_bottom"], prep["hier"]
    ad = artifacts_dir(cfg)
    fc = pd.read_parquet(ad / f"fold_{fold_name}" / "test_forecasts.parquet")
    off = hier.S.shape[0] - hier.S.shape[1]
    d = fc[(fc.model == method) & (fc.node >= off)].copy()
    d["sid"] = d.node - off
    H = int(d.h.max()); L = int(ic["lead_time_weeks"])
    ser = bottom.series
    test_dates = bottom.dates[int(d.target_idx.min()): int(d.target_idx.min()) + H]
    wk = np.array([ic["scenario_weights"]["low"], ic["scenario_weights"]["mid"], ic["scenario_weights"]["high"]])
    wk = wk / wk.sum()
    plans, store_rows, comp_rows = [], [], []
    t0 = time.time()
    cache = ad / "inv_cache"
    cache.mkdir(exist_ok=True)
    for store, sub in d.groupby(ser.Store.to_numpy()[d.sid.to_numpy()]):
        cp = cache / f"{fold_name}_{method}_store_{int(store):02d}.pkl"
        if cp.exists():   # resume after an interrupted run
            sr, cr, pl = pickle.loads(cp.read_bytes())
            store_rows.append(sr); comp_rows.extend(cr); plans.extend(pl)
            continue
        n_c, n_p = len(comp_rows), len(plans)
        piv = {c: sub.pivot(index="sid", columns="h", values=c) for c in ("y_hat", "lo", "hi", "y")}
        sids = piv["y_hat"].index.to_numpy()
        mid = np.clip(piv["y_hat"].to_numpy(), 0, None); lo = np.clip(piv["lo"].to_numpy(), 0, None)
        hi = np.clip(piv["hi"].to_numpy(), 0, None)
        ok = np.isfinite(mid).all(axis=1) & np.isfinite(lo).all(axis=1) & np.isfinite(hi).all(axis=1) & (mid.sum(axis=1) > 0)
        sids, mid, lo, hi = sids[ok], mid[ok], np.minimum(lo[ok], mid[ok]), np.maximum(hi[ok], mid[ok])
        actual = np.nan_to_num(piv["y"].to_numpy()[ok], nan=0.0).clip(0)
        D = np.stack([lo, mid, hi], axis=2) / 1000.0
        opening = ic["opening_cover_weeks"] * mid.mean(axis=1) / 1000.0
        size = float(ser.Size.iloc[sids[0]])
        cap = max(ic["storage_capacity_per_sqft"] * size / 1000.0, opening.sum() * 1.05)
        Q, Z, status, objv, bound = solve_store(D, wk, opening, cap, L, ic["ordering_cost"], ic["holding_cost_rate_weekly"],
                                                ic["stockout_penalty_rate"], ic["time_limit_seconds_per_store"],
                                                ic["min_order_quantity"] / 1000.0)
        Q *= 1000.0; opening_d = opening * 1000.0
        bs = base_stock_plan(mid, hi - lo, opening_d, L)
        sim_kw = dict(L=L, order_cost=ic["ordering_cost"], hold=ic["holding_cost_rate_weekly"], penalty=ic["stockout_penalty_rate"])
        r_opt = simulate(Q, actual, opening_d, **sim_kw)
        r_bs = simulate(bs, actual, opening_d, **sim_kw)
        r_none = simulate(np.zeros_like(Q), actual, opening_d, **sim_kw)
        gap = abs(objv - bound) / max(abs(objv), 1e-9) if np.isfinite(objv) else np.nan
        store_rows.append(dict(Store=store, n_items=len(sids), solver_status=status, mip_gap=gap,
                               capacity_usd=cap * 1000, opening_usd=opening_d.sum(), planned_order_usd=Q.sum()))
        for name, r in (("optimised_plan", r_opt), ("base_stock_reference", r_bs), ("no_reorder", r_none)):
            comp_rows.append(dict(Store=store, policy=name, **r))
        for j, sid in enumerate(sids):
            for t in range(H):
                plans.append(dict(Store=store, Dept=int(ser.Dept.iloc[sid]), sid=int(sid), week=test_dates[t], h=t + 1,
                                  forecast_p50=mid[j, t], forecast_lo=lo[j, t], forecast_hi=hi[j, t], actual_sales=actual[j, t],
                                  order_qty=Q[j, t], arrival_week=(test_dates[t + L] if t + L < H else pd.NaT),
                                  opening_inventory=opening_d[j] if t == 0 else np.nan))
        cp.write_bytes(pickle.dumps((store_rows[-1], comp_rows[n_c:], plans[n_p:])))
        print(f"  store {store:02d}: {len(sids)} items {status} gap={gap:.3f} ({time.time()-t0:.0f}s)", flush=True)
    plan = pd.DataFrame(plans)
    stores = pd.DataFrame(store_rows)
    comp = pd.DataFrame(comp_rows)
    res = ROOT / "reports/results"
    plan.to_parquet(ad / "inventory_plan.parquet")
    stores.to_csv(res / "inventory_solver_summary.csv", index=False)
    comp.to_csv(res / "inventory_policy_comparison_by_store.csv", index=False)
    summ = comp.groupby("policy")[["ordering", "holding", "stockout", "total", "n_orders"]].sum()
    tot_d = plan.groupby("Store").actual_sales.sum().sum()
    fr = comp.groupby("policy").apply(lambda g: np.average(g.fill_rate, weights=plan.groupby("Store").actual_sales.sum().loc[g.Store].to_numpy()))
    summ["fill_rate"] = fr
    summ.to_csv(res / "inventory_policy_comparison.csv")
    print(summ.round(3).to_string(), flush=True)
    return plan, stores, summ

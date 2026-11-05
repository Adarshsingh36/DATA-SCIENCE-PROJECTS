"""Uplift / treatment-effect modelling with CausalML (S-Learner and T-Learner).

!!! IMPORTANT !!!  The IBM Telco data contain NO retention-offer assignment and NO post-offer outcomes.
The treatment/control experiment below is therefore SIMULATED with explicit, documented assumptions
(see simulate_experiment). The control-arm outcome is the REAL observed retention label; only the
treatment effect and treatment assignment are simulated. Estimated uplift is a demonstration of the
method on a known data-generating process, NOT experimentally validated causal evidence.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold, train_test_split
from causalml.inference.meta import BaseSClassifier, BaseTClassifier
from causalml.metrics import auuc_score, qini_score

from src import config
from src.preprocessing import model_frame, build_preprocessor

SIM_ASSUMPTIONS = {
    "treatment_assignment": f"Bernoulli({config.SIM_TREATMENT_PROB}) independent of covariates (a randomized A/B test), seed {config.RANDOM_STATE}",
    "control_outcome": "REAL observed retention (1 - churn)",
    "treatment_effect_for_would_be_churners": "P(saved | would churn, treated at 10% discount) = tau(x), additive rules below, clipped to [0, 0.45]",
    "tau_rules": {"base": 0.05, "monthly_charges>=70": 0.18, "30<=monthly_charges<70": 0.08, "month_to_month": 0.10,
                  "tenure<=12": 0.08, "paperless_billing": 0.05, "auto_pay": -0.05, "protection_bundle": -0.06},
    "sleeping_dogs": "Treated non-churners with contract>=12m and tenure>24m churn with prob 0.04 (offer prompts plan review)",
    "discount_depth": f"effects calibrated to a {config.SIM_REFERENCE_DISCOUNT_PCT:.0f}% discount; other depths use a diminishing-returns multiplier (src/roi.py)",
}


def response_propensity(df: pd.DataFrame) -> np.ndarray:
    """tau(x): probability that a would-be churner is saved by the offer (SIMULATION ASSUMPTION)."""
    mc, ten = df["monthly_charges"].values, df["tenure_months"].values
    tau = (0.05 + 0.18 * (mc >= 70) + 0.08 * ((mc >= 30) & (mc < 70)) + 0.10 * df["is_month_to_month"].values
           + 0.08 * (ten <= 12) + 0.05 * df["paperless_billing"].values - 0.05 * df["auto_pay"].values
           - 0.06 * df["has_protection_bundle"].values)
    return np.clip(tau, 0.0, config.SIM_MAX_RESPONSE)


def sleeping_dog_prob(df: pd.DataFrame) -> np.ndarray:
    return np.where((df["contract_months"].values >= 12) & (df["tenure_months"].values > 24), 0.04, 0.0)


def simulate_experiment(df: pd.DataFrame, baseline_churn_prob, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Reproducible simulated randomized offer experiment. Outcome y = 1 means RETAINED."""
    rng = np.random.default_rng(seed)
    n = len(df)
    w = rng.binomial(1, config.SIM_TREATMENT_PROB, n)
    churn = df["churn"].values.astype(int)
    tau, delta = response_propensity(df), sleeping_dog_prob(df)
    u = rng.random(n)
    y_control = 1 - churn
    y_treated = np.where(churn == 1, (u < tau).astype(int), (u >= delta).astype(int))
    y = np.where(w == 1, y_treated, y_control)
    p0 = np.asarray(baseline_churn_prob, float)
    reference = p0 * tau - (1 - p0) * delta      # expected ITE given x under the simulation (for validation only)
    return pd.DataFrame({"customer_id": df["customer_id"].values, "sim_treatment": w, "sim_retained": y,
                         "tau": tau, "reference_effect": reference})


def _make(kind: str):
    lg = LGBMClassifier(n_estimators=200, learning_rate=0.04, num_leaves=7, min_child_samples=40, subsample=0.8,
                        subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, random_state=config.RANDOM_STATE,
                        verbose=-1, n_jobs=1)
    if kind == "S":
        return BaseSClassifier(learner=lg)
    return BaseTClassifier(learner=lg)


def _fit_predict(kind, Xtr, wtr, ytr, Xte) -> np.ndarray:
    m = _make(kind)
    m.fit(X=Xtr, treatment=wtr, y=ytr)
    return np.asarray(m.predict(X=Xte)).ravel()


def _design(df: pd.DataFrame):
    pre = build_preprocessor()
    X = pre.fit_transform(model_frame(df))          # unsupervised one-hot / passthrough; no label used
    return X.values.astype(float), list(X.columns)


def uplift_table(y, w, score, n_bins: int = 10) -> pd.DataFrame:
    d = pd.DataFrame({"y": y, "w": w, "s": score}).sort_values("s", ascending=False).reset_index(drop=True)
    d["decile"] = pd.qcut(d.index, n_bins, labels=range(1, n_bins + 1))
    rows = []
    for k, g in d.groupby("decile", observed=True):
        t, c = g[g.w == 1], g[g.w == 0]
        rows.append({"decile": int(k), "n": len(g), "mean_predicted_ite": g.s.mean(),
                     "retention_treated": t.y.mean(), "retention_control": c.y.mean(),
                     "observed_uplift": t.y.mean() - c.y.mean()})
    return pd.DataFrame(rows)


def evaluate_uplift(y, w, scores: dict, reference=None) -> dict:
    """Qini / AUUC on held-out simulated data plus (simulation-only) recovery of the known effect."""
    frame = pd.DataFrame({"y": y, "w": w, **scores})
    cols = list(scores)
    q = qini_score(frame, outcome_col="y", treatment_col="w")
    a = auuc_score(frame, outcome_col="y", treatment_col="w")
    out = {"qini": {c: float(q[c]) for c in cols}, "auuc": {c: float(a[c]) for c in cols}}
    if reference is not None:
        out["spearman_vs_reference_effect"] = {c: float(spearmanr(scores[c], reference)[0]) for c in cols}
    return out


def run_uplift(df: pd.DataFrame, baseline_churn_prob, seed: int = config.RANDOM_STATE, save: bool = True) -> dict:
    sim = simulate_experiment(df, baseline_churn_prob, seed)
    X, names = _design(df)
    w, y = sim["sim_treatment"].values, sim["sim_retained"].values
    p0 = np.asarray(baseline_churn_prob, float)

    strat = w * 2 + y
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.30, stratify=strat, random_state=seed)
    s_te = _fit_predict("S", X[idx_tr], w[idx_tr], y[idx_tr], X[idx_te])
    t_te = _fit_predict("T", X[idx_tr], w[idx_tr], y[idx_tr], X[idx_te])
    rng = np.random.default_rng(seed)
    scores = {"S_learner": s_te, "T_learner": t_te, "Churn_risk_baseline": p0[idx_te],
              "Random": rng.random(len(idx_te))}
    ev = evaluate_uplift(y[idx_te], w[idx_te], scores, reference=sim["reference_effect"].values[idx_te])
    best = "S_learner" if ev["qini"]["S_learner"] >= ev["qini"]["T_learner"] else "T_learner"

    # cross-fitted ITE for every customer (each customer scored by models that never saw them)
    ite_s, ite_t = np.zeros(len(df)), np.zeros(len(df))
    for tr, te in StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=seed).split(X, strat):
        ite_s[te] = _fit_predict("S", X[tr], w[tr], y[tr], X[te])
        ite_t[te] = _fit_predict("T", X[tr], w[tr], y[tr], X[te])
    ite_sel = ite_s if best == "S_learner" else ite_t

    treated = sim["sim_treatment"] == 1
    summary = {
        "n_customers": int(len(df)), "n_treated": int(treated.sum()), "n_control": int((~treated).sum()),
        "retention_treated": float(y[w == 1].mean()), "retention_control": float(y[w == 0].mean()),
        "simulated_average_treatment_effect": float(y[w == 1].mean() - y[w == 0].mean()),
        "mean_reference_effect": float(sim["reference_effect"].mean()),
        "mean_estimated_ite_selected": float(ite_sel.mean()),
        "share_positive_uplift": float((ite_sel > 0).mean()),
        "selected_learner": best, "evaluation_holdout": ev,
        "uplift_deciles_holdout_selected": uplift_table(y[idx_te], w[idx_te], (s_te if best == "S_learner" else t_te)).round(4).to_dict("records"),
        "uplift_deciles_holdout_churn_risk": uplift_table(y[idx_te], w[idx_te], p0[idx_te]).round(4).to_dict("records"),
        "assumptions": SIM_ASSUMPTIONS,
        "disclaimer": "SIMULATED treatment/control experiment; not experimentally validated causal evidence.",
    }
    out = sim.assign(ite_s_learner=ite_s, ite_t_learner=ite_t, ite_selected=ite_sel, selected_learner=best)
    out["positive_uplift"] = out["ite_selected"] > 0
    out["quadrant"] = np.select(
        [out["ite_selected"] > 0.02, out["ite_selected"] < -0.01, p0 >= 0.5],
        ["Persuadable", "Sleeping dog", "Lost cause"], default="Sure thing")
    out["is_simulated"] = True
    out["_holdout_index"] = False
    out.loc[out.index[idx_te], "_holdout_index"] = True
    if save:
        (config.REPORTS_DIR / "uplift_metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    return {"table": out, "summary": summary, "holdout_curves": {"y": y[idx_te], "w": w[idx_te], **scores}}


def add_economics(upl: pd.DataFrame, df: pd.DataFrame, churn_prob, discount_pct=10.0, offer_cost=10.0, duration=12.0) -> pd.DataFrame:
    from src.roi import customer_economics
    econ = customer_economics(df["monthly_charges"].values, churn_prob, upl["ite_selected"].values,
                              discount_pct, offer_cost, duration)
    out = upl.copy()
    out["expected_net_value"] = econ["net_benefit"].values
    out["economically_worthwhile"] = (econ["net_benefit"].values > 0) & (upl["ite_selected"].values > 0)
    return out

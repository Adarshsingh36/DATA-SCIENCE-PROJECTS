import numpy as np
import pandas as pd
import pytest
from src import churn_model as cm, shap_analysis as sa, survival_model as sm, uplift_model as um
from src import clv as clvmod, segmentation as seg, config


@pytest.fixture(scope="module")
def trained(small):
    return cm.train_and_evaluate(small, n_iter=2, save=False)


def test_churn_training_metrics(trained):
    for key in ("baseline_logreg", "lightgbm"):
        m = trained[key]["test_at_0.5"]
        for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
            assert 0 <= m[k] <= 1
        cmx = m["confusion_matrix"]; assert sum(cmx.values()) == trained["n_test"]
    assert trained["lightgbm"]["test_at_0.5"]["roc_auc"] > 0.75           # clearly better than chance
    assert trained["lightgbm"]["test_at_0.5"]["roc_auc"] > 0.5 + 0.2


def test_stratified_split_preserves_rate(small):
    Xtr, Xte, ytr, yte = cm.split_data(small)
    assert abs(ytr.mean() - yte.mean()) < 0.01 and len(Xtr) + len(Xte) == len(small)


def test_inference(trained, small):
    model = trained["_objects"]["final_model"]
    p = cm.predict_proba_new(small.head(20), model)
    assert p.shape == (20,) and ((p >= 0) & (p <= 1)).all()
    assert set(cm.risk_tier([0.1, 0.3, 0.9])) == {"Low", "Medium", "High"}


def test_oof_scoring(trained, small):
    o = trained["_objects"]
    s = cm.score_all_customers(small, o["final_model"], o["best_pipeline"], o["threshold"])
    assert len(s) == len(small) and s.customer_id.is_unique and s.churn_probability.between(0, 1).all()


def test_saved_model_inference(df):
    p = cm.predict_proba_new(df.head(50))
    assert p.shape == (50,) and ((p >= 0) & (p <= 1)).all()


def test_shap_outputs(trained, small):
    pipe = trained["_objects"]["best_pipeline"]
    sdf, Xt, base = sa.compute_shap(pipe, small)
    assert sdf.shape == Xt.shape and len(sdf) == len(small)
    raw = pipe.named_steps["clf"].predict(Xt.values, raw_score=True)
    assert np.allclose(sdf.sum(axis=1).values + base, raw, atol=1e-4)      # SHAP additivity
    imp = sa.global_importance(sdf); assert imp.mean_abs_shap.is_monotonic_decreasing
    tf = sa.top_factors(sdf.iloc[0], Xt.iloc[0], k=5)
    assert len(tf) == 5 and set(tf.direction) <= {"raises churn risk", "lowers churn risk"}
    rec = sa.explanation_records(sdf.head(3), Xt.head(3), base); assert len(rec) == 3 and rec.shap_json.str.len().min() > 10


def test_kaplan_meier_and_cox(df):
    km = sm.fit_kaplan_meier(df)
    s = km.survival_function_.iloc[:, 0]; assert s.is_monotonic_decreasing and 0 < s.min() <= 1
    cph, design = sm.fit_cox(df)
    assert design.churn.sum() == df.churn.sum() and (design.tenure_months > 0).all()
    hr = sm.hazard_ratios(cph); assert (hr.hazard_ratio > 0).all()
    ph = sm.check_ph_assumption(cph, design); assert {"covariate", "p", "violates_ph_at_5pct"} <= set(ph.columns)
    assert 0.5 < cph.concordance_index_ < 1


def test_survival_predictions(df):
    cph, _ = sm.fit_cox(df)
    sub = df.sample(300, random_state=1)
    M = sm.conditional_survival_matrix(cph, sub, horizon=24)
    assert M.shape == (300, 24) and (M >= 0).all() and (M <= 1 + 1e-9).all()
    assert (np.diff(M, axis=1) <= 1e-9).all()                        # non-increasing
    sp = sm.survival_predictions(cph, sub)
    assert (sp.surv_prob_12m >= sp.surv_prob_24m - 1e-9).all() and (sp.surv_prob_24m >= sp.surv_prob_36m - 1e-9).all()
    assert sp.expected_months_next_24.between(0, 24).all()
    c = sm.customer_curve(cph, sub.head(1)); assert c.survival_probability.iloc[0] == 1.0 and len(c) == 37


def test_cox_higher_risk_for_month_to_month_vs_two_year(df):
    cph, _ = sm.fit_cox(df, stratify_contract=False)
    hr = sm.hazard_ratios(cph).set_index("covariate")
    assert hr.loc["contract_two_year", "hazard_ratio"] < 1 < 100


def test_uplift_simulation_reproducible(small):
    p0 = np.full(len(small), 0.27)
    a, b = um.simulate_experiment(small, p0, seed=7), um.simulate_experiment(small, p0, seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert not a.sim_treatment.equals(um.simulate_experiment(small, p0, seed=8).sim_treatment)
    assert abs(a.sim_treatment.mean() - config.SIM_TREATMENT_PROB) < 0.05
    ctrl = a[a.sim_treatment == 0]
    assert (ctrl.sim_retained.values == 1 - small.loc[ctrl.index, "churn"].values).all()   # control = real labels
    assert um.response_propensity(small).max() <= config.SIM_MAX_RESPONSE


def test_uplift_estimation(small):
    p0 = np.clip(small.churn.values * 0.5 + 0.1, 0, 1)
    out = um.run_uplift(small, p0, save=False)
    t = out["table"]
    assert len(t) == len(small) and t.ite_selected.between(-1, 1).all()
    assert set(t.quadrant) <= {"Persuadable", "Sleeping dog", "Lost cause", "Sure thing"}
    assert out["summary"]["selected_learner"] in ("S_learner", "T_learner")
    assert out["summary"]["evaluation_holdout"]["qini"]["Random"] < out["summary"]["evaluation_holdout"]["qini"][out["summary"]["selected_learner"]]
    assert "SIMULATED" in out["summary"]["disclaimer"] and t.is_simulated.all()


def test_clv_matches_closed_form():
    h, H, r, m = 0.02, 36, 0.10, 20.0
    S = np.tile((1 - h) ** np.arange(1, H + 1), (3, 1))
    v = clvmod.compute_clv([50, 50, 50], S, margin=0.4, annual_rate=r)
    assert np.allclose(v, clvmod.clv_constant_hazard(m, h, H, r))
    assert clvmod.compute_clv([50], np.zeros((1, 12)))[0] == 0
    assert clvmod.compute_clv([100], np.ones((1, 12)), 0.5, 0.0)[0] == pytest.approx(600)


def test_clv_table_real(df):
    cph, _ = sm.fit_cox(df); sub = df.head(200)
    M = sm.conditional_survival_matrix(cph, sub, horizon=36)
    t = clvmod.clv_table(sub, M, np.full(200, 0.3))
    assert (t.clv > 0).all() and (t.clv <= sub.monthly_charges.values * config.CONTRIBUTION_MARGIN * 36 + 1e-6).all()
    assert np.allclose(t.revenue_at_risk_annual, 0.3 * sub.monthly_charges.values * 12)


def test_segmentation():
    rng = np.random.default_rng(0); n = 600
    g = rng.integers(0, 3, n)
    f = pd.DataFrame({"customer_id": [f"c{i}" for i in range(n)], "churn": rng.binomial(1, 0.3, n),
        "tenure_months": rng.normal(10 + 25 * g, 3), "monthly_charges": rng.normal(30 + 25 * g, 4),
        "clv": rng.normal(200 + 400 * g, 30).clip(1), "service_count": rng.normal(2 + 2 * g, .4),
        "churn_probability": rng.uniform(0, 1, n) * (1 - g * 0.3), "surv_prob_12m": rng.uniform(.5, 1, n)})
    r = seg.run_segmentation(f, save=False)
    assert 3 <= r["k"] <= 8 and len(r["segments"]) == n and r["segments"].customer_id.is_unique
    assert len(set(r["profile"].segment_label)) == r["k"]            # labels unique per cluster
    assert r["silhouette"] > 0.2
    assert seg.choose_k(pd.DataFrame({"k": [2, 3, 4], "silhouette": [0.9, 0.3, 0.5]})) == 4

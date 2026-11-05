"""End-to-end pipeline:  python -m src.pipeline [--n-iter 25]
ingest -> EDA -> churn model -> SHAP -> survival -> CLV -> uplift (simulated) -> segmentation -> DB + reports."""
from __future__ import annotations
import argparse, json, logging, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import text

from src import config, ingestion, eda, churn_model, shap_analysis, survival_model as sm, clv as clvmod
from src import uplift_model as um, segmentation as seg, roi, preprocessing as pp
from database.connection import get_engine
from database import models

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("pipeline")
P = config.DATA_PROCESSED


def write_table(df: pd.DataFrame, name: str, engine) -> None:
    cols = [c.name for c in models.metadata.tables[name].columns if c.name in df.columns]
    with engine.begin() as c:
        c.execute(text(f"DELETE FROM {name}"))
    df[cols].to_sql(name, engine, if_exists="append", index=False, chunksize=1000, method="multi")


def km_figures(df, km_all, cox, ph):
    plt.figure(figsize=(8, 5)); km_all.plot_survival_function(); plt.title("Kaplan-Meier survival (all customers)")
    plt.xlabel("tenure (months)"); plt.ylabel("P(still a customer)"); plt.tight_layout(); plt.savefig(config.FIGURES_DIR / "km_overall.png", dpi=130); plt.close()
    plt.figure(figsize=(8, 5))
    for k in sm.km_by_group(df, "contract").values(): k.plot_survival_function()
    plt.title("Kaplan-Meier by contract"); plt.xlabel("tenure (months)"); plt.tight_layout(); plt.savefig(config.FIGURES_DIR / "km_by_contract.png", dpi=130); plt.close()
    plt.figure(figsize=(8, 6)); cox.plot(); plt.title("Cox model coefficients (log HR, 95% CI), stratified by contract")
    plt.tight_layout(); plt.savefig(config.FIGURES_DIR / "cox_coefficients.png", dpi=130); plt.close()


def run(n_iter: int = 25):
    t0 = time.time(); eng = get_engine()
    log.info("backend=%s", eng.dialect.name)
    summary = ingestion.ingest(engine=eng)
    df = pd.read_sql_table("customers", eng)
    df.columns = [str(c) for c in df.columns]
    df = df.sort_values("customer_id").reset_index(drop=True)   # deterministic order for seeded stages
    raw = ingestion.load_raw_excel()
    leak = pp.assess_cltv_leakage(pp.clean_raw(raw))
    pp.feature_dictionary().to_csv(config.REPORTS_DIR / "feature_dictionary.csv", index=False)
    eda_s = eda.run_eda(df); log.info("EDA done")

    # ---- churn ----
    res = churn_model.train_and_evaluate(df, n_iter=n_iter)
    obj = res.pop("_objects")
    scores = churn_model.score_all_customers(df, obj["final_model"], obj["best_pipeline"], obj["threshold"])
    scores.to_csv(P / "churn_scores.csv", index=False)
    write_table(scores, "churn_predictions", eng); log.info("churn done %.0fs", time.time() - t0)

    # ---- SHAP ----
    pipe = shap_analysis.fit_explainer_model(obj["best_pipeline"], df)
    sdf, Xt, base = shap_analysis.compute_shap(pipe, df)
    imp = shap_analysis.global_importance(sdf); imp.to_csv(config.REPORTS_DIR / "shap_global_importance.csv", index=False)
    shap_analysis.make_plots(sdf, Xt)
    write_table(shap_analysis.explanation_records(sdf, Xt, base), "churn_explanations", eng); log.info("shap done")

    # ---- survival ----
    km = sm.fit_kaplan_meier(df)
    cox, design = sm.fit_cox(df, stratify_contract=True)
    cox_u, design_u = sm.fit_cox(df, stratify_contract=False)
    ph, ph_u = sm.check_ph_assumption(cox, design), sm.check_ph_assumption(cox_u, design_u)
    hr, hr_u = sm.hazard_ratios(cox), sm.hazard_ratios(cox_u)
    hr.to_csv(config.REPORTS_DIR / "cox_hazard_ratios_stratified.csv", index=False)
    hr_u.to_csv(config.REPORTS_DIR / "cox_hazard_ratios_with_contract.csv", index=False)
    ph.to_csv(config.REPORTS_DIR / "cox_ph_test_stratified.csv", index=False)
    ph_u.to_csv(config.REPORTS_DIR / "cox_ph_test_with_contract.csv", index=False)
    km_figures(df, km, cox, ph); sm.save_cox(cox)
    sm.save_cox(cox_u, config.MODELS_DIR / "cox_model_unstratified.joblib")
    surv = sm.survival_predictions(cox, df, stratified=True)
    write_table(surv, "survival_predictions", eng)
    M = sm.conditional_survival_matrix(cox, df, horizon=config.CLV_HORIZON_MONTHS, stratified=True)
    surv_s = {"km_survival": sm.km_summary(km).round(4).to_dict("records"),
              "logrank_contract": sm.logrank_by_group(df, "contract"),
              "cox_concordance": float(cox.concordance_index_), "cox_concordance_with_contract_dummies": float(cox_u.concordance_index_),
              "ph_violations_stratified": ph.loc[ph.violates_ph_at_5pct, "covariate"].tolist(),
              "ph_violations_with_contract_dummies": ph_u.loc[ph_u.violates_ph_at_5pct, "covariate"].tolist(),
              "n_events": int(df.churn.sum()), "n_censored": int((1 - df.churn).sum())}
    (config.REPORTS_DIR / "survival_metrics.json").write_text(json.dumps(surv_s, indent=2)); log.info("survival done")

    # ---- CLV ----
    clv_df = clvmod.clv_table(df, M, scores["churn_probability"].values)
    write_table(clv_df, "customer_clv", eng)
    clv_s = {"mean_clv": float(clv_df.clv.mean()), "median_clv": float(clv_df.clv.median()), "total_clv": float(clv_df.clv.sum()),
             "revenue_at_risk_annual_total": float(clv_df.revenue_at_risk_annual.sum()),
             "assumptions": {"margin": config.CONTRIBUTION_MARGIN, "annual_discount_rate": config.ANNUAL_DISCOUNT_RATE, "horizon_months": config.CLV_HORIZON_MONTHS},
             "ibm_cltv_assessment": leak}

    # ---- uplift ----
    up = um.run_uplift(df, scores["churn_probability"].values)
    utab = um.add_economics(up["table"], df, scores["churn_probability"].values)
    write_table(utab, "uplift_predictions", eng)
    utab.to_csv(P / "uplift_table.csv", index=False)
    pd.DataFrame(up["holdout_curves"]).to_csv(P / "uplift_holdout.csv", index=False)
    log.info("uplift done")

    # ---- segmentation ----
    frame = df[["customer_id", "tenure_months", "monthly_charges", "service_count", "churn"]].merge(clv_df[["customer_id", "clv"]], on="customer_id") \
        .merge(scores[["customer_id", "churn_probability"]], on="customer_id").merge(surv[["customer_id", "surv_prob_12m"]], on="customer_id")
    sg = seg.run_segmentation(frame)
    write_table(sg["segments"], "customer_segments", eng)
    sg["profile"].round(3).to_csv(config.REPORTS_DIR / "segment_profiles.csv")
    sg["k_scores"].to_csv(config.REPORTS_DIR / "segment_k_scores.csv", index=False)
    seg_s = {"k": sg["k"], "silhouette": sg["silhouette"], "k_scores": sg["k_scores"].round(4).to_dict("records"),
             "profiles": sg["profile"].round(3).reset_index().to_dict("records")}
    (config.REPORTS_DIR / "segmentation_metrics.json").write_text(json.dumps(seg_s, indent=2, default=float))

    # ---- default ROI scenarios saved to DB ----
    roi_frame = df[["customer_id", "monthly_charges"]].merge(scores[["customer_id", "churn_probability"]], on="customer_id") \
        .merge(utab[["customer_id", "ite_selected"]], on="customer_id")
    scen = []
    for strat in roi.STRATEGIES:
        r = roi.simulate_campaign(roi_frame, 10.0, 20000.0, 1000, 10.0, 12.0, strat)
        scen.append({k: v for k, v in r.items() if not k.startswith("_")})
        with eng.begin() as c:
            cid = c.execute(text("INSERT INTO retention_campaigns (name,strategy,discount_pct,budget,n_targeted_requested,offer_cost_per_customer,retention_duration_months) "
                                 "VALUES (:n,:s,10,20000,1000,10,12) RETURNING campaign_id") if eng.dialect.name == "postgresql" else
                            text("INSERT INTO retention_campaigns (name,strategy,discount_pct,budget,n_targeted_requested,offer_cost_per_customer,retention_duration_months) VALUES (:n,:s,10,20000,1000,10,12) RETURNING campaign_id"),
                            {"n": f"Default scenario: {strat}", "s": strat}).scalar()
            c.execute(text("INSERT INTO retention_results (campaign_id,n_targeted,baseline_churn_prob,incremental_retention_prob,expected_customers_retained,expected_retained_revenue,incremental_benefit,discount_cost,campaign_cost,net_incremental_benefit,roi_pct) "
                           "VALUES (:cid,:n,:b,:i,:e,:rr,:ib,:dc,:cc,:nb,:roi)"),
                      {"cid": cid, "n": r["n_targeted"], "b": r["baseline_churn_prob"], "i": r["incremental_retention_prob"], "e": r["expected_customers_retained"],
                       "rr": r["expected_retained_revenue"], "ib": r["incremental_benefit"], "dc": r["discount_cost"], "cc": r["campaign_cost"], "nb": r["net_incremental_benefit"],
                       "roi": None if np.isnan(r["roi_pct"]) else r["roi_pct"]})
    (config.REPORTS_DIR / "default_roi_scenarios.json").write_text(json.dumps(scen, indent=2, default=float))

    # ---- model_runs + master summary ----
    runs = []
    for mname, key in [("logreg_baseline", "baseline_logreg"), ("lightgbm", "lightgbm")]:
        for k, v in res[key]["test_at_0.5"].items():
            if k not in ("confusion_matrix", "threshold"): runs.append({"model_name": mname, "metric_name": k, "metric_value": float(v), "split": "test"})
    write_table(pd.DataFrame(runs), "model_runs", eng)
    full = {"churn": res, "eda": eda_s, "survival": surv_s, "clv": clv_s, "uplift": up["summary"], "segmentation": seg_s,
            "roi_default_scenarios": scen, "data_quality": summary["quality"], "backend": eng.dialect.name,
            "runtime_seconds": round(time.time() - t0, 1)}
    (config.REPORTS_DIR / "all_metrics.json").write_text(json.dumps(full, indent=2, default=float))
    log.info("PIPELINE COMPLETE in %.0fs", time.time() - t0)
    return full


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n-iter", type=int, default=25)
    run(ap.parse_args().n_iter)

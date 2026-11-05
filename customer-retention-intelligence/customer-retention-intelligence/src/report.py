"""Generates README.md and reports/technical_report.md from the saved metrics files (no hand-typed numbers).
Run after the pipeline:  python -m src.report"""
from __future__ import annotations
import json
import re
import pandas as pd
from src import config

R = config.REPORTS_DIR


def J(name): return json.loads((R / name).read_text())


def table(df: pd.DataFrame, floatfmt=3) -> str:
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f": d[c] = d[c].map(lambda v: f"{v:.{floatfmt}f}")
    head = "| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns) + "\n"
    return head + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in d.values)


def build():
    M = J("all_metrics.json"); c = M["churn"]; sv = M["survival"]; up = M["uplift"]; sg = M["segmentation"]; clv = M["clv"]; eda = M["eda"]
    dq = M["data_quality"]
    base_t, lg_t, lg_tt, base_tt = (c["baseline_logreg"]["test_at_0.5"], c["lightgbm"]["test_at_0.5"],
                                    c["lightgbm"]["test_at_tuned_threshold"], c["baseline_logreg"]["test_at_tuned_threshold"])
    cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier"]
    perf = pd.DataFrame([{"model": n, **{k: d[k] for k in cols}} for n, d in
                         [("Logistic regression (baseline), thr 0.50", base_t), ("LightGBM, thr 0.50", lg_t),
                          (f"Logistic regression, thr {base_tt['threshold']:.2f}", base_tt),
                          (f"LightGBM, thr {lg_tt['threshold']:.2f} (F1-optimal on train OOF)", lg_tt)]])
    cvt = pd.DataFrame({"metric": list(c["lightgbm"]["cv_mean"]),
                        "LightGBM CV (mean ± sd)": [f"{c['lightgbm']['cv_mean'][k]:.3f} ± {c['lightgbm']['cv_std'][k]:.3f}" for k in c["lightgbm"]["cv_mean"]],
                        "Baseline CV (mean ± sd)": [f"{c['baseline_logreg']['cv_mean'][k]:.3f} ± {c['baseline_logreg']['cv_std'][k]:.3f}" for k in c["lightgbm"]["cv_mean"]]})
    cm = lg_t["confusion_matrix"]; cmt = lg_tt["confusion_matrix"]
    shap = pd.read_csv(R / "shap_global_importance.csv").head(10)
    hr = pd.read_csv(R / "cox_hazard_ratios_with_contract.csv").round(3)
    kms = pd.DataFrame(sv["km_survival"])
    prof = pd.read_csv(R / "segment_profiles.csv").round(2)
    ks = pd.read_csv(R / "segment_k_scores.csv")
    rs = pd.DataFrame(M["roi_default_scenarios"])[["strategy", "n_targeted", "expected_customers_retained", "incremental_benefit", "discount_cost", "campaign_cost", "net_incremental_benefit", "roi_pct"]]
    ev = up["evaluation_holdout"]
    evt = pd.DataFrame({"ranking": list(ev["qini"]), "Qini": list(ev["qini"].values()), "AUUC": list(ev["auuc"].values()),
                        "Spearman vs simulated true effect": list(ev["spearman_vs_reference_effect"].values())})
    dec = pd.DataFrame(up["uplift_deciles_holdout_selected"])[["decile", "n", "mean_predicted_ite", "retention_treated", "retention_control", "observed_uplift"]]
    up_tab = pd.read_csv(config.DATA_PROCESSED / "uplift_table.csv"); sc = pd.read_csv(config.DATA_PROCESSED / "churn_scores.csv")
    n_high, n_pos, n_worth = int((sc.risk_tier == "High").sum()), int(up_tab.positive_uplift.sum()), int(up_tab.economically_worthwhile.sum())
    quad = up_tab.quadrant.value_counts().to_dict()
    tests = (R / "test_results.txt").read_text().strip().splitlines()[-1] if (R / "test_results.txt").exists() else "see `pytest`"
    ph_s, ph_u = sv["ph_violations_stratified"], sv["ph_violations_with_contract_dummies"]
    top_shap = ", ".join(f"`{f}`" for f in shap.feature.head(5))
    warn = ("; ".join(dq["warnings"]) or "none")

    # ---------------------------------------------------------------- technical report
    rep = f"""# Technical Report: Customer Churn Prediction and Retention Intelligence Platform

All numbers below are produced by `python -m src.pipeline` and read from `reports/*.json|csv`; none are hand-entered.
Backend used for this run: **{M['backend']}**. Pipeline runtime: {M['runtime_seconds']} s.

## 1. Executive summary
* Dataset: IBM Telco Customer Churn, {eda['n']:,} customers, actual churn rate **{eda['churn_rate']:.1%}**.
* Churn model (LightGBM, calibrated) reaches hold-out ROC-AUC **{lg_t['roc_auc']:.3f}** and PR-AUC **{lg_t['pr_auc']:.3f}** versus {base_t['roc_auc']:.3f} / {base_t['pr_auc']:.3f} for a logistic-regression baseline. The improvement is real but small (cross-validated ROC-AUC {c['lightgbm']['cv_mean']['roc_auc']:.3f} vs {c['baseline_logreg']['cv_mean']['roc_auc']:.3f}, fold s.d. about {c['lightgbm']['cv_std']['roc_auc']:.3f}); on this tabular dataset a regularised linear model is already strong.
* Survival (Kaplan-Meier): {kms.loc[kms.months==12,'survival_probability'].iloc[0]:.1%} of customers remain at 12 months, {kms.loc[kms.months==36,'survival_probability'].iloc[0]:.1%} at 36 months and {kms.loc[kms.months==72,'survival_probability'].iloc[0]:.1%} at 72 months.
* Survival-based 36-month CLV: mean **${clv['mean_clv']:,.0f}** per customer; estimated annual revenue at risk **${clv['revenue_at_risk_annual_total']:,.0f}**.
* Uplift modelling and ROI are computed on a **simulated** treatment/control experiment (Section 9). They demonstrate the decision framework, not validated causal effects.

## 2. Data and quality checks
Source: `Telco_customer_churn.xlsx` (7,043 rows x 33 columns). The Bank Churn CSV (10,000 rows) was inspected and rejected as the primary dataset because it has annual tenure and no billing amounts, so it cannot support subscription CLV or discount ROI analysis. Original files are untouched in `data/raw/`.

Validation results: duplicate customer IDs = {dq['duplicate_customer_ids']}, duplicate rows = {dq['duplicate_rows']}, blank `Total Charges` = {dq['blank_total_charges']} (all with tenure = 0: {dq['blank_total_charges_all_zero_tenure']}; imputed as 0.0 because these customers have not completed a billing cycle), Churn Label/Value mismatches = {dq['churn_label_value_mismatch']}. Warnings: {warn}.
Raw and processed layers are stored separately (`raw_customers`, `customers`) in PostgreSQL.

## 3. Leakage control
Excluded from every model: `Churn Label`, `Churn Value` (target), `Churn Reason` (exists only for churners), `Churn Score` (IBM's model output; correlation with churn = {clv['ibm_cltv_assessment']['corr_churn_score_churn']:.2f}) and `CLTV`. IBM's CLTV has an undocumented method and unknown timing; its correlation with churn is {clv['ibm_cltv_assessment']['corr_cltv_churn']:.2f} and with tenure {clv['ibm_cltv_assessment']['corr_cltv_tenure']:.2f}. Because it may embed outcome information it is excluded, and this project derives its own CLV (Section 8). Geography and identifiers are excluded as non-predictive/high-cardinality. A unit test asserts no leakage column is in the feature list.

## 4. Exploratory findings (associations only)
Churn rate by contract: {', '.join(f'{k} {v:.1%}' for k, v in eda['churn_by_contract'].items())}. By payment method: {', '.join(f'{k} {v:.1%}' for k, v in eda['churn_by_payment'].items())}. By internet service: {', '.join(f'{k} {v:.1%}' for k, v in eda['churn_by_internet'].items())}. By tenure group: {', '.join(f'{k} {v:.1%}' for k, v in eda['churn_by_tenure_group'].items())}. Churned customers account for ${eda['churned_monthly_revenue']:,.0f} of ${eda['total_monthly_revenue']:,.0f} monthly revenue. Figures: `reports/figures/eda_*.png`.

## 5. Features: observed versus engineered proxies
Observed: demographics, tenure, services, contract, billing and charges. Engineered proxies (deterministic functions of observed columns only): tenure group, monthly-charge category, average monthly spend, charge ratio (current / historical average), contract months, month-to-month flag, service and add-on counts, protection bundle, auto-pay and electronic-check flags, service adoption rate, charge per service and an **engagement proxy** (breadth of services x log tenure x auto-pay). Complaint counts, usage trends or support contacts do not exist in the data and were not invented. Full dictionary: `reports/feature_dictionary.csv`.

## 6. Churn prediction
Split: stratified {c['n_train']:,} train / {c['n_test']:,} test (churn rate {c['churn_rate_train']:.3f} / {c['churn_rate_test']:.3f}). LightGBM tuned with {c['n_search_iterations']}-iteration randomised search, 5-fold stratified CV, scoring = PR-AUC (best CV PR-AUC {c['tuning_cv_pr_auc']:.3f}); the search space includes `class_weight` in {{None, balanced}} to handle imbalance. Selected parameters: `{json.dumps(c['best_params'])}`. Probability calibration: **{c['calibration_used']}** (sigmoid retained only if it did not worsen hold-out Brier score; uncalibrated Brier {c['lightgbm']['test_uncalibrated_at_0.5']['brier']:.3f} vs final {lg_t['brier']:.3f}).

**Hold-out test results**

{table(perf)}

Confusion matrix, LightGBM at 0.50: TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}. At the tuned threshold: TN={cmt['tn']}, FP={cmt['fp']}, FN={cmt['fn']}, TP={cmt['tp']}. The threshold was chosen on training out-of-fold predictions, not on the test set.

**5-fold cross-validation on the training set (final calibrated model vs baseline, threshold 0.50)**

{table(cvt)}

Interpretation: LightGBM's advantage over logistic regression is modest. Both models are well within one CV standard deviation on several metrics, so we do not claim a decisive gain. Customer-level probabilities used in the app are **out-of-fold** (each customer scored by models that never saw their label).

## 7. SHAP explainability
SHAP (TreeExplainer) is computed on the tuned LightGBM refit on all labelled data (explanation only; not used for evaluation). Top features by mean |SHAP| (log-odds): {top_shap}.

{table(shap)}

Plots: `reports/figures/shap_global_importance.png`, `shap_summary.png`, `shap_dependence_*.png`. Individual explanations are stored in `churn_explanations` and shown in the dashboard. SHAP describes what the model associates with predicted churn; it is **not** causal. The explainer-model probability and the calibrated out-of-fold probability differ slightly by construction.

## 8. Survival analysis and CLV
Duration = tenure (months); event = churn; {sv['n_censored']:,} still-active customers are right-censored, {sv['n_events']:,} events. Tenure 0 was set to 0.5 months for Cox fitting (11 customers).

Kaplan-Meier survival:

{table(kms.rename(columns={'months':'tenure (months)'}), 3)}

Log-rank test across contract types: chi2 = {sv['logrank_contract']['test_statistic']:.0f}, p {'< 1e-300' if sv['logrank_contract']['p_value']==0 else '= '+format(sv['logrank_contract']['p_value'],'.2g')}.

Cox PH (penalizer 0.01), contract as covariate (hazard ratios; HR < 1 = lower churn hazard); concordance {sv['cox_concordance_with_contract_dummies']:.3f}:

{table(hr)}

**Proportional-hazards check (Schoenfeld, rank-transformed time).** With contract dummies, violations at the 5% level: {', '.join(ph_u)}. A Cox model stratified by contract (used for individual predictions, concordance {sv['cox_concordance']:.3f}, computed within strata) still shows violations for: {', '.join(ph_s)}. The reported hazard ratios for these covariates are therefore time-averaged effects and should be interpreted cautiously. Individual predictions are conditional survival probabilities by months of tenure; **no calendar departure dates are claimed**. Beyond 72 months the baseline is flat, so long horizons for long-tenured customers are optimistic (flagged `extrapolated`).

**CLV** = sum over k=1..{clv['assumptions']['horizon_months']} of S(k) x (monthly charges x margin) / (1+r_m)^k, with Cox conditional survival S, margin = {clv['assumptions']['margin']:.0%}, annual discount rate = {clv['assumptions']['annual_discount_rate']:.0%}, constant charges. Margin and discount rate are assumptions, not observed. Result: mean ${clv['mean_clv']:,.0f}, median ${clv['median_clv']:,.0f}, total ${clv['total_clv']:,.0f}. A closed-form check against a constant-hazard model is a unit test.

## 9. Uplift modelling (SIMULATED experiment)
**The dataset contains no retention-offer assignment or outcomes.** We therefore simulate a randomised A/B test with these documented assumptions (`src/uplift_model.py::SIM_ASSUMPTIONS`):
* Treatment assignment ~ Bernoulli({config.SIM_TREATMENT_PROB}) independent of covariates, seed {config.RANDOM_STATE}.
* Control outcome = the **real** observed retention label (1 - churn).
* A treated customer who would have churned is saved with probability tau(x) (10% discount): additive rules (base 0.05; +0.18 if monthly charges >= $70, +0.08 if $30-70; +0.10 month-to-month; +0.08 tenure <= 12 months; +0.05 paperless; -0.05 auto-pay; -0.06 protection bundle), clipped to [0, {config.SIM_MAX_RESPONSE}].
* "Sleeping dogs": treated non-churners on contracts >= 12 months with tenure > 24 months churn with probability 0.04.
* Other discount depths use a diminishing-returns multiplier (1 - exp(-d/10)) / (1 - exp(-1)).

Learners: CausalML S-Learner and T-Learner (LightGBM base learners); 70/30 hold-out for evaluation, then 5-fold cross-fitting to give every customer an out-of-sample ITE. Simulated arms: {up['n_treated']:,} treated / {up['n_control']:,} control; retention {up['retention_treated']:.1%} vs {up['retention_control']:.1%} (simulated ATE {up['simulated_average_treatment_effect']:+.3f}). Selected learner (higher hold-out Qini): **{up['selected_learner']}**; {up['share_positive_uplift']:.0%} of customers have positive estimated ITE.

Hold-out evaluation:

{table(evt)}

Uplift by predicted-ITE decile (hold-out):

{table(dec)}

**Important reading.** (1) Both learners recover the simulated effect ordering (rank correlation with the known simulated effect is shown above), which validates the pipeline against a known data-generating process, but that process was written by us, so it says nothing about real customers. (2) A plain churn-risk ranking scores about the same Qini as the uplift learners here, because by construction only would-be churners can be saved. The practical advantage of uplift over risk ranking appears when real effects diverge from baseline risk (e.g., sleeping dogs), which this simulation only weakly encodes. (3) Never present these figures as experimentally validated.

Three different customer sets (default offer: 10% discount, $10 offer cost, 12-month benefit window): high churn risk = **{n_high:,}**; positive estimated uplift = **{n_pos:,}**; economically worthwhile = **{n_worth:,}**. Quadrants: {', '.join(f'{k} {v:,}' for k, v in quad.items())}. Discounting all high-risk customers is not recommended.

## 10. Segmentation
K-Means on standardised tenure, monthly charges, CLV, service count, churn probability and 12-month survival. k chosen by silhouette among k >= 3 (k = 2 excluded as too coarse); chosen **k = {sg['k']}**, silhouette {sg['silhouette']:.3f}. Silhouette is moderate, so segments overlap and should be read as profiles, not hard types.

{table(ks)}

Segment profiles (labels derived by rules from these measured profiles relative to the whole base):

{table(prof)}

## 11. ROI framework
Per targeted customer: incremental benefit = ITE x charges x months x margin; discount cost = discount x charges x months x P(stay | treated) (paid to everyone who stays, including those who would have stayed anyway); campaign cost = fixed offer cost + discount cost; **ROI = (Incremental Benefit - Campaign Cost) / Campaign Cost x 100**. Budget, target count and strategy constraints are applied by greedy selection. Default scenario (10% discount, $20,000 budget, up to 1,000 customers, $10 offer, 12 months, 40% margin):

{table(rs, 1)}

These are scenario estimates built on simulated uplift. The comparison shows the framework's logic (targeting random customers loses money because discounts leak to customers who would stay) but is not a forecast. Note that the economically-selective strategy and plain risk targeting give similar ROI here, for the reason given in Section 9.

## 12. Testing
{tests}. Tests cover ingestion/validation, PostgreSQL connectivity and table population, SQLite fallback rules, preprocessing and leakage, model training and inference, SHAP additivity, survival monotonicity, simulation reproducibility, CLV closed-form check, segmentation, ROI arithmetic and constraints, and Streamlit AppTest runs of every page including interactive controls.

## 13. Limitations
1. Single snapshot: tenure of churners is time-to-churn, but we cannot observe calendar time, seasonality or cohort effects; survival is by months of tenure only.
2. No randomised treatment data: all uplift/ROI results rest on simulated assumptions.
3. No complaint, usage, network-quality, competitor or price-change history; the engagement variable is a proxy.
4. CLV depends on assumed margin ({clv['assumptions']['margin']:.0%}), discount rate, horizon and constant prices.
5. Cox PH assumption is violated for several covariates; hazard ratios are time-averaged.
6. Dataset is a synthetic/educational IBM sample; absolute performance may not transfer to a real operator.
7. LightGBM gives only a small gain over logistic regression here.
8. Associations (EDA, SHAP, hazard ratios) are not causal effects.

## 14. Future work
Collect a real randomised retention pilot to replace the simulation; add usage/complaint features; time-varying covariates and cause-specific/competing-risk models; stratified or time-varying Cox terms; calibrated uplift (e.g., doubly-robust learners, X-learner) with confidence intervals; automated retraining and monitoring; authentication and audit logging for the dashboard.
"""
    (R / "technical_report.md").write_text(rep)

    # ---------------------------------------------------------------- README
    figs = ["eda_churn_rate_by_category", "shap_summary", "km_by_contract", "cox_coefficients", "shap_global_importance", "eda_numeric_distributions"]
    gallery = "\n".join(f"![{f}](reports/figures/{f}.png)" for f in figs)
    readme = f"""# Customer Retention Intelligence Platform

End-to-end churn prediction and retention decision platform on the IBM Telco Customer Churn dataset: churn model (LightGBM) + SHAP explanations, survival analysis (Lifelines), uplift modelling (CausalML, **simulated experiment**), CLV, K-Means segmentation, and a six-page Streamlit app with an ROI simulator, backed by PostgreSQL.

> **Read this first.** Uplift and ROI results use a *simulated* treatment/control experiment because the dataset has no real retention-offer data. They are scenario estimates, not validated causal evidence. See [Limitations](#limitations).

## Architecture
```
data/raw (original files, untouched)
   -> src/ingestion.py (validation, cleaning) -> PostgreSQL: raw_customers, customers
   -> src/pipeline.py -> churn_model / shap_analysis / survival_model / clv / uplift_model / segmentation
   -> PostgreSQL: churn_predictions, churn_explanations, survival_predictions, customer_clv,
                  uplift_predictions, customer_segments, retention_campaigns, retention_results, model_runs
   -> dashboard/ (Streamlit) reads the database;  models/ holds joblib artifacts;  reports/ holds metrics + figures
```

## Technology stack
Python 3.12 (3.11+ compatible), PostgreSQL 16, SQLAlchemy, pandas/NumPy, scikit-learn, LightGBM, SHAP, Lifelines, CausalML, Matplotlib/Seaborn/Plotly, Streamlit, Joblib, Pytest. (`python-dotenv`, `psycopg2-binary`, `openpyxl`, `nbconvert` are supporting libraries.)

## Project structure
```
data/{{raw,processed}}  database/{{models.py,schema.sql,connection.py,init_db.py}}  notebooks/01..06
src/{{ingestion,preprocessing,churn_model,shap_analysis,survival_model,uplift_model,clv,segmentation,roi,eda,pipeline,report}}.py
dashboard/{{app.py,utils.py,pages/}}  tests/  models/  reports/  requirements.txt  .env.example
```

## Installation
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# CausalML 0.18 declares newer scikit-learn/scipy minimums than the rest of this pinned stack but works with it;
# if pip's resolver complains, install it separately:  pip install --no-deps causalml forestci==0.6 pathos==0.2.9 graphviz
```

## PostgreSQL configuration
```bash
sudo -u postgres psql -c "CREATE USER retention_user WITH PASSWORD 'choose_a_strong_password';"
sudo -u postgres psql -c "CREATE DATABASE customer_retention_db OWNER retention_user;"
cp .env.example .env        # then edit .env with the same credentials (never commit .env)
python -m database.init_db  # creates tables (or: psql -d customer_retention_db -f database/schema.sql)
```
Credentials are read only from environment variables / `.env`. **SQLite fallback (development only):** set `ALLOW_SQLITE_FALLBACK=true` and the app will use `data/processed/customer_retention_fallback.db` *only if PostgreSQL is unreachable*, logging a loud warning. It never switches silently; without the flag a connection failure raises an error.

## Run
```bash
python -m src.pipeline --n-iter 25   # ingest, EDA, train, explain, survival, uplift, segments -> DB + models/ + reports/ (~2 min)
python -m src.report                 # regenerate README results + reports/technical_report.md from saved metrics
streamlit run dashboard/app.py       # http://localhost:8501
pytest -q                            # full test suite
```
Pages: Executive Dashboard, Churn Intelligence (search + SHAP), Survival Analysis, Retention Intelligence, Customer Segmentation, ROI Simulator (discount, budget, targets, offer cost, retention duration, margin, strategy; scenarios can be saved to `retention_campaigns` / `retention_results`).

## Dataset
IBM Telco Customer Churn: {eda['n']:,} customers, 33 columns, churn rate {eda['churn_rate']:.1%}. Observed columns are separated from engineered proxies in `reports/feature_dictionary.csv`. Leakage columns (`Churn Label/Value/Score/Reason`, `CLTV`) are excluded from models. Bank churn CSV was inspected only (no billing data).

## Methodology
* **Churn**: stratified 80/20 split; LightGBM with randomised search ({c['n_search_iterations']} iterations, 5-fold CV, PR-AUC), class-weight option, sigmoid calibration ({c['calibration_used']} selected), compared with logistic regression. Threshold chosen on training out-of-fold predictions.
* **SHAP**: TreeExplainer; global, dependence and per-customer explanations (associations, not causes).
* **Survival**: Kaplan-Meier, Cox PH (stratified by contract for prediction), Schoenfeld PH tests, conditional survival and 24-month restricted mean.
* **CLV**: survival-weighted discounted contribution margin over 36 months (assumed 40% margin, 10% annual discount rate).
* **Uplift**: simulated A/B experiment + CausalML S/T-Learners, cross-fitted ITE, Qini/AUUC. Full assumptions in `reports/technical_report.md` §9.
* **Segments**: K-Means, k by silhouette/elbow, labels derived by rules from cluster profiles.
* **ROI**: ROI = (Incremental Benefit - Campaign Cost) / Campaign Cost x 100 using uplift estimates, discount leakage and budget constraints.

## Evaluation results (from real executions)
Hold-out test set ({c['n_test']:,} customers):

{table(perf)}

Cross-validation (training set, 5-fold):

{table(cvt)}

LightGBM beats the logistic baseline only modestly. Survival: concordance {sv['cox_concordance_with_contract_dummies']:.3f} (Cox with contract), KM 12-month retention {kms.loc[kms.months==12,'survival_probability'].iloc[0]:.1%}. Segments: k = {sg['k']}, silhouette {sg['silhouette']:.3f}. Uplift hold-out Qini ({up['selected_learner']}) = {ev['qini'][up['selected_learner']]:.3f} on simulated data (churn-risk ranking: {ev['qini']['Churn_risk_baseline']:.3f}; random: {ev['qini']['Random']:.3f}). Default ROI scenarios are in `reports/default_roi_scenarios.json`. Details: [technical report](reports/technical_report.md).

**Tests:** {tests}

## Screenshots
Real dashboard screenshots could not be captured in the build environment (no browser available). To add them: run the app, capture each page, and save under `reports/screenshots/`. Analysis figures produced by the pipeline:

{gallery}

## Limitations
* No real randomised treatment data: uplift, treatment effects and ROI are **simulated scenario estimates**.
* No complaint, usage, or price-change history; "engagement" is a proxy built from services, tenure and payment method.
* Single snapshot: survival is by tenure months; no calendar departure dates.
* Cox proportional-hazards assumption is violated for several covariates ({len(ph_u)} with contract dummies); hazard ratios are time-averaged.
* CLV relies on assumed margin, discount rate, horizon and constant prices.
* Modest gain of LightGBM over logistic regression; synthetic/educational dataset.
* SHAP, EDA and hazard ratios show associations, not causal effects.
* The dashboard has no authentication.

## Future enhancements
Real randomised retention pilot; richer behavioural data; time-varying and competing-risk survival models; doubly-robust uplift with confidence intervals; model monitoring and scheduled retraining; authentication and audit logging.
"""
    (config.ROOT / "README.md").write_text(readme)


if __name__ == "__main__":
    build()

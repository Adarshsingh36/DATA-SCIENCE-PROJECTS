# Technical Report: Customer Churn Prediction and Retention Intelligence Platform

All numbers below are produced by `python -m src.pipeline` and read from `reports/*.json|csv`; none are hand-entered.
Backend used for this run: **postgresql**. Pipeline runtime: 83.7 s.

## 1. Executive summary
* Dataset: IBM Telco Customer Churn, 7,043 customers, actual churn rate **26.5%**.
* Churn model (LightGBM, calibrated) reaches hold-out ROC-AUC **0.857** and PR-AUC **0.684** versus 0.850 / 0.670 for a logistic-regression baseline. The improvement is real but small (cross-validated ROC-AUC 0.863 vs 0.858, fold s.d. about 0.015); on this tabular dataset a regularised linear model is already strong.
* Survival (Kaplan-Meier): 84.3% of customers remain at 12 months, 74.9% at 36 months and 59.3% at 72 months.
* Survival-based 36-month CLV: mean **$620** per customer; estimated annual revenue at risk **$1,665,605**.
* Uplift modelling and ROI are computed on a **simulated** treatment/control experiment (Section 9). They demonstrate the decision framework, not validated causal effects.

## 2. Data and quality checks
Source: `Telco_customer_churn.xlsx` (7,043 rows x 33 columns). The Bank Churn CSV (10,000 rows) was inspected and rejected as the primary dataset because it has annual tenure and no billing amounts, so it cannot support subscription CLV or discount ROI analysis. Original files are untouched in `data/raw/`.

Validation results: duplicate customer IDs = 0, duplicate rows = 0, blank `Total Charges` = 11 (all with tenure = 0: True; imputed as 0.0 because these customers have not completed a billing cycle), Churn Label/Value mismatches = 0. Warnings: 11 blank Total Charges (all tenure=0: True); imputed as 0.0; 2 rows where total_charges differs >50% from tenure*monthly (plausible: plan changes over time); kept as observed..
Raw and processed layers are stored separately (`raw_customers`, `customers`) in PostgreSQL.

## 3. Leakage control
Excluded from every model: `Churn Label`, `Churn Value` (target), `Churn Reason` (exists only for churners), `Churn Score` (IBM's model output; correlation with churn = 0.66) and `CLTV`. IBM's CLTV has an undocumented method and unknown timing; its correlation with churn is -0.13 and with tenure 0.40. Because it may embed outcome information it is excluded, and this project derives its own CLV (Section 8). Geography and identifiers are excluded as non-predictive/high-cardinality. A unit test asserts no leakage column is in the feature list.

## 4. Exploratory findings (associations only)
Churn rate by contract: Month-to-month 42.7%, One year 11.3%, Two year 2.8%. By payment method: Bank transfer (automatic) 16.7%, Credit card (automatic) 15.2%, Electronic check 45.3%, Mailed check 19.1%. By internet service: DSL 19.0%, Fiber optic 41.9%, No 7.4%. By tenure group: 0-12m 47.4%, 13-24m 28.7%, 25-48m 20.4%, 49-72m 9.5%. Churned customers account for $139,131 of $456,117 monthly revenue. Figures: `reports/figures/eda_*.png`.

## 5. Features: observed versus engineered proxies
Observed: demographics, tenure, services, contract, billing and charges. Engineered proxies (deterministic functions of observed columns only): tenure group, monthly-charge category, average monthly spend, charge ratio (current / historical average), contract months, month-to-month flag, service and add-on counts, protection bundle, auto-pay and electronic-check flags, service adoption rate, charge per service and an **engagement proxy** (breadth of services x log tenure x auto-pay). Complaint counts, usage trends or support contacts do not exist in the data and were not invented. Full dictionary: `reports/feature_dictionary.csv`.

## 6. Churn prediction
Split: stratified 5,634 train / 1,409 test (churn rate 0.265 / 0.265). LightGBM tuned with 25-iteration randomised search, 5-fold stratified CV, scoring = PR-AUC (best CV PR-AUC 0.681); the search space includes `class_weight` in {None, balanced} to handle imbalance. Selected parameters: `{"subsample_freq": 1, "subsample": 0.85, "reg_lambda": 0.0, "reg_alpha": 0.0, "num_leaves": 7, "n_estimators": 600, "min_child_samples": 80, "max_depth": 5, "learning_rate": 0.01, "colsample_bytree": 1.0, "class_weight": "balanced"}`. Probability calibration: **sigmoid** (sigmoid retained only if it did not worsen hold-out Brier score; uncalibrated Brier 0.159 vs final 0.132).

**Hold-out test results**

| model | accuracy | precision | recall | f1 | roc_auc | pr_auc | brier |
|---|---|---|---|---|---|---|---|
| Logistic regression (baseline), thr 0.50 | 0.798 | 0.641 | 0.540 | 0.586 | 0.850 | 0.670 | 0.134 |
| LightGBM, thr 0.50 | 0.803 | 0.641 | 0.583 | 0.611 | 0.857 | 0.684 | 0.132 |
| Logistic regression, thr 0.36 | 0.779 | 0.565 | 0.717 | 0.632 | 0.850 | 0.670 | 0.134 |
| LightGBM, thr 0.37 (F1-optimal on train OOF) | 0.789 | 0.583 | 0.722 | 0.645 | 0.857 | 0.684 | 0.132 |

Confusion matrix, LightGBM at 0.50: TN=913, FP=122, FN=156, TP=218. At the tuned threshold: TN=842, FP=193, FN=104, TP=270. The threshold was chosen on training out-of-fold predictions, not on the test set.

**5-fold cross-validation on the training set (final calibrated model vs baseline, threshold 0.50)**

| metric | LightGBM CV (mean ± sd) | Baseline CV (mean ± sd) |
|---|---|---|
| roc_auc | 0.863 ± 0.015 | 0.858 ± 0.015 |
| pr_auc | 0.682 ± 0.035 | 0.678 ± 0.035 |
| f1 | 0.622 ± 0.013 | 0.616 ± 0.020 |
| recall | 0.584 ± 0.010 | 0.568 ± 0.023 |
| precision | 0.666 ± 0.026 | 0.673 ± 0.024 |
| accuracy | 0.811 ± 0.009 | 0.812 ± 0.010 |

Interpretation: LightGBM's advantage over logistic regression is modest. Both models are well within one CV standard deviation on several metrics, so we do not claim a decisive gain. Customer-level probabilities used in the app are **out-of-fold** (each customer scored by models that never saw their label).

## 7. SHAP explainability
SHAP (TreeExplainer) is computed on the tuned LightGBM refit on all labelled data (explanation only; not used for evaluation). Top features by mean |SHAP| (log-odds): `contract_months`, `dependents`, `tenure_months`, `internet_service_Fiber optic`, `electronic_check`.

| feature | mean_abs_shap |
|---|---|
| contract_months | 0.883 |
| dependents | 0.570 |
| tenure_months | 0.425 |
| internet_service_Fiber optic | 0.220 |
| electronic_check | 0.163 |
| has_internet | 0.156 |
| paperless_billing | 0.091 |
| monthly_charges | 0.075 |
| streaming_movies | 0.070 |
| streaming_tv | 0.065 |

Plots: `reports/figures/shap_global_importance.png`, `shap_summary.png`, `shap_dependence_*.png`. Individual explanations are stored in `churn_explanations` and shown in the dashboard. SHAP describes what the model associates with predicted churn; it is **not** causal. The explainer-model probability and the calibrated out-of-fold probability differ slightly by construction.

## 8. Survival analysis and CLV
Duration = tenure (months); event = churn; 5,174 still-active customers are right-censored, 1,869 events. Tenure 0 was set to 0.5 months for Cox fitting (11 customers).

Kaplan-Meier survival:

| tenure (months) | survival_probability |
|---|---|
| 6 | 0.885 |
| 12 | 0.843 |
| 24 | 0.789 |
| 36 | 0.749 |
| 48 | 0.709 |
| 60 | 0.664 |
| 72 | 0.593 |

Log-rank test across contract types: chi2 = 2353, p < 1e-300.

Cox PH (penalizer 0.01), contract as covariate (hazard ratios; HR < 1 = lower churn hazard); concordance 0.869:

| covariate | hazard_ratio | hr_lower_95 | hr_upper_95 | p_value |
|---|---|---|---|---|
| pay_electronic_check | 1.732 | 1.522 | 1.970 | 0.000 |
| pay_mailed_check | 1.668 | 1.421 | 1.957 | 0.000 |
| internet_fiber | 1.347 | 1.096 | 1.656 | 0.005 |
| paperless_billing | 1.203 | 1.080 | 1.340 | 0.001 |
| monthly_charges_per10usd | 1.028 | 0.963 | 1.097 | 0.405 |
| phone_service | 1.023 | 0.814 | 1.285 | 0.846 |
| streaming_tv | 0.925 | 0.818 | 1.045 | 0.210 |
| senior_citizen | 0.894 | 0.803 | 0.996 | 0.042 |
| pay_credit_card_auto | 0.883 | 0.750 | 1.041 | 0.138 |
| streaming_movies | 0.834 | 0.737 | 0.943 | 0.004 |
| device_protection | 0.709 | 0.635 | 0.791 | 0.000 |
| partner | 0.689 | 0.624 | 0.761 | 0.000 |
| tech_support | 0.652 | 0.573 | 0.741 | 0.000 |
| multiple_lines | 0.633 | 0.569 | 0.705 | 0.000 |
| online_backup | 0.543 | 0.487 | 0.606 | 0.000 |
| online_security | 0.537 | 0.472 | 0.611 | 0.000 |
| internet_none | 0.390 | 0.301 | 0.505 | 0.000 |
| dependents | 0.370 | 0.308 | 0.444 | 0.000 |
| contract_one_year | 0.238 | 0.203 | 0.278 | 0.000 |
| contract_two_year | 0.081 | 0.064 | 0.103 | 0.000 |

**Proportional-hazards check (Schoenfeld, rank-transformed time).** With contract dummies, violations at the 5% level: contract_one_year, dependents, multiple_lines, online_backup, online_security, partner, pay_electronic_check, pay_mailed_check, streaming_movies, streaming_tv, tech_support. A Cox model stratified by contract (used for individual predictions, concordance 0.729, computed within strata) still shows violations for: dependents, multiple_lines, online_backup, online_security, partner, pay_electronic_check, pay_mailed_check, streaming_movies, streaming_tv, tech_support. The reported hazard ratios for these covariates are therefore time-averaged effects and should be interpreted cautiously. Individual predictions are conditional survival probabilities by months of tenure; **no calendar departure dates are claimed**. Beyond 72 months the baseline is flat, so long horizons for long-tenured customers are optimistic (flagged `extrapolated`).

**CLV** = sum over k=1..36 of S(k) x (monthly charges x margin) / (1+r_m)^k, with Cox conditional survival S, margin = 40%, annual discount rate = 10%, constant charges. Margin and discount rate are assumptions, not observed. Result: mean $620, median $598, total $4,369,584. A closed-form check against a constant-hazard model is a unit test.

## 9. Uplift modelling (SIMULATED experiment)
**The dataset contains no retention-offer assignment or outcomes.** We therefore simulate a randomised A/B test with these documented assumptions (`src/uplift_model.py::SIM_ASSUMPTIONS`):
* Treatment assignment ~ Bernoulli(0.5) independent of covariates, seed 42.
* Control outcome = the **real** observed retention label (1 - churn).
* A treated customer who would have churned is saved with probability tau(x) (10% discount): additive rules (base 0.05; +0.18 if monthly charges >= $70, +0.08 if $30-70; +0.10 month-to-month; +0.08 tenure <= 12 months; +0.05 paperless; -0.05 auto-pay; -0.06 protection bundle), clipped to [0, 0.45].
* "Sleeping dogs": treated non-churners on contracts >= 12 months with tenure > 24 months churn with probability 0.04.
* Other discount depths use a diminishing-returns multiplier (1 - exp(-d/10)) / (1 - exp(-1)).

Learners: CausalML S-Learner and T-Learner (LightGBM base learners); 70/30 hold-out for evaluation, then 5-fold cross-fitting to give every customer an out-of-sample ITE. Simulated arms: 3,471 treated / 3,572 control; retention 81.8% vs 73.4% (simulated ATE +0.084). Selected learner (higher hold-out Qini): **S_learner**; 75% of customers have positive estimated ITE.

Hold-out evaluation:

| ranking | Qini | AUUC | Spearman vs simulated true effect |
|---|---|---|---|
| S_learner | 0.285 | 0.795 | 0.917 |
| T_learner | 0.184 | 0.696 | 0.771 |
| Churn_risk_baseline | 0.282 | 0.787 | 0.969 |
| Random | 0.052 | 0.550 | -0.008 |

Uplift by predicted-ITE decile (hold-out):

| decile | n | mean_predicted_ite | retention_treated | retention_control | observed_uplift |
|---|---|---|---|---|---|
| 1 | 212 | 0.264 | 0.620 | 0.375 | 0.245 |
| 2 | 211 | 0.211 | 0.606 | 0.382 | 0.223 |
| 3 | 211 | 0.144 | 0.667 | 0.537 | 0.130 |
| 4 | 211 | 0.082 | 0.800 | 0.681 | 0.119 |
| 5 | 212 | 0.035 | 0.866 | 0.810 | 0.056 |
| 6 | 211 | 0.012 | 0.879 | 0.786 | 0.093 |
| 7 | 211 | 0.003 | 0.967 | 0.916 | 0.051 |
| 8 | 211 | -0.000 | 0.927 | 0.970 | -0.043 |
| 9 | 211 | -0.003 | 0.956 | 0.969 | -0.013 |
| 10 | 212 | -0.014 | 0.876 | 0.950 | -0.073 |

**Important reading.** (1) Both learners recover the simulated effect ordering (rank correlation with the known simulated effect is shown above), which validates the pipeline against a known data-generating process, but that process was written by us, so it says nothing about real customers. (2) A plain churn-risk ranking scores about the same Qini as the uplift learners here, because by construction only would-be churners can be saved. The practical advantage of uplift over risk ranking appears when real effects diverge from baseline risk (e.g., sleeping dogs), which this simulation only weakly encodes. (3) Never present these figures as experimentally validated.

Three different customer sets (default offer: 10% discount, $10 offer cost, 12-month benefit window): high churn risk = **1,648**; positive estimated uplift = **5,291**; economically worthwhile = **1,224**. Quadrants: Persuadable 3,557, Sure thing 3,337, Sleeping dog 149. Discounting all high-risk customers is not recommended.

## 10. Segmentation
K-Means on standardised tenure, monthly charges, CLV, service count, churn probability and 12-month survival. k chosen by silhouette among k >= 3 (k = 2 excluded as too coarse); chosen **k = 4**, silhouette 0.351. Silhouette is moderate, so segments overlap and should be read as profiles, not hard types.

| k | inertia | silhouette |
|---|---|---|
| 2 | 28576.059 | 0.299 |
| 3 | 20492.566 | 0.328 |
| 4 | 15853.183 | 0.351 |
| 5 | 14001.769 | 0.307 |
| 6 | 12294.167 | 0.304 |
| 7 | 11028.266 | 0.293 |
| 8 | 10130.575 | 0.293 |

Segment profiles (labels derived by rules from these measured profiles relative to the whole base):

| cluster_id | n_customers | tenure_months | monthly_charges | clv | service_count | churn_probability | surv_prob_12m | actual_churn_rate | segment_label |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 2226 | 27.110 | 28.440 | 280.690 | 1.790 | 0.130 | 0.850 | 0.130 | Low-value customers |
| 1 | 1705 | 53.930 | 88.420 | 980.550 | 6.600 | 0.130 | 0.920 | 0.120 | High-value loyal |
| 2 | 2158 | 11.460 | 76.930 | 799.790 | 3.970 | 0.570 | 0.880 | 0.570 | High-value at-risk |
| 3 | 954 | 53.400 | 79.710 | 363.710 | 5.670 | 0.160 | 0.370 | 0.170 | Mid-value, low-survival (watch list) |

## 11. ROI framework
Per targeted customer: incremental benefit = ITE x charges x months x margin; discount cost = discount x charges x months x P(stay | treated) (paid to everyone who stays, including those who would have stayed anyway); campaign cost = fixed offer cost + discount cost; **ROI = (Incremental Benefit - Campaign Cost) / Campaign Cost x 100**. Budget, target count and strategy constraints are applied by greedy selection. Default scenario (10% discount, $20,000 budget, up to 1,000 customers, $10 offer, 12 months, 40% margin):

| strategy | n_targeted | expected_customers_retained | incremental_benefit | discount_cost | campaign_cost | net_incremental_benefit | roi_pct |
|---|---|---|---|---|---|---|---|
| Uplift-ranked, economically worthwhile only | 316 | 88.6 | 36173.4 | 16767.5 | 19927.5 | 16245.9 | 81.5 |
| Uplift-ranked (ignore economics) | 315 | 90.9 | 35123.2 | 16803.0 | 19953.0 | 15170.3 | 76.0 |
| Highest churn risk | 351 | 89.7 | 35598.9 | 16414.7 | 19924.7 | 15674.2 | 78.7 |
| Random | 284 | 22.1 | 7882.6 | 17155.7 | 19995.7 | -12113.0 | -60.6 |

These are scenario estimates built on simulated uplift. The comparison shows the framework's logic (targeting random customers loses money because discounts leak to customers who would stay) but is not a forecast. Note that the economically-selective strategy and plain risk targeting give similar ROI here, for the reason given in Section 9.

## 12. Testing
54 passed in 54.13s. Tests cover ingestion/validation, PostgreSQL connectivity and table population, SQLite fallback rules, preprocessing and leakage, model training and inference, SHAP additivity, survival monotonicity, simulation reproducibility, CLV closed-form check, segmentation, ROI arithmetic and constraints, and Streamlit AppTest runs of every page including interactive controls.

## 13. Limitations
1. Single snapshot: tenure of churners is time-to-churn, but we cannot observe calendar time, seasonality or cohort effects; survival is by months of tenure only.
2. No randomised treatment data: all uplift/ROI results rest on simulated assumptions.
3. No complaint, usage, network-quality, competitor or price-change history; the engagement variable is a proxy.
4. CLV depends on assumed margin (40%), discount rate, horizon and constant prices.
5. Cox PH assumption is violated for several covariates; hazard ratios are time-averaged.
6. Dataset is a synthetic/educational IBM sample; absolute performance may not transfer to a real operator.
7. LightGBM gives only a small gain over logistic regression here.
8. Associations (EDA, SHAP, hazard ratios) are not causal effects.

## 14. Future work
Collect a real randomised retention pilot to replace the simulation; add usage/complaint features; time-varying covariates and cause-specific/competing-risk models; stratified or time-varying Cox terms; calibrated uplift (e.g., doubly-robust learners, X-learner) with confidence intervals; automated retraining and monitoring; authentication and audit logging for the dashboard.

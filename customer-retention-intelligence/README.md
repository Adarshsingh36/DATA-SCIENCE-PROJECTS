# Customer Retention Intelligence Platform

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
data/{raw,processed}  database/{models.py,schema.sql,connection.py,init_db.py}  notebooks/01..06
src/{ingestion,preprocessing,churn_model,shap_analysis,survival_model,uplift_model,clv,segmentation,roi,eda,pipeline,report}.py
dashboard/{app.py,utils.py,pages/}  tests/  models/  reports/  requirements.txt  .env.example
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
IBM Telco Customer Churn: 7,043 customers, 33 columns, churn rate 26.5%. Observed columns are separated from engineered proxies in `reports/feature_dictionary.csv`. Leakage columns (`Churn Label/Value/Score/Reason`, `CLTV`) are excluded from models. Bank churn CSV was inspected only (no billing data).

## Methodology
* **Churn**: stratified 80/20 split; LightGBM with randomised search (25 iterations, 5-fold CV, PR-AUC), class-weight option, sigmoid calibration (sigmoid selected), compared with logistic regression. Threshold chosen on training out-of-fold predictions.
* **SHAP**: TreeExplainer; global, dependence and per-customer explanations (associations, not causes).
* **Survival**: Kaplan-Meier, Cox PH (stratified by contract for prediction), Schoenfeld PH tests, conditional survival and 24-month restricted mean.
* **CLV**: survival-weighted discounted contribution margin over 36 months (assumed 40% margin, 10% annual discount rate).
* **Uplift**: simulated A/B experiment + CausalML S/T-Learners, cross-fitted ITE, Qini/AUUC. Full assumptions in `reports/technical_report.md` §9.
* **Segments**: K-Means, k by silhouette/elbow, labels derived by rules from cluster profiles.
* **ROI**: ROI = (Incremental Benefit - Campaign Cost) / Campaign Cost x 100 using uplift estimates, discount leakage and budget constraints.

## Evaluation results (from real executions)
Hold-out test set (1,409 customers):

| model | accuracy | precision | recall | f1 | roc_auc | pr_auc | brier |
|---|---|---|---|---|---|---|---|
| Logistic regression (baseline), thr 0.50 | 0.798 | 0.641 | 0.540 | 0.586 | 0.850 | 0.670 | 0.134 |
| LightGBM, thr 0.50 | 0.803 | 0.641 | 0.583 | 0.611 | 0.857 | 0.684 | 0.132 |
| Logistic regression, thr 0.36 | 0.779 | 0.565 | 0.717 | 0.632 | 0.850 | 0.670 | 0.134 |
| LightGBM, thr 0.37 (F1-optimal on train OOF) | 0.789 | 0.583 | 0.722 | 0.645 | 0.857 | 0.684 | 0.132 |

Cross-validation (training set, 5-fold):

| metric | LightGBM CV (mean ± sd) | Baseline CV (mean ± sd) |
|---|---|---|
| roc_auc | 0.863 ± 0.015 | 0.858 ± 0.015 |
| pr_auc | 0.682 ± 0.035 | 0.678 ± 0.035 |
| f1 | 0.622 ± 0.013 | 0.616 ± 0.020 |
| recall | 0.584 ± 0.010 | 0.568 ± 0.023 |
| precision | 0.666 ± 0.026 | 0.673 ± 0.024 |
| accuracy | 0.811 ± 0.009 | 0.812 ± 0.010 |

LightGBM beats the logistic baseline only modestly. Survival: concordance 0.869 (Cox with contract), KM 12-month retention 84.3%. Segments: k = 4, silhouette 0.351. Uplift hold-out Qini (S_learner) = 0.285 on simulated data (churn-risk ranking: 0.282; random: 0.052). Default ROI scenarios are in `reports/default_roi_scenarios.json`. Details: [technical report](reports/technical_report.md).

**Tests:** 54 passed in 54.13s

## Screenshots
Real dashboard screenshots could not be captured in the build environment (no browser available). To add them: run the app, capture each page, and save under `reports/screenshots/`. Analysis figures produced by the pipeline:

![eda_churn_rate_by_category](reports/figures/eda_churn_rate_by_category.png)
![shap_summary](reports/figures/shap_summary.png)
![km_by_contract](reports/figures/km_by_contract.png)
![cox_coefficients](reports/figures/cox_coefficients.png)
![shap_global_importance](reports/figures/shap_global_importance.png)
![eda_numeric_distributions](reports/figures/eda_numeric_distributions.png)

## Limitations
* No real randomised treatment data: uplift, treatment effects and ROI are **simulated scenario estimates**.
* No complaint, usage, or price-change history; "engagement" is a proxy built from services, tenure and payment method.
* Single snapshot: survival is by tenure months; no calendar departure dates.
* Cox proportional-hazards assumption is violated for several covariates (11 with contract dummies); hazard ratios are time-averaged.
* CLV relies on assumed margin, discount rate, horizon and constant prices.
* Modest gain of LightGBM over logistic regression; synthetic/educational dataset.
* SHAP, EDA and hazard ratios show associations, not causal effects.
* The dashboard has no authentication.

## Future enhancements
Real randomised retention pilot; richer behavioural data; time-varying and competing-risk survival models; doubly-robust uplift with confidence intervals; model monitoring and scheduled retraining; authentication and audit logging.

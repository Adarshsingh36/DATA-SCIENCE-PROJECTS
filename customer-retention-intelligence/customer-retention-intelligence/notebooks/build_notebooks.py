"""Generates the six analysis notebooks (execute with: jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb)."""
import nbformat as nbf
from pathlib import Path

HDR = "import sys, warnings; sys.path.insert(0, '..'); warnings.filterwarnings('ignore')\nimport pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns\nfrom IPython.display import display, Image\nfrom src import config\npd.set_option('display.width', 160); sns.set_theme(style='whitegrid')\n"
LOAD = "df = pd.read_csv(config.DATA_PROCESSED / 'customers_processed.csv')   # cleaned + engineered (written by ingestion)\n"

def nb(name, cells):
    n = nbf.v4.new_notebook()
    n["cells"] = [nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s) for k, s in cells]
    nbf.write(n, Path(__file__).with_name(name))

nb("01_eda.ipynb", [
 ("md", "# 01 · Exploratory data analysis\nIBM Telco Customer Churn (primary dataset). Observed fields vs engineered proxies are listed in `reports/feature_dictionary.csv`."),
 ("code", HDR + "from src.ingestion import load_raw_excel, validate, bank_dataset_summary\nraw = load_raw_excel(); rep = validate(raw)\nprint(rep.to_dict()); print(bank_dataset_summary())"),
 ("md", "The bank dataset was inspected only: it has annual tenure and no billing amounts, so it cannot support the subscription-revenue analyses (CLV, ROI). Telco is used."),
 ("code", LOAD + "print(df.shape, 'churn rate', round(df.churn.mean(),4))\ndisplay(df[['tenure_months','monthly_charges','total_charges']].describe().round(2))"),
 ("code", "from src import eda\ns = eda.run_eda(df)\nfor k in ['churn_by_contract','churn_by_payment','churn_by_internet','churn_by_tenure_group']: print(k, s[k])"),
 ("code", "for f in ['eda_churn_distribution','eda_numeric_distributions','eda_churn_rate_by_category','eda_demographics','eda_addons','eda_charges_by_contract','eda_correlation']:\n    display(Image(str(config.FIGURES_DIR / (f + '.png'))))"),
 ("md", "**Reading:** churn concentrates in month-to-month contracts, fiber-optic internet, electronic-check payers and the first year of tenure. These are associations in a single snapshot, not causal effects."),
])
nb("02_feature_engineering.ipynb", [
 ("md", "# 02 · Feature engineering and leakage review"),
 ("code", HDR + "from src import preprocessing as pp\nfrom src.ingestion import load_raw_excel\n" + LOAD + "display(pp.feature_dictionary())"),
 ("md", "## Leakage assessment of IBM's precomputed fields"),
 ("code", "import json; a = pp.assess_cltv_leakage(pp.clean_raw(load_raw_excel())); print(json.dumps(a, indent=2))"),
 ("md", "`Churn Score` correlates strongly with the target (it is a model output); `Churn Reason` exists only for churners. Both, plus `CLTV` (opaque methodology), are excluded from all models. This project derives its own CLV."),
 ("code", "X = pp.model_frame(df); print(X.shape, 'NaNs:', int(X.isna().sum().sum()))\nassert not set(X.columns) & set(pp.LEAKAGE_COLUMNS)\nXt = pp.build_preprocessor().fit_transform(X); print('encoded matrix', Xt.shape)"),
 ("code", "display(df.groupby('churn')[pp.ENGINEERED_NUMERIC].mean().T.round(3))"),
])
nb("03_churn_prediction.ipynb", [
 ("md", "# 03 · Churn prediction (LightGBM vs logistic-regression baseline)\nFinal results (25-iteration search) are in `reports/churn_model_metrics.json`; this notebook re-runs a shorter search (`n_iter=8`, not saved) to demonstrate the workflow."),
 ("code", HDR + "import json\nfrom src import churn_model as cm\n" + LOAD + "res = cm.train_and_evaluate(df, n_iter=8, save=False); obj = res.pop('_objects')\nfor k in ['baseline_logreg','lightgbm']:\n    print(k); print(pd.DataFrame({'cv_mean': res[k]['cv_mean'], 'cv_std': res[k]['cv_std']}).round(3))\nprint('best params', res['best_params'], '| calibration:', res['calibration_used'])"),
 ("code", "rows = {n: {m: v for m, v in res[k][s].items() if m != 'confusion_matrix'} for n, k, s in [('LogReg @0.5','baseline_logreg','test_at_0.5'),('LightGBM @0.5','lightgbm','test_at_0.5'),('LightGBM @tuned thr','lightgbm','test_at_tuned_threshold')]}\ndisplay(pd.DataFrame(rows).T.round(3))\nprint(res['lightgbm']['test_at_0.5']['confusion_matrix'])"),
 ("code", "from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay\nfig, ax = plt.subplots(1, 3, figsize=(16, 4.5)); y, p, b = obj['y_test'], obj['proba_test'], obj['base_proba_test']\nRocCurveDisplay.from_predictions(y, p, name='LightGBM', ax=ax[0]); RocCurveDisplay.from_predictions(y, b, name='LogReg', ax=ax[0])\nPrecisionRecallDisplay.from_predictions(y, p, name='LightGBM', ax=ax[1]); PrecisionRecallDisplay.from_predictions(y, b, name='LogReg', ax=ax[1])\nConfusionMatrixDisplay.from_predictions(y, (p >= 0.5).astype(int), ax=ax[2]); plt.tight_layout()"),
 ("code", "c = res['calibration_curve']; plt.plot(c['prob_pred'], c['prob_true'], 'o-'); plt.plot([0,1],[0,1],'--'); plt.xlabel('predicted'); plt.ylabel('observed'); plt.title('Calibration (hold-out)')"),
 ("md", "## Saved final model (25-iteration search)"),
 ("code", "final = json.load(open(config.REPORTS_DIR / 'churn_model_metrics.json'))\nprint({k: round(v,3) for k, v in final['lightgbm']['test_at_0.5'].items() if k != 'confusion_matrix'})"),
 ("md", "## SHAP explanations\nSHAP values describe what the model associates with churn; they are not causal."),
 ("code", "from src import shap_analysis as sa\nimport joblib\npipe = joblib.load(config.MODELS_DIR / 'churn_lgbm_explainer.joblib')\nsdf, Xt, base = sa.compute_shap(pipe, df)\ndisplay(sa.global_importance(sdf).head(12).round(3))\nfor f in ['shap_summary','shap_global_importance']: display(Image(str(config.FIGURES_DIR / (f + '.png'))))"),
 ("code", "i = int(np.argmax(df.churn.values)); print(df.customer_id[i]); display(sa.top_factors(sdf.iloc[i], Xt.iloc[i]).round(3))"),
])
nb("04_survival_analysis.ipynb", [
 ("md", "# 04 · Survival analysis (lifelines)\nDuration = `tenure_months`; event = `churn`; active customers are right-censored."),
 ("code", HDR + "from src import survival_model as sm\n" + LOAD + "print('events', df.churn.sum(), 'censored', (1-df.churn).sum())\nkm = sm.fit_kaplan_meier(df); display(sm.km_summary(km).round(3))\nprint(sm.logrank_by_group(df, 'contract'))"),
 ("code", "fig, ax = plt.subplots(1, 2, figsize=(14, 4.5)); km.plot_survival_function(ax=ax[0])\nfor k in sm.km_by_group(df, 'contract').values(): k.plot_survival_function(ax=ax[1])\nax[0].set_title('Overall'); ax[1].set_title('By contract'); plt.tight_layout()"),
 ("code", "cph_u, d_u = sm.fit_cox(df, stratify_contract=False)\nprint('concordance', round(cph_u.concordance_index_, 3)); display(sm.hazard_ratios(cph_u).round(3))"),
 ("code", "ph = sm.check_ph_assumption(cph_u, d_u); display(ph.round(4))\nprint('violations:', ph.loc[ph.violates_ph_at_5pct, 'covariate'].tolist())"),
 ("md", "Several covariates violate proportional hazards. A Cox model stratified by contract (used for predictions) removes the contract assumption, but other violations remain, so hazard ratios are time-averaged effects."),
 ("code", "cph, d = sm.fit_cox(df, stratify_contract=True); print('stratified concordance', round(cph.concordance_index_, 3))\nsp = sm.survival_predictions(cph, df); display(sp.describe().round(3))"),
 ("code", "r = df.sort_values('monthly_charges').tail(1); display(sm.customer_curve(cph, r).set_index('months_from_now').T.iloc[:, ::6].round(3))"),
])
nb("05_uplift_modelling.ipynb", [
 ("md", "# 05 · Uplift modelling (CausalML) on a SIMULATED experiment\n**The dataset has no real retention-offer data.** Treatment assignment and treated outcomes are simulated; control outcomes are the real retention labels. Results are not experimentally validated causal evidence."),
 ("code", HDR + "import json\nfrom src import uplift_model as um\n" + LOAD + "p0 = pd.read_csv(config.DATA_PROCESSED / 'churn_scores.csv').set_index('customer_id').loc[df.customer_id, 'churn_probability'].values\nprint(json.dumps(um.SIM_ASSUMPTIONS, indent=2))"),
 ("code", "out = um.run_uplift(df, p0, save=False); s = out['summary']; t = out['table']\nprint({k: s[k] for k in ['n_treated','n_control','retention_treated','retention_control','simulated_average_treatment_effect','selected_learner','share_positive_uplift']})\ndisplay(pd.DataFrame(s['evaluation_holdout']).round(3))"),
 ("code", "display(pd.DataFrame(s['uplift_deciles_holdout_selected']).round(3)); display(t.quadrant.value_counts())"),
 ("code", "from causalml.metrics import plot_qini\nh = pd.DataFrame(out['holdout_curves'])\nplot_qini(h, outcome_col='y', treatment_col='w'); plt.title('Qini curves (hold-out, simulated experiment)')"),
 ("md", "## Likely to churn ≠ likely to respond ≠ worth the money"),
 ("code", "u = um.add_economics(t, df, p0)\nhi = p0 >= 0.5\nprint('high risk:', hi.sum(), '| positive uplift:', int(u.positive_uplift.sum()), '| worthwhile at default offer:', int(u.economically_worthwhile.sum()))\nprint('high-risk but NOT worthwhile:', int((hi & ~u.economically_worthwhile.values).sum()))"),
 ("md", "In this simulation, ranking by churn risk performs about as well as the uplift learners (see Qini above) because only would-be churners can be saved by construction. Uplift modelling adds value when real treatment effects and baseline risk diverge (e.g., sleeping dogs)."),
])
nb("06_customer_segmentation.ipynb", [
 ("md", "# 06 · CLV and customer segmentation"),
 ("code", HDR + "from sqlalchemy import text\nfrom database.connection import get_engine\nfrom src import segmentation as seg, clv as clvmod\nfrom dashboard.utils import QUERY\nm = pd.read_sql(text(QUERY), get_engine()); m.columns = [str(c) for c in m.columns]\nprint(m.shape); display(m[['clv','revenue_at_risk_annual']].describe().round(1))"),
 ("md", "CLV = Σₖ S(k)·(monthly charges × margin)/(1+r)ᵏ over a 36-month horizon with Cox conditional survival S, margin = 40%, annual discount rate = 10% (assumptions; see `src/clv.py`). IBM's CLTV is not used."),
 ("code", "r = seg.run_segmentation(m[['customer_id','tenure_months','monthly_charges','service_count','churn','clv','churn_probability','surv_prob_12m']], save=False)\ndisplay(r['k_scores'].round(3)); print('chosen k', r['k'], 'silhouette', round(r['silhouette'], 3))"),
 ("code", "fig, ax = plt.subplots(1, 2, figsize=(13, 4)); ax[0].plot(r['k_scores'].k, r['k_scores'].inertia, 'o-'); ax[0].set_title('Elbow'); ax[1].plot(r['k_scores'].k, r['k_scores'].silhouette, 'o-'); ax[1].set_title('Silhouette'); plt.tight_layout()"),
 ("code", "display(r['profile'].round(2))\ns = r['segments']; sns.scatterplot(data=s.sample(3000, random_state=0), x='pc1', y='pc2', hue='segment_label', alpha=.6); plt.legend(bbox_to_anchor=(1,1))"),
 ("md", "Segment names are produced by rules applied to each cluster's mean CLV, churn probability, tenure and 12-month survival relative to the whole base, not assigned by hand."),
])

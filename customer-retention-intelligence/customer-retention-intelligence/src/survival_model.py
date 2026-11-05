"""Survival analysis with lifelines: Kaplan-Meier, Cox PH (with assumption checks), individual predictions.

Duration = tenure_months (observed), event = churn (1 = churned/observed; 0 = right-censored: still a customer
at the snapshot). The data are a single snapshot, so we can estimate survival by months of tenure, but we cannot
give calendar departure dates for any individual.
"""
from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import proportional_hazard_test, multivariate_logrank_test

from src import config

DURATION, EVENT = "tenure_months", "churn"
MAX_OBSERVED_MONTHS = 72
BINARY_COVARIATES = ["senior_citizen", "partner", "dependents", "phone_service", "multiple_lines", "online_security",
                     "online_backup", "device_protection", "tech_support", "streaming_tv", "streaming_movies",
                     "paperless_billing"]
MODEL_VERSION = "cox-v1"


def cox_frame(df: pd.DataFrame, include_contract_dummies: bool = False) -> pd.DataFrame:
    """Design matrix. Contract is kept as a raw column (for stratification) unless dummies are requested."""
    out = pd.DataFrame(index=df.index)
    out[DURATION] = df[DURATION].astype(float).clip(lower=0.5)   # 11 brand-new customers have tenure 0
    out[EVENT] = df[EVENT].astype(int)
    for c in BINARY_COVARIATES:
        out[c] = df[c].astype(int)
    out["monthly_charges_per10usd"] = df["monthly_charges"] / 10.0
    out["internet_fiber"] = (df["internet_service"] == "Fiber optic").astype(int)
    out["internet_none"] = (df["internet_service"] == "No").astype(int)
    for pm, name in [("Electronic check", "pay_electronic_check"), ("Mailed check", "pay_mailed_check"),
                     ("Credit card (automatic)", "pay_credit_card_auto")]:   # reference: bank transfer (automatic)
        out[name] = (df["payment_method"] == pm).astype(int)
    if include_contract_dummies:
        out["contract_one_year"] = (df["contract"] == "One year").astype(int)
        out["contract_two_year"] = (df["contract"] == "Two year").astype(int)
    else:
        out["contract"] = df["contract"].values
    return out


def fit_kaplan_meier(df: pd.DataFrame, label: str = "All customers") -> KaplanMeierFitter:
    km = KaplanMeierFitter()
    km.fit(df[DURATION], event_observed=df[EVENT], label=label)
    return km


def km_by_group(df: pd.DataFrame, col: str) -> dict:
    out = {}
    for g, sub in df.groupby(col):
        out[str(g)] = fit_kaplan_meier(sub, label=str(g))
    return out


def logrank_by_group(df: pd.DataFrame, col: str) -> dict:
    r = multivariate_logrank_test(df[DURATION], df[col], df[EVENT])
    return {"test_statistic": float(r.test_statistic), "p_value": float(r.p_value), "df": int(r.degrees_of_freedom)}


def km_summary(km: KaplanMeierFitter, times=(6, 12, 24, 36, 48, 60, 72)) -> pd.DataFrame:
    rows = [{"months": t, "survival_probability": float(km.predict(t))} for t in times]
    return pd.DataFrame(rows)


def fit_cox(df: pd.DataFrame, stratify_contract: bool = True, penalizer: float = 0.01):
    """Returns (fitted CoxPHFitter, design frame used)."""
    d = cox_frame(df, include_contract_dummies=not stratify_contract)
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(d, duration_col=DURATION, event_col=EVENT, strata=["contract"] if stratify_contract else None)
    return cph, d


def hazard_ratios(cph: CoxPHFitter) -> pd.DataFrame:
    s = cph.summary.reset_index()
    out = s[["covariate", "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].copy()
    out.columns = ["covariate", "hazard_ratio", "hr_lower_95", "hr_upper_95", "p_value"]
    return out.sort_values("hazard_ratio", ascending=False).reset_index(drop=True)


def check_ph_assumption(cph: CoxPHFitter, design: pd.DataFrame, time_transform: str = "rank") -> pd.DataFrame:
    """Schoenfeld-residual based test of proportional hazards for every covariate."""
    res = proportional_hazard_test(cph, design, time_transform=time_transform)
    t = res.summary.reset_index()
    t.columns = ["covariate"] + list(t.columns[1:])
    t["violates_ph_at_5pct"] = t["p"] < 0.05
    return t[["covariate", "test_statistic", "p", "violates_ph_at_5pct"]]


def time_grid(horizon: int = 96) -> np.ndarray:
    return np.arange(0, horizon + 1, dtype=float)


def conditional_survival_matrix(cph: CoxPHFitter, df: pd.DataFrame, horizon: int = 36, stratified: bool = True) -> np.ndarray:
    """P(customer still active at tenure t+k | active at tenure t), k = 1..horizon. Shape (n, horizon).

    Beyond the observed 72-month maximum tenure the Cox baseline is flat (no extrapolated churn) -> flagged.
    """
    d = cox_frame(df, include_contract_dummies=not stratified).drop(columns=[DURATION, EVENT])
    grid = time_grid(int(df[DURATION].max()) + horizon + 1)
    S = cph.predict_survival_function(d, times=grid)          # index=times, columns=customers
    S = S.values.T                                              # (n, T)
    t0 = np.clip(df[DURATION].astype(int).values, 0, len(grid) - 1)
    rows = np.arange(len(df))
    s_now = np.clip(S[rows, t0], 1e-9, None)
    out = np.empty((len(df), horizon))
    for k in range(1, horizon + 1):
        idx = np.clip(t0 + k, 0, len(grid) - 1)
        out[:, k - 1] = np.minimum(S[rows, idx] / s_now, 1.0)
    return out


def survival_predictions(cph: CoxPHFitter, df: pd.DataFrame, stratified: bool = True) -> pd.DataFrame:
    M = conditional_survival_matrix(cph, df, horizon=36, stratified=stratified)
    d = cox_frame(df, include_contract_dummies=not stratified).drop(columns=[DURATION, EVENT])
    return pd.DataFrame({
        "customer_id": df["customer_id"].values,
        "partial_hazard": cph.predict_partial_hazard(d).values,
        "surv_prob_12m": M[:, 11], "surv_prob_24m": M[:, 23], "surv_prob_36m": M[:, 35],
        "expected_months_next_24": M[:, :24].sum(axis=1),        # restricted mean survival over the next 24 months
        "extrapolated": (df[DURATION].values + 24) > MAX_OBSERVED_MONTHS,
        "model_version": MODEL_VERSION,
    })


def customer_curve(cph: CoxPHFitter, row: pd.DataFrame, stratified: bool = True, horizon: int = 36) -> pd.DataFrame:
    """Conditional survival curve (months from now) for one customer row (1-row DataFrame)."""
    M = conditional_survival_matrix(cph, row.reset_index(drop=True), horizon=horizon, stratified=stratified)[0]
    return pd.DataFrame({"months_from_now": np.arange(0, horizon + 1), "survival_probability": np.r_[1.0, M]})


def save_cox(cph, path=config.MODELS_DIR / "cox_model.joblib"):
    joblib.dump(cph, path)


def load_cox(path=config.MODELS_DIR / "cox_model.joblib"):
    return joblib.load(path)

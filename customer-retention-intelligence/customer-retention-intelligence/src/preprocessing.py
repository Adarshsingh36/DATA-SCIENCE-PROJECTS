"""Cleaning, feature engineering and sklearn preprocessing.

OBSERVED features come straight from the source file. ENGINEERED features are deterministic
proxies computed from observed columns only; nothing is invented (no complaint counts, usage trends, etc.).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

# ---------- raw -> snake_case ----------
RAW_RENAME = {
    "CustomerID": "customer_id", "Count": "count", "Country": "country", "State": "state", "City": "city",
    "Zip Code": "zip_code", "Lat Long": "lat_long", "Latitude": "latitude", "Longitude": "longitude",
    "Gender": "gender", "Senior Citizen": "senior_citizen", "Partner": "partner", "Dependents": "dependents",
    "Tenure Months": "tenure_months", "Phone Service": "phone_service", "Multiple Lines": "multiple_lines",
    "Internet Service": "internet_service", "Online Security": "online_security",
    "Online Backup": "online_backup", "Device Protection": "device_protection", "Tech Support": "tech_support",
    "Streaming TV": "streaming_tv", "Streaming Movies": "streaming_movies", "Contract": "contract",
    "Paperless Billing": "paperless_billing", "Payment Method": "payment_method",
    "Monthly Charges": "monthly_charges", "Total Charges": "total_charges", "Churn Label": "churn_label",
    "Churn Value": "churn_value", "Churn Score": "churn_score", "CLTV": "cltv", "Churn Reason": "churn_reason",
}

# ---------- leakage policy ----------
# churn_label/churn_value: the target itself. churn_reason: only exists for customers who already churned.
# churn_score: IBM's precomputed model score (target-derived). cltv: IBM's precomputed value; see assess_cltv_leakage().
LEAKAGE_COLUMNS = ["churn_label", "churn_value", "churn_score", "churn_reason", "cltv"]
NON_PREDICTIVE_COLUMNS = ["customer_id", "count", "country", "state", "city", "zip_code", "lat_long",
                          "latitude", "longitude"]

YES_NO_COLS = ["senior_citizen", "partner", "dependents", "phone_service", "multiple_lines", "online_security",
               "online_backup", "device_protection", "tech_support", "streaming_tv", "streaming_movies",
               "paperless_billing"]
ADDON_COLS = ["online_security", "online_backup", "device_protection", "tech_support", "streaming_tv",
              "streaming_movies"]
SERVICE_COLS = ["phone_service", "multiple_lines", "has_internet"] + ADDON_COLS

OBSERVED_NUMERIC = ["tenure_months", "monthly_charges", "total_charges"]
OBSERVED_BINARY = ["senior_citizen", "partner", "dependents", "phone_service", "multiple_lines", "online_security",
                   "online_backup", "device_protection", "tech_support", "streaming_tv", "streaming_movies",
                   "paperless_billing"]
OBSERVED_CATEGORICAL = ["gender", "internet_service", "contract", "payment_method"]

ENGINEERED_NUMERIC = ["avg_monthly_spend", "charge_ratio", "contract_months", "is_month_to_month", "service_count",
                      "addon_count", "has_internet", "has_protection_bundle", "auto_pay", "electronic_check",
                      "service_adoption_rate", "charge_per_service", "engagement_proxy"]
ENGINEERED_CATEGORICAL = ["tenure_group", "monthly_charge_category"]

NUMERIC_FEATURES = OBSERVED_NUMERIC + OBSERVED_BINARY + ENGINEERED_NUMERIC
CATEGORICAL_FEATURES = OBSERVED_CATEGORICAL + ENGINEERED_CATEGORICAL
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TENURE_BINS = [-1, 12, 24, 48, 72]
TENURE_LABELS = ["0-12m", "13-24m", "25-48m", "49-72m"]
CHARGE_BINS = [0, 35, 70, 90, np.inf]
CHARGE_LABELS = ["Low", "Medium", "High", "Premium"]
CONTRACT_MONTHS = {"Month-to-month": 1, "One year": 12, "Two year": 24}


def clean_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, fix types and handle missing values (documented decisions)."""
    df = raw.rename(columns=RAW_RENAME).copy()
    df["customer_id"] = df["customer_id"].astype(str).str.strip()
    # Total Charges: 11 blank strings, all with tenure == 0 (customer has not completed a billing cycle).
    tc = pd.to_numeric(df["total_charges"].astype(str).str.strip().replace({"": np.nan}), errors="coerce")
    new_cust = df["tenure_months"].eq(0)
    tc = tc.mask(tc.isna() & new_cust, 0.0)
    if tc.isna().any():  # anything else missing: impute with tenure * monthly (documented fallback)
        tc = tc.fillna(df["tenure_months"] * df["monthly_charges"])
    df["total_charges"] = tc.astype(float)
    for c in YES_NO_COLS:  # 'No internet service' / 'No phone service' -> 0; captured by has_internet / phone_service
        df[c] = df[c].astype(str).str.strip().eq("Yes").astype(int)
    df["churn"] = df["churn_value"].astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered proxies. Requires cleaned columns."""
    out = df.copy()
    ten = out["tenure_months"]
    out["tenure_group"] = pd.cut(ten, TENURE_BINS, labels=TENURE_LABELS).astype(str)
    out["monthly_charge_category"] = pd.cut(out["monthly_charges"], CHARGE_BINS, labels=CHARGE_LABELS).astype(str)
    out["avg_monthly_spend"] = np.where(ten > 0, out["total_charges"] / ten.clip(lower=1), out["monthly_charges"])
    out["charge_ratio"] = out["monthly_charges"] / out["avg_monthly_spend"].replace(0, np.nan)
    out["charge_ratio"] = out["charge_ratio"].fillna(1.0)
    out["contract_months"] = out["contract"].map(CONTRACT_MONTHS).astype(int)
    out["is_month_to_month"] = (out["contract"] == "Month-to-month").astype(int)
    out["has_internet"] = (out["internet_service"] != "No").astype(int)
    out["addon_count"] = out[ADDON_COLS].sum(axis=1)
    out["service_count"] = out[SERVICE_COLS].sum(axis=1)
    out["has_protection_bundle"] = ((out["online_security"] == 1) & (out["tech_support"] == 1)).astype(int)
    out["auto_pay"] = out["payment_method"].str.contains("automatic", case=False).astype(int)
    out["electronic_check"] = out["payment_method"].eq("Electronic check").astype(int)
    out["service_adoption_rate"] = np.where(out["has_internet"] == 1, out["addon_count"] / len(ADDON_COLS), 0.0)
    out["charge_per_service"] = out["monthly_charges"] / out["service_count"].clip(lower=1)
    # Engagement PROXY (not observed engagement): breadth of products x relationship length x autopay stickiness
    out["engagement_proxy"] = (out["service_count"] / len(SERVICE_COLS)) * np.log1p(ten) * (1 + 0.25 * out["auto_pay"])
    return out


def prepare_customers(raw: pd.DataFrame) -> pd.DataFrame:
    return engineer_features(clean_raw(raw))


def assess_cltv_leakage(raw_clean: pd.DataFrame) -> dict:
    """Evidence used to decide whether IBM's precomputed CLTV may be a predictor."""
    d = raw_clean
    return {
        "corr_cltv_churn": float(d["cltv"].corr(d["churn"])),
        "corr_cltv_churn_score": float(d["cltv"].corr(d["churn_score"])),
        "corr_churn_score_churn": float(d["churn_score"].corr(d["churn"])),
        "corr_cltv_tenure": float(d["cltv"].corr(d["tenure_months"])),
        "corr_cltv_monthly_charges": float(d["cltv"].corr(d["monthly_charges"])),
        "decision": "EXCLUDED: IBM's CLTV is a precomputed, undocumented-methodology field of unknown timing "
                    "(it may embed churn/survival information). It is kept only as a reference column, never as "
                    "a model input; this project derives its own transparent CLV (src/clv.py).",
    }


def model_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")
    leak = set(FEATURES) & set(LEAKAGE_COLUMNS + NON_PREDICTIVE_COLUMNS)
    assert not leak, f"Leakage/non-predictive columns in feature list: {leak}"
    return df[FEATURES].copy()


class IdentityNumeric(BaseEstimator, TransformerMixin):
    """Numeric passthrough that remembers column names (robust under sklearn pandas output)."""
    def fit(self, X, y=None):
        self.columns_ = list(X.columns) if hasattr(X, "columns") else [f"x{i}" for i in range(X.shape[1])]
        return self

    def transform(self, X):
        return X.values.astype(float) if hasattr(X, "values") else np.asarray(X, float)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns_)


def build_preprocessor() -> ColumnTransformer:
    ct = ColumnTransformer(
        [("num", IdentityNumeric(), NUMERIC_FEATURES),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)],
        verbose_feature_names_out=False)
    ct.set_output(transform="pandas")
    return ct


def feature_dictionary() -> pd.DataFrame:
    rows = []
    for c in OBSERVED_NUMERIC + OBSERVED_BINARY + OBSERVED_CATEGORICAL:
        rows.append((c, "observed", "source column (cleaned)"))
    desc = {
        "avg_monthly_spend": "total_charges / tenure (monthly_charges if tenure=0)",
        "charge_ratio": "current monthly_charges / avg_monthly_spend (>1 suggests recent price/plan increase)",
        "contract_months": "contract term length in months (1/12/24)",
        "is_month_to_month": "contract == Month-to-month",
        "service_count": "count of subscribed services (phone, multi-line, internet, 6 add-ons)",
        "addon_count": "count of 6 internet add-ons subscribed",
        "has_internet": "internet_service != No", "has_protection_bundle": "online security AND tech support",
        "auto_pay": "payment by automatic bank transfer / credit card",
        "electronic_check": "payment by electronic check",
        "service_adoption_rate": "addon_count / 6 for internet customers",
        "charge_per_service": "monthly_charges / service_count",
        "engagement_proxy": "(service_count/9) * ln(1+tenure) * (1+0.25*auto_pay): PROXY, not measured engagement",
        "tenure_group": "0-12m / 13-24m / 25-48m / 49-72m",
        "monthly_charge_category": "Low <35, Medium 35-70, High 70-90, Premium >90",
    }
    for c in ENGINEERED_NUMERIC + ENGINEERED_CATEGORICAL:
        rows.append((c, "engineered proxy", desc[c]))
    for c in LEAKAGE_COLUMNS:
        rows.append((c, "EXCLUDED (leakage)", "target or target-derived; never used as a feature"))
    return pd.DataFrame(rows, columns=["feature", "type", "description"])

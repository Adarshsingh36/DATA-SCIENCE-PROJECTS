"""SQLAlchemy schema (works on PostgreSQL; also creatable on the documented SQLite fallback)."""
from sqlalchemy import (MetaData, Table, Column, Integer, BigInteger, Float, String, Text,
                        Boolean, DateTime, ForeignKey, func)

metadata = MetaData()

# ---- RAW layer: exactly what is in the source file (snake_case names, Total Charges kept as text) ----
raw_customers = Table(
    "raw_customers", metadata,
    Column("customer_id", String(20), primary_key=True),
    Column("count", Integer), Column("country", String(50)), Column("state", String(50)),
    Column("city", String(100)), Column("zip_code", Integer), Column("lat_long", String(60)),
    Column("latitude", Float), Column("longitude", Float), Column("gender", String(10)),
    Column("senior_citizen", String(5)), Column("partner", String(5)), Column("dependents", String(5)),
    Column("tenure_months", Integer), Column("phone_service", String(5)),
    Column("multiple_lines", String(30)), Column("internet_service", String(30)),
    Column("online_security", String(30)), Column("online_backup", String(30)),
    Column("device_protection", String(30)), Column("tech_support", String(30)),
    Column("streaming_tv", String(30)), Column("streaming_movies", String(30)),
    Column("contract", String(30)), Column("paperless_billing", String(5)),
    Column("payment_method", String(40)), Column("monthly_charges", Float),
    Column("total_charges", String(20)), Column("churn_label", String(5)),
    Column("churn_value", Integer), Column("churn_score", Integer), Column("cltv", Integer),
    Column("churn_reason", String(100)),
)

# ---- PROCESSED layer: cleaned observed fields + engineered proxies (no leakage columns) ----
_customer_cols = [
    Column("customer_id", String(20), primary_key=True),
    Column("state", String(50)), Column("city", String(100)), Column("zip_code", Integer),
    Column("latitude", Float), Column("longitude", Float),
    Column("gender", String(10)), Column("senior_citizen", Integer), Column("partner", Integer),
    Column("dependents", Integer), Column("tenure_months", Integer), Column("phone_service", Integer),
    Column("multiple_lines", Integer), Column("internet_service", String(30)),
    Column("online_security", Integer), Column("online_backup", Integer),
    Column("device_protection", Integer), Column("tech_support", Integer),
    Column("streaming_tv", Integer), Column("streaming_movies", Integer),
    Column("contract", String(30)), Column("paperless_billing", Integer),
    Column("payment_method", String(40)), Column("monthly_charges", Float),
    Column("total_charges", Float), Column("churn", Integer, nullable=False),
    # engineered proxies
    Column("tenure_group", String(10)), Column("monthly_charge_category", String(10)),
    Column("avg_monthly_spend", Float), Column("charge_ratio", Float),
    Column("contract_months", Integer), Column("is_month_to_month", Integer),
    Column("service_count", Integer), Column("addon_count", Integer),
    Column("has_internet", Integer), Column("has_protection_bundle", Integer),
    Column("auto_pay", Integer), Column("electronic_check", Integer),
    Column("service_adoption_rate", Float), Column("charge_per_service", Float),
    Column("engagement_proxy", Float),
]
customers = Table("customers", metadata, *_customer_cols)

_fk = lambda: ForeignKey("customers.customer_id", ondelete="CASCADE")

churn_predictions = Table(
    "churn_predictions", metadata,
    Column("customer_id", String(20), _fk(), primary_key=True),
    Column("churn_probability", Float, nullable=False),
    Column("churn_probability_uncalibrated", Float),
    Column("risk_tier", String(10)), Column("predicted_label", Integer),
    Column("prediction_type", String(30)), Column("model_version", String(40)),
    Column("scored_at", DateTime, server_default=func.now()),
)

churn_explanations = Table(
    "churn_explanations", metadata,
    Column("customer_id", String(20), _fk(), primary_key=True),
    Column("base_value", Float), Column("shap_json", Text), Column("feature_json", Text),
)

survival_predictions = Table(
    "survival_predictions", metadata,
    Column("customer_id", String(20), _fk(), primary_key=True),
    Column("partial_hazard", Float),
    Column("surv_prob_12m", Float), Column("surv_prob_24m", Float), Column("surv_prob_36m", Float),
    Column("expected_months_next_24", Float), Column("extrapolated", Boolean),
    Column("model_version", String(40)),
)

uplift_predictions = Table(
    "uplift_predictions", metadata,
    Column("customer_id", String(20), _fk(), primary_key=True),
    Column("sim_treatment", Integer), Column("sim_retained", Integer),
    Column("ite_s_learner", Float), Column("ite_t_learner", Float),
    Column("ite_selected", Float), Column("selected_learner", String(20)),
    Column("reference_effect", Float), Column("positive_uplift", Boolean),
    Column("quadrant", String(20)), Column("expected_net_value", Float),
    Column("economically_worthwhile", Boolean), Column("is_simulated", Boolean, server_default="true"),
)

customer_clv = Table(
    "customer_clv", metadata,
    Column("customer_id", String(20), _fk(), primary_key=True),
    Column("clv", Float), Column("monthly_margin", Float), Column("revenue_at_risk_annual", Float),
)

customer_segments = Table(
    "customer_segments", metadata,
    Column("customer_id", String(20), _fk(), primary_key=True),
    Column("cluster_id", Integer), Column("segment_label", String(60)),
    Column("pc1", Float), Column("pc2", Float),
)

retention_campaigns = Table(
    "retention_campaigns", metadata,
    Column("campaign_id", Integer, primary_key=True, autoincrement=True),
    Column("name", String(100)), Column("strategy", String(80)),
    Column("discount_pct", Float), Column("budget", Float), Column("n_targeted_requested", Integer),
    Column("offer_cost_per_customer", Float), Column("retention_duration_months", Float),
    Column("created_at", DateTime, server_default=func.now()),
    Column("is_simulated_uplift", Boolean, server_default="true"),
)

retention_results = Table(
    "retention_results", metadata,
    Column("result_id", Integer, primary_key=True, autoincrement=True),
    Column("campaign_id", Integer, ForeignKey("retention_campaigns.campaign_id", ondelete="CASCADE")),
    Column("n_targeted", Integer), Column("baseline_churn_prob", Float),
    Column("incremental_retention_prob", Float), Column("expected_customers_retained", Float),
    Column("expected_retained_revenue", Float), Column("incremental_benefit", Float),
    Column("discount_cost", Float), Column("campaign_cost", Float),
    Column("net_incremental_benefit", Float), Column("roi_pct", Float),
)

model_runs = Table(
    "model_runs", metadata,
    Column("run_id", Integer, primary_key=True, autoincrement=True),
    Column("model_name", String(60)), Column("metric_name", String(60)),
    Column("metric_value", Float), Column("split", String(20)),
    Column("created_at", DateTime, server_default=func.now()),
)

# Order matters for truncation (children first)
TABLE_ORDER = ["retention_results", "retention_campaigns", "customer_segments", "customer_clv",
               "uplift_predictions", "survival_predictions", "churn_explanations",
               "churn_predictions", "customers", "raw_customers", "model_runs"]

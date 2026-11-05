-- PostgreSQL schema for customer_retention_db
-- Generated from database/models.py via `python -m database.export_schema`.
-- Create the database/user first (see README), then:  psql -d customer_retention_db -f database/schema.sql

CREATE TABLE IF NOT EXISTS customers (
	customer_id VARCHAR(20) NOT NULL, 
	state VARCHAR(50), 
	city VARCHAR(100), 
	zip_code INTEGER, 
	latitude FLOAT, 
	longitude FLOAT, 
	gender VARCHAR(10), 
	senior_citizen INTEGER, 
	partner INTEGER, 
	dependents INTEGER, 
	tenure_months INTEGER, 
	phone_service INTEGER, 
	multiple_lines INTEGER, 
	internet_service VARCHAR(30), 
	online_security INTEGER, 
	online_backup INTEGER, 
	device_protection INTEGER, 
	tech_support INTEGER, 
	streaming_tv INTEGER, 
	streaming_movies INTEGER, 
	contract VARCHAR(30), 
	paperless_billing INTEGER, 
	payment_method VARCHAR(40), 
	monthly_charges FLOAT, 
	total_charges FLOAT, 
	churn INTEGER NOT NULL, 
	tenure_group VARCHAR(10), 
	monthly_charge_category VARCHAR(10), 
	avg_monthly_spend FLOAT, 
	charge_ratio FLOAT, 
	contract_months INTEGER, 
	is_month_to_month INTEGER, 
	service_count INTEGER, 
	addon_count INTEGER, 
	has_internet INTEGER, 
	has_protection_bundle INTEGER, 
	auto_pay INTEGER, 
	electronic_check INTEGER, 
	service_adoption_rate FLOAT, 
	charge_per_service FLOAT, 
	engagement_proxy FLOAT, 
	PRIMARY KEY (customer_id)
);

CREATE TABLE IF NOT EXISTS model_runs (
	run_id SERIAL NOT NULL, 
	model_name VARCHAR(60), 
	metric_name VARCHAR(60), 
	metric_value FLOAT, 
	split VARCHAR(20), 
	created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT now(), 
	PRIMARY KEY (run_id)
);

CREATE TABLE IF NOT EXISTS raw_customers (
	customer_id VARCHAR(20) NOT NULL, 
	count INTEGER, 
	country VARCHAR(50), 
	state VARCHAR(50), 
	city VARCHAR(100), 
	zip_code INTEGER, 
	lat_long VARCHAR(60), 
	latitude FLOAT, 
	longitude FLOAT, 
	gender VARCHAR(10), 
	senior_citizen VARCHAR(5), 
	partner VARCHAR(5), 
	dependents VARCHAR(5), 
	tenure_months INTEGER, 
	phone_service VARCHAR(5), 
	multiple_lines VARCHAR(30), 
	internet_service VARCHAR(30), 
	online_security VARCHAR(30), 
	online_backup VARCHAR(30), 
	device_protection VARCHAR(30), 
	tech_support VARCHAR(30), 
	streaming_tv VARCHAR(30), 
	streaming_movies VARCHAR(30), 
	contract VARCHAR(30), 
	paperless_billing VARCHAR(5), 
	payment_method VARCHAR(40), 
	monthly_charges FLOAT, 
	total_charges VARCHAR(20), 
	churn_label VARCHAR(5), 
	churn_value INTEGER, 
	churn_score INTEGER, 
	cltv INTEGER, 
	churn_reason VARCHAR(100), 
	PRIMARY KEY (customer_id)
);

CREATE TABLE IF NOT EXISTS retention_campaigns (
	campaign_id SERIAL NOT NULL, 
	name VARCHAR(100), 
	strategy VARCHAR(80), 
	discount_pct FLOAT, 
	budget FLOAT, 
	n_targeted_requested INTEGER, 
	offer_cost_per_customer FLOAT, 
	retention_duration_months FLOAT, 
	created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT now(), 
	is_simulated_uplift BOOLEAN DEFAULT 'true', 
	PRIMARY KEY (campaign_id)
);

CREATE TABLE IF NOT EXISTS churn_explanations (
	customer_id VARCHAR(20) NOT NULL, 
	base_value FLOAT, 
	shap_json TEXT, 
	feature_json TEXT, 
	PRIMARY KEY (customer_id), 
	FOREIGN KEY(customer_id) REFERENCES customers (customer_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS churn_predictions (
	customer_id VARCHAR(20) NOT NULL, 
	churn_probability FLOAT NOT NULL, 
	churn_probability_uncalibrated FLOAT, 
	risk_tier VARCHAR(10), 
	predicted_label INTEGER, 
	prediction_type VARCHAR(30), 
	model_version VARCHAR(40), 
	scored_at TIMESTAMP WITHOUT TIME ZONE DEFAULT now(), 
	PRIMARY KEY (customer_id), 
	FOREIGN KEY(customer_id) REFERENCES customers (customer_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS customer_clv (
	customer_id VARCHAR(20) NOT NULL, 
	clv FLOAT, 
	monthly_margin FLOAT, 
	revenue_at_risk_annual FLOAT, 
	PRIMARY KEY (customer_id), 
	FOREIGN KEY(customer_id) REFERENCES customers (customer_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS customer_segments (
	customer_id VARCHAR(20) NOT NULL, 
	cluster_id INTEGER, 
	segment_label VARCHAR(60), 
	pc1 FLOAT, 
	pc2 FLOAT, 
	PRIMARY KEY (customer_id), 
	FOREIGN KEY(customer_id) REFERENCES customers (customer_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS retention_results (
	result_id SERIAL NOT NULL, 
	campaign_id INTEGER, 
	n_targeted INTEGER, 
	baseline_churn_prob FLOAT, 
	incremental_retention_prob FLOAT, 
	expected_customers_retained FLOAT, 
	expected_retained_revenue FLOAT, 
	incremental_benefit FLOAT, 
	discount_cost FLOAT, 
	campaign_cost FLOAT, 
	net_incremental_benefit FLOAT, 
	roi_pct FLOAT, 
	PRIMARY KEY (result_id), 
	FOREIGN KEY(campaign_id) REFERENCES retention_campaigns (campaign_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS survival_predictions (
	customer_id VARCHAR(20) NOT NULL, 
	partial_hazard FLOAT, 
	surv_prob_12m FLOAT, 
	surv_prob_24m FLOAT, 
	surv_prob_36m FLOAT, 
	expected_months_next_24 FLOAT, 
	extrapolated BOOLEAN, 
	model_version VARCHAR(40), 
	PRIMARY KEY (customer_id), 
	FOREIGN KEY(customer_id) REFERENCES customers (customer_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS uplift_predictions (
	customer_id VARCHAR(20) NOT NULL, 
	sim_treatment INTEGER, 
	sim_retained INTEGER, 
	ite_s_learner FLOAT, 
	ite_t_learner FLOAT, 
	ite_selected FLOAT, 
	selected_learner VARCHAR(20), 
	reference_effect FLOAT, 
	positive_uplift BOOLEAN, 
	quadrant VARCHAR(20), 
	expected_net_value FLOAT, 
	economically_worthwhile BOOLEAN, 
	is_simulated BOOLEAN DEFAULT 'true', 
	PRIMARY KEY (customer_id), 
	FOREIGN KEY(customer_id) REFERENCES customers (customer_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS ix_churn_pred_prob ON churn_predictions (churn_probability DESC);
CREATE INDEX IF NOT EXISTS ix_segments_cluster ON customer_segments (cluster_id);

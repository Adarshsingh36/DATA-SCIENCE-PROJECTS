import numpy as np
from src import preprocessing as pp


def test_total_charges_cleaned(df):
    assert df.total_charges.notna().all() and (df.loc[df.tenure_months == 0, "total_charges"] == 0).all()


def test_no_leakage_in_features():
    assert not set(pp.FEATURES) & set(pp.LEAKAGE_COLUMNS)
    assert not set(pp.FEATURES) & set(pp.NON_PREDICTIVE_COLUMNS)
    assert "cltv" in pp.LEAKAGE_COLUMNS and "churn_score" in pp.LEAKAGE_COLUMNS and "churn_reason" in pp.LEAKAGE_COLUMNS


def test_model_frame_shape_and_no_nans(df):
    X = pp.model_frame(df)
    assert list(X.columns) == pp.FEATURES and not X.isna().any().any()


def test_engineered_features(df):
    assert set(df.tenure_group.unique()) == set(pp.TENURE_LABELS)
    assert set(df.monthly_charge_category.unique()) <= set(pp.CHARGE_LABELS)
    assert (df.contract_months.isin([1, 12, 24])).all()
    assert (df.is_month_to_month == (df.contract == "Month-to-month")).all()
    assert df.service_count.between(0, 9).all() and df.addon_count.between(0, 6).all()
    new = df[df.tenure_months == 0]
    assert np.allclose(new.avg_monthly_spend, new.monthly_charges)
    old = df[df.tenure_months > 0]
    assert np.allclose(old.avg_monthly_spend, old.total_charges / old.tenure_months)


def test_preprocessor_output(df):
    pre = pp.build_preprocessor(); Xt = pre.fit_transform(pp.model_frame(df))
    assert Xt.shape[0] == len(df) and not Xt.isna().any().any() and Xt.shape[1] > len(pp.NUMERIC_FEATURES)
    # unseen category does not crash
    X2 = pp.model_frame(df).head(3).copy(); X2.loc[X2.index[0], "contract"] = "Unknown"
    assert pre.transform(X2).shape[0] == 3


def test_cltv_assessment_and_feature_dictionary(df):
    a = pp.assess_cltv_leakage(df)
    assert "EXCLUDED" in a["decision"]
    fd = pp.feature_dictionary()
    assert {"observed", "engineered proxy", "EXCLUDED (leakage)"} <= set(fd.type)

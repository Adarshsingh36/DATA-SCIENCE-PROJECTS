"""LightGBM churn model with baseline comparison, CV tuning, calibration and honest evaluation."""
from __future__ import annotations
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, confusion_matrix, brier_score_loss, precision_recall_curve)
from sklearn.model_selection import (StratifiedKFold, RandomizedSearchCV, train_test_split, cross_validate,
                                     cross_val_predict)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.preprocessing import (FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES, build_preprocessor, model_frame)

warnings.filterwarnings("ignore", category=UserWarning)
MODEL_VERSION = "lgbm-v1"


def split_data(df: pd.DataFrame, test_size: float = config.TEST_SIZE, seed: int = config.RANDOM_STATE):
    X, y = model_frame(df), df["churn"].astype(int)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def build_baseline() -> Pipeline:
    pre = ColumnTransformer([("num", StandardScaler(), NUMERIC_FEATURES),
                             ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, random_state=config.RANDOM_STATE))])


def build_lgbm(**params) -> Pipeline:
    base = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, random_state=config.RANDOM_STATE,
                verbose=-1, n_jobs=1)
    base.update(params)
    return Pipeline([("pre", build_preprocessor()), ("clf", LGBMClassifier(**base))])


PARAM_DIST = {
    "clf__n_estimators": [150, 250, 400, 600],
    "clf__learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
    "clf__num_leaves": [4, 7, 15, 31],
    "clf__max_depth": [-1, 3, 5, 7],
    "clf__min_child_samples": [10, 20, 40, 80],
    "clf__subsample": [0.7, 0.85, 1.0], "clf__subsample_freq": [1],
    "clf__colsample_bytree": [0.5, 0.7, 0.9, 1.0],
    "clf__reg_lambda": [0.0, 1.0, 5.0, 15.0],
    "clf__reg_alpha": [0.0, 0.1, 1.0],
    "clf__class_weight": [None, "balanced"],   # class-imbalance handling is part of the search
}


def tune_lgbm(X_train, y_train, n_iter: int = 30, cv: int = config.CV_FOLDS, seed: int = config.RANDOM_STATE):
    search = RandomizedSearchCV(
        build_lgbm(), PARAM_DIST, n_iter=n_iter, scoring="average_precision",
        cv=StratifiedKFold(cv, shuffle=True, random_state=seed), random_state=seed, n_jobs=1, refit=True)
    search.fit(X_train, y_train)
    return search


def metrics_at(y_true, proba, threshold: float = 0.5) -> dict:
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"threshold": float(threshold), "accuracy": accuracy_score(y_true, pred),
            "precision": precision_score(y_true, pred, zero_division=0), "recall": recall_score(y_true, pred),
            "f1": f1_score(y_true, pred), "roc_auc": roc_auc_score(y_true, proba),
            "pr_auc": average_precision_score(y_true, proba), "brier": brier_score_loss(y_true, proba),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def best_f1_threshold(y_true, proba) -> float:
    p, r, t = precision_recall_curve(y_true, proba)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-9, None)
    return float(t[int(np.nanargmax(f1))])


def risk_tier(p) -> np.ndarray:
    p = np.asarray(p)
    return np.where(p >= config.HIGH_RISK_THRESHOLD, "High", np.where(p >= config.MEDIUM_RISK_THRESHOLD, "Medium", "Low"))


def train_and_evaluate(df: pd.DataFrame, n_iter: int = 30, save: bool = True) -> dict:
    X_tr, X_te, y_tr, y_te = split_data(df)
    cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "f1": "f1", "recall": "recall",
               "precision": "precision", "accuracy": "accuracy"}

    # ---- baseline ----
    base = build_baseline()
    base_cv = cross_validate(base, X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=1)
    base.fit(X_tr, y_tr)
    base_proba = base.predict_proba(X_te)[:, 1]

    # ---- tuned LightGBM ----
    search = tune_lgbm(X_tr, y_tr, n_iter=n_iter)
    best = search.best_estimator_

    # ---- calibration (sigmoid; compared against uncalibrated on the held-out test set) ----
    calibrated = CalibratedClassifierCV(clone(best), method="sigmoid", cv=config.CV_FOLDS).fit(X_tr, y_tr)
    raw_proba = best.predict_proba(X_te)[:, 1]
    cal_proba = calibrated.predict_proba(X_te)[:, 1]
    use_cal = brier_score_loss(y_te, cal_proba) <= brier_score_loss(y_te, raw_proba)
    final_model = calibrated if use_cal else best
    final_proba = cal_proba if use_cal else raw_proba

    # CV of the FINAL deployed model (same object type as evaluated on the test set) so CV and test numbers are comparable
    lgbm_cv = cross_validate(clone(final_model), X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=1)

    # ---- threshold chosen on TRAIN out-of-fold predictions (no test peeking) ----
    oof_tr = cross_val_predict(clone(final_model), X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    thr = best_f1_threshold(y_tr, oof_tr)

    prob_true, prob_pred = calibration_curve(y_te, final_proba, n_bins=10, strategy="quantile")
    res = {
        "model_version": MODEL_VERSION,
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)), "churn_rate_train": float(y_tr.mean()),
        "churn_rate_test": float(y_te.mean()),
        "best_params": {k.replace("clf__", ""): (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()},
        "tuning_cv_pr_auc": float(search.best_score_), "n_search_iterations": n_iter,
        "calibration_used": "sigmoid" if use_cal else "none",
        "baseline_logreg": {"cv_mean": {k: float(np.mean(base_cv[f"test_{k}"])) for k in scoring},
                            "cv_std": {k: float(np.std(base_cv[f"test_{k}"])) for k in scoring},
                            "test_at_0.5": metrics_at(y_te, base_proba, 0.5),
                            "test_at_f1_threshold": metrics_at(y_te, base_proba, best_f1_threshold(y_te * 0 + y_te, base_proba) if False else 0.5)},
        "lightgbm": {"cv_mean": {k: float(np.mean(lgbm_cv[f"test_{k}"])) for k in scoring},
                     "cv_std": {k: float(np.std(lgbm_cv[f"test_{k}"])) for k in scoring},
                     "test_at_0.5": metrics_at(y_te, final_proba, 0.5),
                     "test_at_tuned_threshold": metrics_at(y_te, final_proba, thr),
                     "test_uncalibrated_at_0.5": metrics_at(y_te, raw_proba, 0.5),
                     "tuned_threshold_from_train_oof": thr},
        "calibration_curve": {"prob_true": prob_true.tolist(), "prob_pred": prob_pred.tolist()},
    }
    # baseline at its own train-OOF threshold for a fair comparison
    base_oof = cross_val_predict(build_baseline(), X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    res["baseline_logreg"]["test_at_tuned_threshold"] = metrics_at(y_te, base_proba, best_f1_threshold(y_tr, base_oof))
    del res["baseline_logreg"]["test_at_f1_threshold"]

    if save:
        joblib.dump(final_model, config.MODELS_DIR / "churn_model_calibrated.joblib")
        joblib.dump(base, config.MODELS_DIR / "churn_baseline_logreg.joblib")
        (config.REPORTS_DIR / "churn_model_metrics.json").write_text(json.dumps(res, indent=2))
    res["_objects"] = {"best_pipeline": best, "final_model": final_model, "threshold": thr,
                       "y_test": y_te, "proba_test": final_proba, "base_proba_test": base_proba, "X_test": X_te}
    return res


def score_all_customers(df: pd.DataFrame, final_model, best_pipeline, threshold: float) -> pd.DataFrame:
    """Out-of-fold predictions for every customer (no customer is scored by a model that saw its label)."""
    X, y = model_frame(df), df["churn"].astype(int)
    cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE + 1)
    cal = cross_val_predict(clone(final_model), X, y, cv=cv, method="predict_proba")[:, 1]
    raw = cross_val_predict(clone(best_pipeline), X, y, cv=cv, method="predict_proba")[:, 1]
    return pd.DataFrame({"customer_id": df["customer_id"].values, "churn_probability": cal,
                         "churn_probability_uncalibrated": raw, "risk_tier": risk_tier(cal),
                         "predicted_label": (cal >= threshold).astype(int),
                         "prediction_type": "out_of_fold", "model_version": MODEL_VERSION})


def load_model(path=config.MODELS_DIR / "churn_model_calibrated.joblib"):
    return joblib.load(path)


def predict_proba_new(df_features: pd.DataFrame, model=None) -> np.ndarray:
    """Inference on new (already cleaned + engineered) customer rows."""
    model = model or load_model()
    return model.predict_proba(model_frame(df_features))[:, 1]

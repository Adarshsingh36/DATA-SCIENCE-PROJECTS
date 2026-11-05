"""SHAP explainability for the LightGBM churn model.

SHAP values explain the model's log-odds output (feature contributions to the PREDICTION).
They are associations learned by the model, NOT causal effects.
"""
from __future__ import annotations
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone

from src import config
from src.preprocessing import model_frame


def fit_explainer_model(best_pipeline, df: pd.DataFrame):
    """Refit the tuned pipeline on all labelled data; explainability only (not used for evaluation)."""
    X, y = model_frame(df), df["churn"].astype(int)
    pipe = clone(best_pipeline).fit(X, y)
    joblib.dump(pipe, config.MODELS_DIR / "churn_lgbm_explainer.joblib")
    return pipe


def _positive_class(sv) -> np.ndarray:
    if isinstance(sv, list):
        return np.asarray(sv[-1])
    sv = np.asarray(sv)
    return sv[..., -1] if sv.ndim == 3 else sv


def compute_shap(pipe, df: pd.DataFrame):
    """Returns (shap_df, X_transformed, base_value)."""
    Xt = pipe.named_steps["pre"].transform(model_frame(df))
    explainer = shap.TreeExplainer(pipe.named_steps["clf"])
    sv = _positive_class(explainer.shap_values(Xt))
    ev = np.atleast_1d(explainer.expected_value)
    base = float(ev[-1])
    return pd.DataFrame(sv, columns=Xt.columns, index=df["customer_id"].values), Xt.set_index(df["customer_id"].values), base


def global_importance(shap_df: pd.DataFrame) -> pd.DataFrame:
    imp = shap_df.abs().mean().sort_values(ascending=False)
    return imp.rename("mean_abs_shap").reset_index().rename(columns={"index": "feature"})


def top_factors(shap_row: pd.Series, feature_row: pd.Series, k: int = 8) -> pd.DataFrame:
    s = shap_row.copy()
    order = s.abs().sort_values(ascending=False).index[:k]
    out = pd.DataFrame({"feature": order, "value": feature_row[order].values, "shap_log_odds": s[order].values})
    out["direction"] = np.where(out["shap_log_odds"] > 0, "raises churn risk", "lowers churn risk")
    return out


def make_plots(shap_df, Xt, out_dir=config.FIGURES_DIR, n_dependence: int = 4):
    out = []
    imp = global_importance(shap_df)
    plt.figure(figsize=(8, 6)); top = imp.head(15).iloc[::-1]
    plt.barh(top["feature"], top["mean_abs_shap"], color="#3b82f6"); plt.xlabel("mean |SHAP| (log-odds)")
    plt.title("Global feature importance (LightGBM churn model)"); plt.tight_layout()
    p = out_dir / "shap_global_importance.png"; plt.savefig(p, dpi=140); plt.close(); out.append(p)

    shap.summary_plot(shap_df.values, Xt, max_display=15, show=False)
    plt.title("SHAP summary (beeswarm)"); plt.tight_layout()
    p = out_dir / "shap_summary.png"; plt.savefig(p, dpi=140, bbox_inches="tight"); plt.close(); out.append(p)

    for f in imp["feature"].head(n_dependence):
        shap.dependence_plot(f, shap_df.values, Xt, show=False, interaction_index="auto")
        plt.tight_layout(); p = out_dir / f"shap_dependence_{f}.png"
        plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close(); out.append(p)
    return out


def explanation_records(shap_df: pd.DataFrame, Xt: pd.DataFrame, base: float) -> pd.DataFrame:
    cols_s, cols_x = list(shap_df.columns), list(Xt.columns)
    return pd.DataFrame({
        "customer_id": shap_df.index,
        "base_value": base,
        "shap_json": [json.dumps({c: round(float(v), 5) for c, v in zip(cols_s, row)}) for row in shap_df.values],
        "feature_json": [json.dumps({c: round(float(v), 5) for c, v in zip(cols_x, row)}) for row in Xt.values],
    })

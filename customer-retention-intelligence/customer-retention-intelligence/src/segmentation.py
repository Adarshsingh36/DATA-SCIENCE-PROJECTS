"""K-Means customer segmentation with data-driven cluster count and rule-based, profile-derived labels."""
from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src import config

SEG_FEATURES = ["tenure_months", "monthly_charges", "clv", "service_count", "churn_probability", "surv_prob_12m"]


def evaluate_k(Z: np.ndarray, k_range=range(2, 9), seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
        rows.append({"k": k, "inertia": float(km.inertia_), "silhouette": float(silhouette_score(Z, km.labels_))})
    return pd.DataFrame(rows)


def choose_k(scores: pd.DataFrame, min_k: int = 3) -> int:
    """Highest silhouette among k >= min_k (k=2 is excluded: too coarse to yield the requested segment types)."""
    s = scores[scores["k"] >= min_k]
    return int(s.loc[s["silhouette"].idxmax(), "k"])


def profile_clusters(frame: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    p = frame.assign(cluster_id=labels).groupby("cluster_id").agg(
        n_customers=("tenure_months", "size"), tenure_months=("tenure_months", "mean"),
        monthly_charges=("monthly_charges", "mean"), clv=("clv", "mean"), service_count=("service_count", "mean"),
        churn_probability=("churn_probability", "mean"), surv_prob_12m=("surv_prob_12m", "mean"))
    if "churn" in frame:
        p["actual_churn_rate"] = frame.assign(cluster_id=labels).groupby("cluster_id")["churn"].mean()
    return p


def derive_labels(profile: pd.DataFrame, frame: pd.DataFrame) -> dict:
    """Labels come from each cluster's measured profile relative to the whole customer base."""
    g_clv, g_risk = frame["clv"].median(), frame["churn_probability"].mean()
    g_ten, g_mc = frame["tenure_months"].median(), frame["monthly_charges"].median()
    labels, used = {}, {}
    for cid, r in profile.iterrows():
        val_hi, risk_hi = r["clv"] >= g_clv, r["churn_probability"] >= g_risk
        new, mc_hi = r["tenure_months"] < 0.6 * g_ten, r["monthly_charges"] >= g_mc
        if val_hi and risk_hi:
            lab = "High-value at-risk"
        elif val_hi:
            lab = "High-value loyal"
        elif new and risk_hi:
            lab = "New customers (early-tenure, elevated risk)" if not mc_hi else "Price-sensitive new customers"
        elif risk_hi and mc_hi:
            lab = "Price-sensitive customers"
        elif new:
            lab = "New customers"
        elif r["clv"] < 0.6 * g_clv:
            lab = "Low-value customers"
        elif r["surv_prob_12m"] < 0.85 * frame["surv_prob_12m"].mean():
            lab = "Mid-value, low-survival (watch list)"
        else:
            lab = "Mid-value stable customers"
        used[lab] = used.get(lab, 0) + 1
        labels[cid] = lab if used[lab] == 1 else f"{lab} ({chr(64 + used[lab])})"
    # make duplicated base labels distinguishable: suffix the first occurrence too
    dup = {l.split(" (")[0] for l in labels.values() if l.endswith(")") and " (" in l and len(l.split(" (")[-1]) == 2}
    for cid, lab in list(labels.items()):
        if lab in dup:
            labels[cid] = f"{lab} (A)"
    return labels


def run_segmentation(frame: pd.DataFrame, seed: int = config.RANDOM_STATE, k: int | None = None, save: bool = True) -> dict:
    """frame must contain customer_id + SEG_FEATURES (+ optional churn)."""
    scaler = StandardScaler().fit(frame[SEG_FEATURES])
    Z = scaler.transform(frame[SEG_FEATURES])
    scores = evaluate_k(Z, seed=seed)
    k = k or choose_k(scores)
    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(Z)
    prof = profile_clusters(frame, km.labels_)
    labels = derive_labels(prof, frame)
    pca = PCA(n_components=2, random_state=seed).fit(Z)
    P = pca.transform(Z)
    seg = pd.DataFrame({"customer_id": frame["customer_id"].values, "cluster_id": km.labels_,
                        "segment_label": [labels[c] for c in km.labels_], "pc1": P[:, 0], "pc2": P[:, 1]})
    prof = prof.assign(segment_label=pd.Series(labels))
    if save:
        joblib.dump({"scaler": scaler, "kmeans": km, "pca": pca, "labels": labels, "features": SEG_FEATURES},
                    config.MODELS_DIR / "segmentation.joblib")
    return {"segments": seg, "profile": prof, "k_scores": scores, "k": k, "silhouette": float(silhouette_score(Z, km.labels_)),
            "pca_explained_variance": pca.explained_variance_ratio_.tolist()}

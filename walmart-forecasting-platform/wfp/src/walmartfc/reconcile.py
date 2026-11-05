"""Forecast reconciliation for the grouped hierarchy (bottom-up, OLS, WLS-structural, MinT-shrink)."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.linalg as sl

from .panel import Hierarchy


def shrunk_covariance(E: np.ndarray):
    """Schafer-Strimmer shrinkage of the residual covariance towards its diagonal.
    E: (T_res, n) residuals; NaN treated as 0 (unobserved)."""
    E = np.nan_to_num(E, nan=0.0)
    T, n = E.shape
    Ec = E - E.mean(axis=0, keepdims=True)
    var = (Ec ** 2).sum(axis=0) / T
    sd = np.sqrt(np.maximum(var, 1e-12))
    X = Ec / sd
    R = X.T @ X / T
    # shrinkage intensity on off-diagonal correlations
    X2 = X ** 2
    v = (X2.T @ X2 / T - R ** 2) * T / (T - 1) ** 2 * T if T > 1 else np.zeros_like(R)
    off = ~np.eye(n, dtype=bool)
    lam = float(np.clip(v[off].sum() / max((R[off] ** 2).sum(), 1e-12) / T, 0.0, 1.0))
    Rs = (1 - lam) * R
    np.fill_diagonal(Rs, 1.0)
    return Rs * np.outer(sd, sd), lam


def projection_matrix(hier: Hierarchy, method: str, W: np.ndarray | None = None) -> np.ndarray | None:
    """Return G (m x n_all) such that reconciled bottom = G @ base_all. None for bottom-up handled separately."""
    S = hier.S.astype(float)
    n_all, m = S.shape
    if method == "bottom_up":
        G = np.zeros((m, n_all)); G[:, n_all - m:] = np.eye(m); return G
    if method == "ols":
        A = (S.T @ S).toarray()
        return np.linalg.solve(A, S.T.toarray())
    if method == "wls_struct":
        w = np.asarray(S.sum(axis=1)).ravel()
        Wi = 1.0 / w
        A = (S.T @ sp.diags(Wi) @ S).toarray()
        return np.linalg.solve(A, (S.T @ sp.diags(Wi)).toarray())
    if method == "mint_shrink":
        if W is None:
            raise ValueError("mint_shrink needs a covariance matrix W")
        c = sl.cho_factor(W + 1e-8 * np.trace(W) / len(W) * np.eye(len(W)), lower=True)
        Wi_S = sl.cho_solve(c, S.toarray())          # W^-1 S   (n_all x m)
        A = S.T @ Wi_S                                # S' W^-1 S
        return np.linalg.solve(A, Wi_S.T)
    raise ValueError(method)


def reconcile(hier: Hierarchy, base_all: np.ndarray, G: np.ndarray) -> np.ndarray:
    """base_all: (n_all, K) -> coherent (n_all, K)."""
    bottom = G @ np.nan_to_num(base_all, nan=0.0)
    return np.asarray(hier.S @ bottom)


def coherence_error(hier: Hierarchy, fc_all: np.ndarray) -> float:
    n_all, m = hier.S.shape
    agg = np.asarray(hier.S @ fc_all[n_all - m:])
    return float(np.max(np.abs(agg - fc_all)))

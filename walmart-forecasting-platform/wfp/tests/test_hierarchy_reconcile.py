import numpy as np
import pytest

from walmartfc.panel import build_full_panel, build_hierarchy
from walmartfc.reconcile import coherence_error, projection_matrix, reconcile, shrunk_covariance


@pytest.fixture(scope="module")
def hier(panel):
    return build_hierarchy(panel.series)


def test_summing_matrix_shape_and_levels(hier, panel):
    n = len(panel.series)
    stores, depts, groups = panel.series.Store.nunique(), panel.series.Dept.nunique(), panel.series.Type.nunique()
    assert hier.S.shape == (1 + groups + stores + depts + n, n)
    assert (hier.S[0].toarray() == 1).all()                      # total sums everything
    assert "proxy" in hier.group_label                           # no state invented


def test_aggregation_equals_sum_of_children(hier, panel):
    full = build_full_panel(panel, hier)
    tot = np.nansum(panel.Y, axis=0)
    assert np.allclose(full.Y[0], tot)
    s2 = hier.level_slices["store"]
    assert np.allclose(full.Y[s2].sum(axis=0), tot)


@pytest.mark.parametrize("method", ["bottom_up", "ols", "wls_struct", "mint_shrink"])
def test_reconciled_forecasts_are_coherent(hier, method):
    rng = np.random.default_rng(0)
    n_all, m = hier.S.shape
    base = rng.normal(100, 10, (n_all, 6))
    W = None
    if method == "mint_shrink":
        W, lam = shrunk_covariance(rng.normal(size=(40, n_all)))
        assert 0 <= lam <= 1
    G = projection_matrix(hier, method, W)
    F = reconcile(hier, base, G)
    assert coherence_error(hier, F) < 1e-8
    assert np.allclose(hier.S.toarray() @ F[n_all - m:], F)       # aggregate rows equal sums of bottom rows


@pytest.mark.parametrize("method", ["ols", "wls_struct"])
def test_projection_leaves_coherent_forecasts_unchanged(hier, method):
    rng = np.random.default_rng(1)
    n_all, m = hier.S.shape
    bottom = rng.uniform(10, 100, (m, 3))
    coherent = hier.S.toarray() @ bottom
    G = projection_matrix(hier, method)
    assert np.allclose(reconcile(hier, coherent, G), coherent, atol=1e-8)


def test_state_mapping_used_when_supplied(panel):
    import pandas as pd
    mp = pd.DataFrame(dict(Store=[1, 2, 3], State=["TX", "TX", "CA"]))
    h = build_hierarchy(panel.series, mp)
    assert h.group_label == "State"
    assert set(h.rows[h.rows.level == "group"].name) == {"TX", "CA"}


def test_state_mapping_must_cover_all_stores(panel):
    import pandas as pd
    with pytest.raises(ValueError):
        build_hierarchy(panel.series, pd.DataFrame(dict(Store=[1, 2], State=["TX", "TX"])))

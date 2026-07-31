"""Tests for divergence-based scoring metrics: Density Power Divergence (DPD).

This module validates the DPD scoring rule implementation and basic consistency
properties: finiteness, ordering (perfect vs imperfect), and the β→0 limit
which recovers the negative log score up to a constant.
"""

import math
import numpy as np
import pytest
import torch

# Force CPU
torch.cuda.is_available = lambda: False

from scoringbench.metrics import (
    compute_scoring_rules,
    DPD_BETAS,
)  # noqa: E402
from scoringbench.wrappers import DistributionPrediction


# ============================================================================
# Numerical Reference Implementations
# ============================================================================

def _reference_unified_density(probas, bin_edges):
    """Independent NumPy re-derivation of the density the metrics module scores.

    The production density is piecewise constant on the bins and is the forward
    difference of the CDF::

        f_k = [F(x_{k+1}) - F(x_k)] / w_k = p_k / w_k

    with ``w_k`` the bin's own width.  ``w_k = 0`` would make that 0/0, but the
    grid never reaches the metrics in that state: ``DistributionPrediction``
    re-expresses the PMF on a regular grid at construction time (see
    ``wrappers.base.regrid_to_uniform``), so every width here is positive and
    the reference needs no special case.  The result is renormalised so that
    ``∑_k f_k w_k = 1``, which for an exact PMF is already true.

    Both terms of a two-term rule must be functionals of this *same* ``f`` for
    the rule to be proper, so the reference builds ``f`` once and returns it
    together with the effective widths that integrate it.

    Returns
    -------
    f : np.ndarray
        Per-bin density values, shape ``(n_bins,)``.
    w_eff : np.ndarray
        Per-bin effective widths, shape ``(n_bins,)``, with ``∑ f w_eff = 1``.
    """
    probas = np.asarray(probas, dtype=float)
    edges = np.asarray(bin_edges, dtype=float)
    widths = np.diff(edges)
    eps = 100 * np.finfo(np.float64).eps

    f = probas / np.maximum(widths, eps)
    w_eff = widths

    return f / max((f * w_eff).sum(), eps), w_eff


def reference_dpd_score(probas, bin_edges, y, beta):
    """Reference DPD score built from the unified bin density.

        S_β = ∫ f(t)^{1+β} dt - (1 + 1/β) f(y)^β    (β > 0)
        S_0 = -log f(y)                              (β = 0 limit)

    Because ``f`` is piecewise constant, ``∫ f^{1+β}`` has the closed form
    ``∑_k f_k^{1+β} w_k^eff`` — no quadrature is needed — and ``f(y)`` is the
    value of that *same* ``f`` on the bin containing ``y``, which is the
    propriety-preserving pairing.
    """
    f, w_eff = _reference_unified_density(probas, bin_edges)
    edges = np.asarray(bin_edges, dtype=float)
    y_bin = int(np.clip(np.searchsorted(edges[1:], y), 0, len(f) - 1))

    eps = 1e-10
    g_y = max(float(f[y_bin]), eps)

    if abs(beta) < 1e-12:
        return -math.log(g_y)

    integral = float((f ** (1.0 + beta) * w_eff).sum())
    point_term = (1.0 + 1.0 / beta) * (g_y ** beta)
    return integral - point_term


# ============================================================================
# Test Fixture: Distribution Builders
# ============================================================================

def get_simple_distribution():
    """Create a simple test distribution with known properties.
    
    Configuration: Bin edges [0, 1, 2, 3], all probability on middle bin [1, 2].
    """
    bin_edges = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    bin_mids = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    # All probability on middle bin
    probas = np.zeros((1, 3), dtype=np.float32)
    probas[0, 1] = 1.0
    mean = np.array([1.5], dtype=np.float64)
    
    return DistributionPrediction(
        probas=probas,
        bin_edges=bin_edges,
        bin_midpoints=bin_mids,
        mean=mean,
    )


def get_perfect_prediction_distribution(bin_idx=0):
    """Create a distribution with all mass on one bin.
    
    Args:
        bin_idx: Which bin (0, 1, or 2) gets all the probability.
    
    Returns:
        Distribution and corresponding target y value for perfect prediction.
    """
    bin_edges = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    bin_mids = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    
    probas = np.zeros((1, 3), dtype=np.float32)
    probas[0, bin_idx] = 1.0
    
    dist = DistributionPrediction(
        probas=probas,
        bin_edges=bin_edges,
        bin_midpoints=bin_mids,
        mean=np.array([bin_mids[bin_idx]], dtype=np.float64),
    )
    
    y_true = np.array([bin_mids[bin_idx]], dtype=np.float32)
    return dist, y_true


def get_imperfect_distribution():
    """Create a distribution with spread across bins.
    
    Configuration: probabilities [0.3, 0.4, 0.3] across bins.
    """
    bin_edges = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    bin_mids = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    
    probas = np.array([[0.3, 0.4, 0.3]], dtype=np.float32)
    
    return DistributionPrediction(
        probas=probas,
        bin_edges=bin_edges,
        bin_midpoints=bin_mids,
        mean=np.array([1.5], dtype=np.float64),
    )


# ============================================================================
# Sanity Check Tests: Finite Values, No NaNs/Infs
# ============================================================================

@pytest.mark.parametrize("beta", DPD_BETAS)
def test_dpd_score_is_finite(beta):
    """Test that DPD scores are finite and well-defined."""
    dist = get_simple_distribution()
    y_true = np.array([1.5], dtype=np.float32)

    metrics = compute_scoring_rules(dist, y_true)
    key = f"dpd_beta_{beta}"

    assert key in metrics, f"Missing key {key}"
    assert isinstance(metrics[key], float)
    assert not np.isnan(metrics[key]), f"{key} is NaN"
    assert not np.isinf(metrics[key]), f"{key} is Inf"


# ============================================================================
# Numerical Validation Tests: Exact Values
# ============================================================================

class TestDPDExactValues:
    """Numerical tests for DPD scores and expected ordering."""

    @pytest.mark.parametrize("beta", DPD_BETAS)
    def test_perfect_vs_imperfect_ordering(self, beta):
        """Perfect prediction should score better (lower) than imperfect prediction.

        ``y`` is placed strictly inside a bin rather than on an edge: bins are
        read as half-open ``(left, right]``, so a target sitting exactly on
        ``edges[15]`` belongs to bin 14, and a spike put in bin 15 would then be
        scored as a *miss* rather than as a perfect prediction.
        """
        edges = np.linspace(0.0, 3.0, 31)
        mids = 0.5 * (edges[:-1] + edges[1:])
        y = np.array([1.55], dtype=np.float32)

        probas_perfect = np.zeros((1, 30))
        probas_perfect[0, 15] = 1.0                      # the bin containing y
        probas_imperfect = np.full((1, 30), 1.0 / 30.0)

        def _dist(probas):
            return DistributionPrediction(
                probas=probas, bin_edges=edges, bin_midpoints=mids,
                mean=(probas * mids).sum(axis=1),
            )

        key = f"dpd_beta_{beta}"
        score_perfect = compute_scoring_rules(_dist(probas_perfect), y)[key]
        score_imperfect = compute_scoring_rules(_dist(probas_imperfect), y)[key]

        assert score_perfect < score_imperfect, (
            f"Perfect prediction ({score_perfect:.6f}) should score lower (better) "
            f"than imperfect ({score_imperfect:.6f}) for {key}"
        )

    def test_various_betas_differ(self):
        """Different β values should produce different numeric results for imperfect predictions."""
        dist = get_imperfect_distribution()
        y_true = np.array([1.5], dtype=np.float32)
        metrics = compute_scoring_rules(dist, y_true)

        scores = [metrics[f"dpd_beta_{b}"] for b in DPD_BETAS]
        assert len(set(scores)) > 1, "Different β values should produce different DPD scores"


class TestComparisonBetweenMetrics:
    """Tests comparing DPD behavior at perfect vs imperfect predictions."""
    
    def test_beta0_matches_log_score(self):
        """DPD at β=0 should match the negative log score computed elsewhere."""
        dist = get_imperfect_distribution()
        y_true = np.array([1.5], dtype=np.float32)

        metrics = compute_scoring_rules(dist, y_true)
        dpd0 = metrics.get("dpd_beta_0.0")
        logscore = metrics.get("log_score")

        assert dpd0 is not None and logscore is not None
        assert math.isclose(dpd0, logscore, rel_tol=1e-6, abs_tol=1e-6), (
            f"dpd_beta_0.0 ({dpd0:.8f}) should equal log_score ({logscore:.8f})"
        )


# ============================================================================
# Integration and Consistency Tests
# ============================================================================

def test_all_new_metrics_present_in_results():
    """Verify all DPD metrics are computed and present in results."""
    dist = get_simple_distribution()
    y_true = np.array([1.5], dtype=np.float32)

    metrics = compute_scoring_rules(dist, y_true)

    for beta in DPD_BETAS:
        key = f"dpd_beta_{beta}"
        assert key in metrics, f"Missing {key}"


def test_different_betas_produce_different_scores():
    """Different β values should produce different DPD results."""
    dist = get_imperfect_distribution()
    y_true = np.array([1.5], dtype=np.float32)

    metrics = compute_scoring_rules(dist, y_true)

    scores = [metrics[f"dpd_beta_{b}"] for b in DPD_BETAS]
    unique_scores = set(scores)

    assert len(unique_scores) > 1, "Different β values should produce different DPD scores"


# ============================================================================
class TestLimitBehavior:
    """Tests verifying limit behavior as parameters approach special values."""
    
    def test_reference_dpd_matches_implementation_for_simple_case(self):
        """Cross-check reference DPD formula against compute_scoring_rules for a simple case."""
        dist = get_simple_distribution()
        y_true = np.array([1.5], dtype=np.float32)

        metrics = compute_scoring_rules(dist, y_true)

        probas = dist.probas[0]
        y = float(y_true[0])

        for beta in DPD_BETAS:
            key = f"dpd_beta_{beta}"
            ref = reference_dpd_score(probas, dist.bin_edges, y, beta)
            # Both sides evaluate the same closed form, so only float summation
            # order separates them; the tolerance is deliberately generous.
            assert math.isclose(metrics[key], ref, rel_tol=1e-4, abs_tol=1e-6), (
                f"DPD implementation {key}={metrics[key]:.8f} differs from reference {ref:.8f}"
            )

    def test_reference_density_integrates_to_one(self):
        """The reference density must be a density (∫ f = 1).

        Guards the reference itself: if the renormalisation or the effective
        widths were wrong, every ∫ f^{1+β} term would be biased and the
        cross-check above would compare two equally wrong numbers.
        """
        dist = get_imperfect_distribution()
        f, w_eff = _reference_unified_density(dist.probas[0], dist.bin_edges)
        mass = float((f * w_eff).sum())
        assert math.isclose(mass, 1.0, rel_tol=1e-12), f"∫ f = {mass:.8f}, expected 1"


class TestEdgeCasesAndRobustness:
    """Tests for edge cases and numerical robustness."""
    
    def test_dpd_very_small_density(self):
        """Test DPD with very small but nonzero density — should be finite and large."""
        bin_edges = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        bin_mids = np.array([0.5, 1.5], dtype=np.float32)
        # Very small probability on first bin
        probas = np.array([[0.001, 0.999]], dtype=np.float32)

        dist = DistributionPrediction(
            probas=probas,
            bin_edges=bin_edges,
            bin_midpoints=bin_mids,
            mean=np.array([1.5], dtype=np.float64),
        )
        y_true = np.array([0.5], dtype=np.float32)
        metrics = compute_scoring_rules(dist, y_true)

        for beta in DPD_BETAS:
            score = metrics[f"dpd_beta_{beta}"]
            assert not np.isnan(score), f"Score is NaN for beta={beta}"
            assert not np.isinf(score), f"Score is Inf for beta={beta}"
            # For β=0 (log-score) very low density should yield a large positive value.
            if abs(beta) < 1e-12:
                assert score > 0.1, f"Very low confidence should give large log-score, got {score}"
            # For other β values DPD can be negative depending on the integral term,
            # so we only require finiteness (already checked above).

    def test_cde_loss_equals_dpd_beta1(self):
        """CDE loss should match DPD with β=1 (∫ f^2 dt - 2 f(y))."""
        dist = get_imperfect_distribution()
        y_true = np.array([1.5], dtype=np.float32)

        metrics = compute_scoring_rules(dist, y_true)

        cde = metrics.get("cde_loss")
        dpd1 = metrics.get("dpd_beta_1.0")

        assert cde is not None and dpd1 is not None
        assert math.isclose(cde, dpd1, rel_tol=1e-6, abs_tol=1e-8), (
            f"cde_loss ({cde}) should equal dpd_beta_1.0 ({dpd1})"
        )

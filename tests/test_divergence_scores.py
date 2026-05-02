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
)
from scoringbench.wrappers import DistributionPrediction


# ============================================================================
# Numerical Reference Implementations
# ============================================================================

def reference_dpd_score(probas, bin_widths, p_at_y, dz_at_y, beta):
    """Reference DPD computation on a histogram grid.

    Uses the discretization:
        integral ≈ ∑_k (p_k^{1+β} / w_k^{β})
        f(y) ≈ p_at_y / dz_at_y
        S_β = integral - (1 + 1/β) f(y)^β   (for β>0)
    For β=0 the limit is -log f(y).
    """
    eps = 1e-10
    bw = np.asarray(bin_widths, dtype=float)
    bw = np.maximum(bw, eps)
    probas = np.asarray(probas, dtype=float)
    p_at_y = float(p_at_y)
    dz_at_y = float(max(dz_at_y, eps))

    if abs(beta) < 1e-12:
        g_y = p_at_y / dz_at_y
        g_y = max(g_y, eps)
        return -math.log(g_y)

    integral = np.sum(probas ** (1.0 + beta) / (bw ** beta))
    g_y = p_at_y / dz_at_y
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
        """Perfect prediction should score better (lower) than imperfect prediction."""
        dist_perfect, y_perfect = get_perfect_prediction_distribution(bin_idx=1)
        dist_imperfect = get_imperfect_distribution()
        y_imperfect = np.array([1.5], dtype=np.float32)

        metrics_perfect = compute_scoring_rules(dist_perfect, y_perfect)
        metrics_imperfect = compute_scoring_rules(dist_imperfect, y_imperfect)

        key = f"dpd_beta_{beta}"
        score_perfect = metrics_perfect[key]
        score_imperfect = metrics_imperfect[key]

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

        # Extract low-level values for the single sample
        probas = dist.probas[0]
        bin_edges = dist.bin_edges
        bin_widths = np.diff(bin_edges)
        # index of y=1.5 is middle bin
        p_at_y = float(probas[1])
        dz_at_y = float(bin_widths[1])

        for beta in DPD_BETAS:
            key = f"dpd_beta_{beta}"
            ref = reference_dpd_score(probas, bin_widths, p_at_y, dz_at_y, beta)
            assert math.isclose(metrics[key], ref, rel_tol=1e-6, abs_tol=1e-6), (
                f"DPD implementation {key}={metrics[key]:.8f} differs from reference {ref:.8f}"
            )


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

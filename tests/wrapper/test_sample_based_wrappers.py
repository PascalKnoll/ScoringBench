"""Integration tests for the sample-/parametric-distribution wrappers.

Each model is exercised on a small but *learnable* synthetic regression task.
A model only "passes" if it can draw conditional predictions well enough that
its predicted mean tracks the true signal and its CRPS is finite and beats a
trivial constant-variance baseline. Tests skip automatically when the backing
library is absent.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from scoringbench.metrics import compute_metrics
from scoringbench.wrappers.base import DistributionPrediction
from scoringbench.wrappers.sample_based import (
    SampleBasedWrapper,
    quantiles_to_distribution,
    samples_to_distribution,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(n_train=400, n_test=120, n_features=5, seed=0):
    rng = np.random.default_rng(seed)
    Xtr = rng.normal(size=(n_train, n_features))
    Xte = rng.normal(size=(n_test, n_features))

    def signal(X):
        return 2.0 * X[:, 0] + np.sin(2.0 * X[:, 1]) - X[:, 2]

    ytr = signal(Xtr) + rng.normal(scale=0.3, size=n_train)
    yte = signal(Xte) + rng.normal(scale=0.3, size=n_test)
    return Xtr.astype(np.float64), ytr, Xte.astype(np.float64), yte


def _validate_distribution(dist: DistributionPrediction, n_test: int):
    assert isinstance(dist, DistributionPrediction)
    assert dist.probas.shape[0] == n_test
    # PMF rows sum to 1.
    np.testing.assert_allclose(dist.probas.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6)
    assert np.all(dist.probas >= -1e-9)
    assert np.all(np.isfinite(dist.bin_edges))
    assert np.all(np.isfinite(dist.bin_midpoints))
    assert np.all(np.isfinite(dist.mean))
    # Edges monotone non-decreasing per sample (per-sample 2-D grid expected).
    if dist.bin_edges.ndim == 2:
        assert np.all(np.diff(dist.bin_edges, axis=1) >= -1e-9)


def _assert_learns(dist: DistributionPrediction, y_test: np.ndarray):
    """Conditional draws are 'good enough': mean tracks signal, CRPS beats baseline."""
    _validate_distribution(dist, len(y_test))
    metrics = compute_metrics(dist, y_test)

    # Predicted mean correlates with the truth.
    r = np.corrcoef(dist.mean, y_test)[0, 1]
    assert r > 0.6, f"predicted mean barely correlates with target (r={r:.3f})"

    crps = metrics["crps"]
    assert np.isfinite(crps), "CRPS is not finite"

    # Trivial baseline: a single wide Gaussian PMF (mean 0, std = std(y_test))
    # encoded on a shared grid — sample-based model should be sharper/better.
    std = float(np.std(y_test)) + 1e-6
    grid = np.linspace(y_test.mean() - 5 * std, y_test.mean() + 5 * std, 200)
    mids = 0.5 * (grid[:-1] + grid[1:])
    from scipy.stats import norm
    pdf = norm.pdf(mids, loc=float(np.mean(y_test)), scale=std)
    probas = np.tile(pdf / pdf.sum(), (len(y_test), 1))
    baseline = DistributionPrediction(
        probas=probas, bin_edges=grid, bin_midpoints=mids,
        mean=np.full(len(y_test), float(np.mean(y_test))),
    )
    base_crps = compute_metrics(baseline, y_test)["crps"]
    assert crps < base_crps, f"CRPS {crps:.3f} not better than baseline {base_crps:.3f}"


# ---------------------------------------------------------------------------
# Unit tests for the shared machinery
# ---------------------------------------------------------------------------

def test_quantiles_to_distribution_shapes():
    alphas = np.array([0.25, 0.5, 0.75])
    q = np.array([[0.0, 1.0, 2.0], [1.0, 1.5, 4.0]])
    dist = quantiles_to_distribution(q, alphas)
    assert dist.probas.shape == (2, len(alphas) + 1)
    assert dist.bin_edges.shape == (2, len(alphas) + 2)
    np.testing.assert_allclose(dist.probas.sum(axis=1), 1.0)


def test_samples_to_distribution_recovers_mean():
    rng = np.random.default_rng(1)
    samples = rng.normal(loc=3.0, scale=1.0, size=(4, 5000))
    dist = samples_to_distribution(samples, n_quantiles=99)
    _validate_distribution(dist, 4)
    np.testing.assert_allclose(dist.mean, 3.0, atol=0.1)


def test_sample_based_timeout_is_respected():
    class SlowWrapper(SampleBasedWrapper):
        N_SAMPLES = 1000
        SAMPLE_CHUNK = 50
        MAX_SAMPLE_SECONDS = 0.3

        def _draw_samples(self, X, n_samples):
            time.sleep(0.1)
            return np.zeros((len(X), n_samples))

    w = SlowWrapper()
    X = np.zeros((3, 2))
    t0 = time.monotonic()
    samples = w._collect_samples(X)
    elapsed = time.monotonic() - t0
    # Stopped early: fewer than the full target, and not absurdly over budget.
    assert samples.shape[1] < 1000
    assert elapsed < 2.0


# ---------------------------------------------------------------------------
# Per-model integration tests (skip if library missing)
# ---------------------------------------------------------------------------

def test_ngboost_integration():
    pytest.importorskip("ngboost")
    from scoringbench.wrappers.ngboost_wrapper import NGBoostWrapper

    Xtr, ytr, Xte, yte = _make_data()
    model = NGBoostWrapper(n_estimators=300, learning_rate=0.03, n_quantiles=99)
    model.fit(Xtr, ytr)
    dist = model.predict_distribution(Xte)
    _assert_learns(dist, yte)


def test_nflows_integration():
    pytest.importorskip("nflows")
    pytest.importorskip("torch")
    from scoringbench.wrappers.nflows_wrapper import NFlowsWrapper

    Xtr, ytr, Xte, yte = _make_data()
    model = NFlowsWrapper(
        n_layers=4, hidden_features=64, num_bins=8,
        n_epochs=300, batch_size=256, lr=1e-3, device="cpu", n_samples=300,
    )
    model.fit(Xtr, ytr)
    dist = model.predict_distribution(Xte)
    _assert_learns(dist, yte)


@pytest.mark.slow
def test_bart_integration():
    pytest.importorskip("pymc")
    pytest.importorskip("pymc_bart")
    from scoringbench.wrappers.bart_wrapper import BARTWrapper

    Xtr, ytr, Xte, yte = _make_data(n_train=200, n_test=80)
    model = BARTWrapper(num_trees=30, draws=150, tune=150, chains=2, cores=1)
    model.fit(Xtr, ytr)
    dist = model.predict_distribution(Xte)
    _assert_learns(dist, yte)

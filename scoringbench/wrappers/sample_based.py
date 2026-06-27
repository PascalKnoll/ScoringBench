"""Shared utilities for sample-/quantile-based ScoringBench wrappers.

Two ingredients live here:

``quantiles_to_distribution``
    Convert a per-sample quantile matrix ``q`` (n_samples, K) evaluated at
    probability levels ``alphas`` into a :class:`DistributionPrediction` with
    per-sample bin edges and a piecewise-uniform PMF — exactly the
    representation the rest of ScoringBench (CatBoost / XGB-quantile / Synthefy
    wrappers) already produces.  Parametric models (e.g. NGBoost) feed it their
    analytic ``ppf`` grid; sample-based models feed it empirical ``np.quantile``
    estimates.

``SampleBasedWrapper``
    Base class for genuinely sample-based models (normalizing flows, BART, …).
    Subclasses only implement ``_draw_samples(X, n)`` returning an
    ``(n_test, n)`` array of conditional draws of the target.  The base class
    accumulates draws in chunks under a wall-clock budget
    (``MAX_SAMPLE_SECONDS``, default 120 s — it may sample for at most two
    minutes, then derives the PMF from whatever was collected) and converts the
    pooled draws into a ``DistributionPrediction``.
"""

from __future__ import annotations

import time

import numpy as np

from .base import DistributionPrediction, ProbabilisticWrapper


# ---------------------------------------------------------------------------
# Quantiles / samples -> DistributionPrediction
# ---------------------------------------------------------------------------

def quantiles_to_distribution(
    q: np.ndarray,
    alphas: np.ndarray,
    mean: np.ndarray | None = None,
    y_range: tuple[float, float] | None = None,
) -> DistributionPrediction:
    """Build a piecewise-uniform ``DistributionPrediction`` from quantiles.

    Parameters
    ----------
    q : (n_samples, K) array
        Per-sample quantile values at the probability levels ``alphas``.
    alphas : (K,) array
        Strictly increasing probability levels in (0, 1).
    mean : (n_samples,) array, optional
        Point prediction to report. If ``None`` the PMF mean is used.
    y_range : (lo, hi), optional
        Fallback finite range used to sanitize non-finite quantiles.
    """
    q = np.asarray(q, dtype=np.float64)
    if q.ndim == 1:
        q = q[np.newaxis, :]
    alphas = np.asarray(alphas, dtype=np.float64).reshape(-1)

    lo, hi = (float(y_range[0]), float(y_range[1])) if y_range is not None else (0.0, 1.0)
    if not np.all(np.isfinite(q)):
        q = np.nan_to_num(q, nan=lo, posinf=hi, neginf=lo)

    # Enforce monotonic quantiles per sample.
    q = np.sort(q, axis=1)
    n_samples = q.shape[0]

    # Per-sample bin edges: extend slightly beyond the outermost quantiles so
    # the full probability mass [0, 1] is covered.
    left_w = np.maximum(q[:, 1] - q[:, 0], 1e-7)
    right_w = np.maximum(q[:, -1] - q[:, -2], 1e-7)
    bin_edges = np.concatenate(
        [(q[:, 0] - left_w)[:, None], q, (q[:, -1] + right_w)[:, None]],
        axis=1,
    )

    # Mass per bin: alpha_0, diff(alphas), 1 - alpha_last.
    masses = np.concatenate([[alphas[0]], np.diff(alphas), [1.0 - alphas[-1]]])
    probas = np.broadcast_to(masses[None, :], (n_samples, len(masses))).copy()

    bin_midpoints = (bin_edges[:, :-1] + bin_edges[:, 1:]) / 2
    pmf_mean = np.sum(probas * bin_midpoints, axis=-1)
    out_mean = pmf_mean if mean is None else np.asarray(mean, dtype=np.float64).reshape(-1)

    return DistributionPrediction(
        probas=probas,
        bin_edges=bin_edges,
        bin_midpoints=bin_midpoints,
        mean=out_mean,
    )


def samples_to_distribution(
    samples: np.ndarray,
    n_quantiles: int = 99,
    y_range: tuple[float, float] | None = None,
) -> DistributionPrediction:
    """Derive a ``DistributionPrediction`` from conditional draws.

    Parameters
    ----------
    samples : (n_test, n_draws) array
        Conditional samples of the target, one row per test instance.
    n_quantiles : int
        Number of equally-spaced probability levels used to summarize the
        empirical distribution of each row.
    """
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim == 1:
        samples = samples[:, None]

    # Replace any non-finite draws with the per-row median (or 0 if a whole row
    # is non-finite) so quantile estimation stays well defined.
    if not np.all(np.isfinite(samples)):
        finite = np.isfinite(samples)
        row_median = np.array([
            np.median(samples[i, finite[i]]) if finite[i].any() else 0.0
            for i in range(samples.shape[0])
        ])
        samples = np.where(finite, samples, row_median[:, None])

    alphas = np.array([k / (n_quantiles + 1) for k in range(1, n_quantiles + 1)])
    # np.quantile over the draw axis -> (K, n_test) -> transpose to (n_test, K).
    q = np.quantile(samples, alphas, axis=1).T
    mean = samples.mean(axis=1)
    return quantiles_to_distribution(q, alphas, mean=mean, y_range=y_range)


def grid_density_to_distribution(
    grid: np.ndarray,
    density: np.ndarray,
    mean: np.ndarray | None = None,
) -> DistributionPrediction:
    """Build a ``DistributionPrediction`` from a conditional density on a shared grid.

    Used by analytic CDE wrappers that can evaluate ``p(y | x)`` on a fixed
    ``y``-grid (the same grid for every test instance). Each grid point owns one
    bin whose width is the gap to its neighbours (outer half-cells mirrored); the
    per-bin mass is ``density * width``, normalized to sum to 1 per row.

    Parameters
    ----------
    grid : (G,) array
        Shared, increasing grid of ``y`` values (the bin midpoints).
    density : (n_samples, G) or (G,) array
        Conditional density evaluated at ``grid`` for each test instance.
    mean : (n_samples,) array, optional
        Point prediction to report. If ``None`` the PMF mean is used.
    """
    grid = np.asarray(grid, dtype=np.float64).reshape(-1)
    density = np.asarray(density, dtype=np.float64)
    if density.ndim == 1:
        density = density[np.newaxis, :]
    density = np.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)
    density = np.clip(density, 0.0, None)

    G = grid.shape[0]
    edges = np.empty(G + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (grid[:-1] + grid[1:])
    edges[0] = grid[0] - 0.5 * (grid[1] - grid[0])
    edges[-1] = grid[-1] + 0.5 * (grid[-1] - grid[-2])
    widths = np.diff(edges)

    mass = density * widths[None, :]
    totals = mass.sum(axis=1, keepdims=True)
    totals = np.where(totals > 0, totals, 1.0)
    probas = mass / totals

    out_mean = probas @ grid if mean is None else np.asarray(mean, dtype=np.float64).reshape(-1)
    return DistributionPrediction(
        probas=probas,
        bin_edges=edges,
        bin_midpoints=grid,
        mean=out_mean,
    )


# ---------------------------------------------------------------------------
# Sample-based wrapper base
# ---------------------------------------------------------------------------

class SampleBasedWrapper(ProbabilisticWrapper):
    """Base class for models whose predictive density is accessed via sampling.

    Subclasses implement :meth:`_draw_samples`. The PMF is derived from the
    pooled draws; sampling is capped at ``MAX_SAMPLE_SECONDS`` wall-clock
    seconds per ``predict_distribution`` call.

    Class attributes
    ----------------
    N_SAMPLES : int
        Target number of conditional draws per test instance (default 300).
    SAMPLE_CHUNK : int
        Draws requested per call to ``_draw_samples``; the budget is checked
        between chunks. Set equal to ``N_SAMPLES`` for one-shot samplers.
    MAX_SAMPLE_SECONDS : float
        Hard wall-clock cap on sampling (default 120 s). Once exceeded, the PMF
        is built from whatever draws were collected so far.
    N_QUANTILES : int
        Resolution of the PMF derived from the draws.
    """

    N_SAMPLES: int = 300
    SAMPLE_CHUNK: int = 100
    MAX_SAMPLE_SECONDS: float = 120.0
    N_QUANTILES: int = 99

    def _draw_samples(self, X, n_samples: int) -> np.ndarray:
        """Return an ``(n_test, n_samples)`` array of conditional target draws."""
        raise NotImplementedError

    def _collect_samples(self, X) -> np.ndarray:
        target = int(self.N_SAMPLES)
        chunk = int(self.SAMPLE_CHUNK) or target
        collected: list[np.ndarray] = []
        n_have = 0
        start = time.monotonic()
        while n_have < target:
            take = min(chunk, target - n_have)
            s = np.asarray(self._draw_samples(X, take), dtype=np.float64)
            if s.ndim == 1:
                s = s[:, None]
            collected.append(s)
            n_have += s.shape[1]
            if time.monotonic() - start >= self.MAX_SAMPLE_SECONDS:
                break
        if not collected:
            raise RuntimeError("No samples were drawn.")
        return np.concatenate(collected, axis=1)

    def predict_distribution(self, X) -> DistributionPrediction:
        samples = self._collect_samples(X)
        return samples_to_distribution(samples, n_quantiles=self.N_QUANTILES)

    def predict(self, X) -> np.ndarray:
        return self._collect_samples(X).mean(axis=1)

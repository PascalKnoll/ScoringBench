"""Floating-point datatype regression tests for the histogram scoring rules.

Several scoring rules form differences of large, nearly-equal quantities:

    CRPS / energy score   term1 - term2      (E|X-y| minus half self-distance)
    predictive variance   E[X²] - E[X]²      (sharpness / dispersion)

Evaluated in float32 these suffer *catastrophic cancellation*: for sharp,
large-magnitude histograms (e.g. many-quantile heads predicting targets in the
1e5–1e7 range) the float32 result drifts far from the true value — CRPS gets
corrupted (driven below zero, then clamped to 0, losing ~17 nats) and the
predictive variance is off by an order of magnitude.

These tests construct such histograms, run each metric explicitly in float32 vs
float64 *on identical inputs* and:

  * showcase the float32 problem (large divergence / corrupted scores), and
  * assert the production (decorated, float64) path is correct.

The decorated metric functions expose the raw, undecorated implementation via
``__wrapped__`` (set by ``functools.wraps``); wrapping it with
``force_precision`` lets us run the very same kernel in either dtype.
"""
import numpy as np
import pytest
import torch

from scoringbench.metrics import (
    _interval,
    compute_cde_loss,
    compute_crls,
    compute_dpd_scores,
    compute_energy_score_histogram_corrected,
    compute_pit_ks,
    compute_quantile_wcrps,
    compute_scoring_rules,
    force_precision,
)
from scoringbench.wrappers import DistributionPrediction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_histogram(scale: float, span: float, K: int):
    """A single sharp, equal-width histogram: ``K`` bins of uniform mass over
    ``[scale - span/2, scale + span/2]``.  Returns (edges, mids, widths, probas)."""
    mids = np.linspace(scale - span / 2.0, scale + span / 2.0, K)
    widths = np.full(K, span / (K - 1))
    edges = np.concatenate([[mids[0] - widths[0] / 2.0], mids + widths / 2.0])
    probas = np.full(K, 1.0 / K)
    return edges, mids, widths, probas


def _energy_kernel(dtype):
    """The raw (undecorated) energy-score kernel forced to a given dtype."""
    return force_precision(dtype)(compute_energy_score_histogram_corrected.__wrapped__)


def _crps(kernel, mids, widths, probas, y):
    out = kernel(
        torch.as_tensor(probas[None, :], dtype=torch.float64),
        torch.as_tensor(mids, dtype=torch.float64),
        torch.as_tensor(widths, dtype=torch.float64),
        torch.as_tensor([y], dtype=torch.float64),
        betas=[1.0],
    )
    return out["energy_score_beta_1.0"]


# ---------------------------------------------------------------------------
# CRPS / energy score
# ---------------------------------------------------------------------------

def test_crps_float32_corrupted_float64_correct():
    """Deterministic sharp, large-scale histogram: float32 cancellation corrupts
    CRPS (clamped to ~0) while float64 yields the correct large value; the
    shipped (decorated) function matches float64."""
    _, mids, widths, probas = _uniform_histogram(scale=1.0e7, span=200.0, K=256)
    y = 1.0e7  # target at the centre

    crps_f32 = _crps(_energy_kernel(torch.float32), mids, widths, probas, y)
    crps_f64 = _crps(_energy_kernel(torch.float64), mids, widths, probas, y)

    # float64 is correct and well away from zero (~16.7); float32 is badly wrong.
    assert crps_f64 > 1.0
    assert crps_f32 < 0.5 * crps_f64
    assert (crps_f64 - crps_f32) > 1.0

    # Production path takes float32 inputs but must compute in float64.
    crps_prod = compute_energy_score_histogram_corrected(
        torch.as_tensor(probas[None, :], dtype=torch.float32),
        torch.as_tensor(mids, dtype=torch.float32),
        torch.as_tensor(widths, dtype=torch.float32),
        torch.as_tensor([y], dtype=torch.float32),
        betas=[1.0],
    )["energy_score_beta_1.0"]
    assert crps_prod >= 0.0
    # Matches the float64 value (small residual is float32 *input* quantization,
    # not the catastrophic arithmetic cancellation the float32 kernel suffers).
    assert crps_prod == pytest.approx(crps_f64, rel=1e-3)


def test_crps_random_sharp_histograms_float32_vs_float64():
    """Over many random sharp, large-scale histograms: float64 always honours
    CRPS >= 0, and float32 diverges from it in a large fraction of cases — the
    datatype demonstrably changes the output."""
    rng = np.random.default_rng(12345)
    k32, k64 = _energy_kernel(torch.float32), _energy_kernel(torch.float64)
    n_total = 200
    n_diverge = 0
    max_diff = 0.0
    for _ in range(n_total):
        K = int(rng.integers(600, 1001))
        scale = float(10.0 ** rng.uniform(4.0, 7.0))
        span = float(rng.uniform(1.0, 100.0))
        _, mids, widths, probas = _uniform_histogram(scale, span, K)
        y = scale + float(rng.uniform(-span / 2.0, span / 2.0))

        c32 = _crps(k32, mids, widths, probas, y)
        c64 = _crps(k64, mids, widths, probas, y)

        assert c64 >= 0.0, f"float64 CRPS went negative: {c64}"
        diff = abs(c32 - c64)
        max_diff = max(max_diff, diff)
        if diff > 0.5:
            n_diverge += 1

    # The float32 datatype problem is common and large, not a one-off.
    assert n_diverge > n_total // 4
    assert max_diff > 1.0


# ---------------------------------------------------------------------------
# Independent ground truth (which dtype is actually right?)
# ---------------------------------------------------------------------------

def _crps_by_definition(probas_row, edges, y, n_grid=2_000_001):
    """Ground-truth CRPS from its definition

        CRPS(F, y) = ∫ (F(t) - 1{t >= y})² dt,

    integrated on a fine float64 grid.  Shares **no code** with the metric
    kernels: ``F`` is the piecewise-linear CDF of the histogram (uniform density
    within each bin), so this is a fully independent reference.
    """
    cdf_at_edges = np.concatenate([[0.0], np.cumsum(probas_row)])
    cdf_at_edges[-1] = 1.0
    t = np.linspace(edges[0] - 1.0, edges[-1] + 1.0, n_grid)
    Ft = np.interp(t, edges, cdf_at_edges)            # 0 below support, 1 above
    step = (t >= y).astype(np.float64)
    return float(np.trapz((Ft - step) ** 2, t))


def test_crps_float64_matches_independent_definition():
    """The float64 kernel equals an independent ground-truth CRPS (the ∫
    definition), while the float32 kernel is far from it — proving float64 is the
    correct one, not merely 'different'."""
    rng = np.random.default_rng(0)
    scale, span, K = 1.0e7, 200.0, 400
    edges, mids, widths, _ = _uniform_histogram(scale, span, K)
    k32, k64 = _energy_kernel(torch.float32), _energy_kernel(torch.float64)

    for _ in range(5):
        p = rng.random(K) + 1.0
        p /= p.sum()
        y = scale + float(rng.uniform(-span / 2.0, span / 2.0))

        truth = _crps_by_definition(p, edges, y)
        c64 = _crps(k64, mids, widths, p, y)
        c32 = _crps(k32, mids, widths, p, y)

        # float64 reproduces the ground truth; float32 is corrupted (clamped ~0).
        assert truth > 1.0
        assert c64 == pytest.approx(truth, rel=5e-3)
        assert abs(c32 - truth) > 1.0


# ---------------------------------------------------------------------------
# Full pipeline: variance / sharpness and overall robustness
# ---------------------------------------------------------------------------

def _hist_std(probas, mids, dtype):
    """Predictive std from the same E[X²] - E[X]² formula the pipeline uses."""
    p = probas.astype(dtype)
    m = mids.astype(dtype)
    mean = (p * m).sum(dtype=dtype)
    msq = (p * (m.astype(np.float64) ** 2).astype(dtype)).sum(dtype=dtype)
    var = float(msq) - float(mean) ** 2
    return float(np.sqrt(max(var, 0.0)))


def test_pipeline_sharpness_and_finiteness_large_scale():
    """End-to-end: a narrow distribution at a large location has a small but
    strictly positive spread.  Computing E[X²] - E[X]² in float32 is off by an
    order of magnitude; the float64 pipeline recovers the correct value and all
    reported metrics stay finite."""
    scale, span, K, n = 1.0e7, 200.0, 400, 8
    edges, mids, widths, probas = _uniform_histogram(scale, span, K)

    dist = DistributionPrediction(
        probas=np.tile(probas, (n, 1)),
        bin_edges=edges,
        bin_midpoints=mids,
        mean=np.full(n, scale),
    )
    y_true = np.full(n, scale)

    out = compute_scoring_rules(dist, y_true)

    std64 = _hist_std(probas, mids, np.float64)
    std32 = _hist_std(probas, mids, np.float32)

    # float32 variance is wildly off (collapse or blow-up); float64 is correct.
    assert std64 > 0.0
    assert abs(std32 - std64) > 0.5 * std64

    # Independent analytic ground truth: uniform mass over the span has
    # std ≈ span / sqrt(12); float64 matches it.
    assert std64 == pytest.approx(span / np.sqrt(12.0), rel=0.05)

    # Pipeline (float64) recovers the true spread and every metric is finite.
    assert out["sharpness"] == pytest.approx(std64, rel=1e-3)
    assert out["sharpness"] > 0.0
    assert out["crps"] >= 0.0
    assert all(np.isfinite(v) for v in out.values())


def test_sharpness_and_dispersion_match_analytic_per_sample():
    """Per-sample grids with *different* spans give each sample an exact
    analytic predictive std; check both sharpness (mean of stds) and dispersion
    (population std of stds) against closed forms, and show float32 is wrong.

    For ``K`` equally-weighted, equally-spaced points spanning width ``W`` the
    variance is ``W² (K+1) / (12 (K-1))`` (variance of a discrete uniform), so
    each sample's std is ``W * sqrt((K+1) / (12 (K-1)))``.
    """
    scale, K = 1.0e7, 300
    spans = np.array([50.0, 100.0, 150.0, 200.0, 250.0])
    n = spans.size

    mids = np.stack([np.linspace(scale - s / 2.0, scale + s / 2.0, K) for s in spans])
    widths = np.stack([np.full(K, s / (K - 1)) for s in spans])
    edges = np.concatenate([mids[:, :1] - widths[:, :1] / 2.0, mids + widths / 2.0], axis=1)
    probas = np.full((n, K), 1.0 / K)

    dist = DistributionPrediction(
        probas=probas, bin_edges=edges, bin_midpoints=mids, mean=mids.mean(axis=1),
    )
    out = compute_scoring_rules(dist, np.full(n, scale))

    # Closed-form per-sample std, then mean (sharpness) and population std (dispersion).
    std_i = spans * np.sqrt((K + 1) / (12.0 * (K - 1)))
    sharpness_ref = float(std_i.mean())
    dispersion_ref = float(std_i.std(ddof=0))  # population std == torch std(unbiased=False)
    assert dispersion_ref > 0.0

    # float64 pipeline matches the analytic values.
    assert out["sharpness"] == pytest.approx(sharpness_ref, rel=1e-3)
    assert out["dispersion"] == pytest.approx(dispersion_ref, rel=1e-3)

    # float32 (E[X²]-E[X]² at this scale) is badly wrong for both.
    std_i_32 = np.array([_hist_std(probas[i], mids[i], np.float32) for i in range(n)])
    assert abs(float(std_i_32.mean()) - sharpness_ref) > 0.5 * sharpness_ref
    assert abs(float(std_i_32.std(ddof=0)) - dispersion_ref) > 0.5 * dispersion_ref


# ---------------------------------------------------------------------------
# Sweep every metric for float32-vs-float64 sensitivity
# ---------------------------------------------------------------------------
#
# Each metric in metrics.py is registered below as an adapter ``fn(inputs) ->
# {label: value}``.  The adapters call the *raw* (undecorated) kernels via
# ``__wrapped__`` so we can run the identical maths in float32 and float64 and
# report where the datatype actually changes the result.  Inline rules that are
# not standalone functions (sharpness/variance, log-score) are reproduced here
# exactly as ``_compute_scoring_rules_torch`` computes them.

def _make_inputs(dtype, scale=1.0e7, span=200.0, K=400, n=8, seed=0):
    """A batch of ``n`` sharp, large-scale histograms (shared grid) plus every
    derived tensor the metric kernels need, all in ``dtype``."""
    rng = np.random.default_rng(seed)
    mids_np = np.linspace(scale - span / 2.0, scale + span / 2.0, K)
    widths_np = np.full(K, span / (K - 1))
    edges_np = np.concatenate([[mids_np[0] - widths_np[0] / 2.0], mids_np + widths_np / 2.0])
    probas_np = rng.random((n, K)) + 1.0
    probas_np /= probas_np.sum(axis=1, keepdims=True)
    y_np = scale + rng.uniform(-span / 2.0, span / 2.0, size=n)

    t = lambda a: torch.as_tensor(a, dtype=dtype)
    device = torch.device("cpu")
    probas = t(probas_np)
    bin_edges = t(edges_np)
    bin_mids = t(mids_np)
    bin_widths = t(widths_np)
    y = t(y_np)
    n_samples, n_bins = probas.shape
    ns_idx = torch.arange(n_samples)
    cdf = torch.cumsum(probas, dim=-1)
    bw = bin_widths[None, :]
    y_bin = torch.searchsorted(bin_edges[1:].contiguous(), y).clamp(0, n_bins - 1)
    p_at_y = probas.gather(1, y_bin.unsqueeze(1)).squeeze(1)
    dz_at_y = bin_widths[y_bin]
    eps = 100 * torch.finfo(dtype).eps
    return dict(
        probas=probas, bin_edges=bin_edges, bin_mids=bin_mids, bin_widths=bin_widths,
        y=y, cdf=cdf, bw=bw, y_bin=y_bin, p_at_y=p_at_y, dz_at_y=dz_at_y,
        ns_idx=ns_idx, eps=eps, n_samples=n_samples, n_bins=n_bins, device=device,
    )


def _m_energy_crps(I):
    return compute_energy_score_histogram_corrected.__wrapped__(
        I["probas"], I["bin_mids"], I["bin_widths"], I["y"], betas=[1.0])


def _m_dpd(I):
    return compute_dpd_scores.__wrapped__(
        I["probas"], I["bin_widths"], I["p_at_y"], I["dz_at_y"], betas=[0.5, 1.0], shared=True)


def _m_wcrps(I):
    return compute_quantile_wcrps.__wrapped__(
        I["cdf"], I["bin_mids"], I["y"], I["n_samples"], I["n_bins"], I["device"], True)


def _m_crls(I):
    return {"crls": compute_crls.__wrapped__(
        I["cdf"], I["bin_widths"], I["y_bin"], I["n_bins"], I["device"], I["eps"], I["bw"], True)}


def _m_cde_loss(I):
    return {"cde_loss": compute_cde_loss.__wrapped__(
        I["probas"], I["bin_widths"], I["y_bin"], I["y"], I["bw"], True, I["ns_idx"])}


def _m_pit_ks(I):
    return compute_pit_ks.__wrapped__(
        I["probas"], I["cdf"], I["bin_edges"], I["bin_widths"], I["y_bin"], I["y"], True, I["ns_idx"])


def _m_interval(I):
    score, cov = _interval.__wrapped__(
        0.1, I["cdf"], I["bin_edges"], I["y"], I["n_samples"], I["n_bins"],
        I["device"], True, I["y_bin"], I["ns_idx"])
    return {"interval_score_90": score, "coverage_90": cov}


def _m_sharpness(I):
    """Inline variance E[X²] - E[X]² (sharpness / dispersion)."""
    probas, mids = I["probas"], I["bin_mids"]
    mean_ = (probas * mids).sum(dim=-1)
    var_ = ((probas * mids.pow(2)).sum(dim=-1) - mean_.pow(2)).clamp(min=0)
    std = var_.sqrt()
    return {"sharpness": std.mean().item(), "dispersion": std.std(unbiased=False).item()}


def _m_log_score(I):
    """Inline log-score -log(density at y)."""
    sel_p = I["probas"][I["ns_idx"], I["y_bin"]]
    sel_w = I["bin_widths"][I["y_bin"]]
    density = sel_p / sel_w.clamp(min=I["eps"])
    return {"log_score": -torch.log(density.clamp(min=I["eps"])).mean().item()}


# Registry of every metric in metrics.py (standalone kernels + inline rules).
METRICS_LIST = [
    ("energy/crps", _m_energy_crps),
    ("dpd",         _m_dpd),
    ("wcrps",       _m_wcrps),
    ("crls",        _m_crls),
    ("cde_loss",    _m_cde_loss),
    ("pit_ks",      _m_pit_ks),
    ("interval",    _m_interval),
    ("sharpness",   _m_sharpness),
    ("log_score",   _m_log_score),
]

# "Sensible absolute amount": flag a metric as dtype-sensitive only when the
# float32/float64 outputs differ by more than this in absolute value.
_DTYPE_ABS_TOL = 1e-3


def test_all_metrics_dtype_sensitivity_report(capsys):
    """Loop over every registered metric, run it in float32 and float64 on
    identical sharp, large-scale histograms, and print which ones are
    dtype-sensitive (|f32 - f64| above a sensible absolute tolerance)."""
    I32 = _make_inputs(torch.float32)
    I64 = _make_inputs(torch.float64)

    sensitive = []
    lines = ["", f"{'metric:output':32s} {'float32':>16s} {'float64':>16s} {'|Δ|':>12s}  flag"]
    for name, fn in METRICS_LIST:
        out32, out64 = fn(I32), fn(I64)
        for key in out64:
            v32, v64 = float(out32[key]), float(out64[key])
            assert np.isfinite(v64), f"{name}:{key} float64 result is not finite"
            diff = abs(v32 - v64)
            flagged = diff > _DTYPE_ABS_TOL
            if flagged:
                sensitive.append((f"{name}:{key}", v32, v64, diff))
            lines.append(
                f"{name + ':' + key:32s} {v32:16.6g} {v64:16.6g} {diff:12.4g}  "
                f"{'DTYPE-SENSITIVE' if flagged else 'stable'}"
            )

    lines.append("")
    lines.append(f"{len(sensitive)}/{sum(len(fn(I64)) for _, fn in METRICS_LIST)} outputs are dtype-sensitive "
                 f"(|Δ| > {_DTYPE_ABS_TOL:g})")
    report = "\n".join(lines)
    # capsys.disabled() prints straight to the terminal so the report is visible
    # even without pytest's -s / --capture=no flag.
    with capsys.disabled():
        print(report)

    # Sanity: the known cancellation-prone rules must show up as sensitive.
    flagged_names = {s[0] for s in sensitive}
    assert any(n.startswith("energy/crps") for n in flagged_names)
    assert any(n.startswith("sharpness") for n in flagged_names)


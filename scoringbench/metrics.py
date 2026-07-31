"""Scoring rules and point metrics for tabular regression.

All functions work on numpy arrays. PyTorch is used internally for GPU acceleration
when available; falls back to CPU otherwise.

Public API
----------
compute_metrics(dist, y_true) -> dict
    All metrics: MAE, RMSE, R², CRPS, log-score, sharpness, dispersion,
    90%/95% coverage and interval scores, energy scores β∈{0.5,1,1.5,2},
    CRLS.

compute_point_metrics(y_true, y_pred) -> dict
    MAE, RMSE, R².

compute_scoring_rules(dist, y_true) -> dict
    CRPS, log-score, sharpness, dispersion, coverage and interval scores,
    energy scores, CRLS, wCRPS_left, wCRPS_right, wCRPS_center.
    dist is a DistributionPrediction from scoringbench.wrappers.
    bin_edges / bin_midpoints may be 1-D (shared grid) or 2-D (per-sample).
    Uses PyTorch on GPU when available; falls back to CPU otherwise.
"""

import functools
import logging
import time

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .wrappers import DistributionPrediction

logger = logging.getLogger(__name__)

# Energy score β values reported as additional metrics
ENERGY_BETAS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.7, 1.8, 1.9]
DPD_BETAS = [0.0, 0.2, 0.5, 1.0]  # β values for Density Power Divergence scoring rule; 0.0 -> log-score (limit)
DPD_BASED_KEYS = ("log_score", "cde_loss", *[f"dpd_beta_{b}" for b in DPD_BETAS])
# Central coverage levels (%) reported via coverage_{level} / interval_score_{level};
# the corresponding significance level is alpha = 1 - level/100.
COVERAGE_LEVELS = [20, 40, 60, 80, 90, 95]


# ---------------------------------------------------------------------------
# Numerical precision
# ---------------------------------------------------------------------------

def force_precision(dtype: torch.dtype = torch.float64):
    """Decorator: upcast every floating-point tensor argument to ``dtype``.

    Histogram scoring rules repeatedly form differences of large, nearly-equal
    quantities — CRPS/energy ``term1 - term2``, variance ``E[X²] - E[X]²``,
    CDE ``∫g² - 2g(y)``, DPD ``∫f^{1+β} - point``.  Evaluated in float32 these
    suffer catastrophic cancellation (observed: CRPS down to ~ -33 on sharp,
    large-scale histograms), violating mathematical guarantees such as
    "energy score ≥ 0" or "variance ≥ 0".  Computing in float64 restores them.

    Integer/index tensors (bin indices, sample indices) and non-tensor
    arguments are passed through unchanged.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def cast(x):
                if isinstance(x, torch.Tensor) and x.is_floating_point():
                    return x.to(dtype)
                return x
            new_args = tuple(cast(a) for a in args)
            new_kwargs = {k: cast(v) for k, v in kwargs.items()}
            return func(*new_args, **new_kwargs)
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_metrics(dist: DistributionPrediction, y_true: np.ndarray) -> dict:
    """All metrics from a DistributionPrediction."""
    return {
        **compute_point_metrics(y_true, dist.mean),
        **compute_scoring_rules(dist, y_true),
    }


def compute_point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE, RMSE, R²."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2":   float(r2_score(y_true, y_pred)),
    }


@force_precision(torch.float64)
def compute_dpd_scores(probas: torch.Tensor, bin_widths: torch.Tensor,
                       g_y: torch.Tensor, betas: list, shared: bool,
                       density_integral=None) -> dict:
    """
    Density Power Divergence (DPD) scoring rule for histogram-based predictive densities.

    For a predictive density f and parameter β>0:
        S_β(f, y) = ∫ f(t)^{1+β} dt - (1 + 1/β) f(y)^β

    Propriety of this rule is a statement about a *single* density f appearing in
    both terms, so the integral and the point term must be two functionals of the
    same f.  ``g_y`` supplies ``f(y)`` and ``density_integral`` supplies
    ``∫ f^{power}``; the caller is responsible for taking both from one density
    (see ``_density_terms``).

    Parameters
    ----------
    density_integral : callable(power) -> (n_samples,) tensor, optional
        Returns ``∫ f(t)^{power} dt`` per sample.  Defaults to the
        piecewise-constant histogram density ``f_hist = p_k / w_k``, for which
        the integral is available in closed form:
        ``∫ f_hist^{1+β} = ∑_k p_k^{1+β} / w_k^{β}``.  The production path passes
        the integral of ``unified_bin_density``, which is that same closed form.

    Returns mean DPD across samples for each β as keys `dpd_beta_{β}`.
    """
    results = {}
    # Broadcast-ready bin widths: (1, n_bins) or (n_samples, n_bins)
    bw = bin_widths[None, :] if shared else bin_widths

    # Prevent division by zero
    eps = 100*torch.finfo(probas.dtype).eps
    g_y = g_y.clamp(min=eps)

    if density_integral is None:
        bw_safe = bw.clamp(min=eps)
        def density_integral(power):
            # ∫ f_hist^{power} dt = ∑_k p_k^{power} / w_k^{power-1}
            return (probas.pow(power) / bw_safe.pow(power - 1.0)).sum(dim=-1)

    for beta in betas:
        if beta < 0:
            raise ValueError("DPD beta must be >= 0")

        # β -> 0 limit recovers the (negative) log score up to an additive constant.
        if abs(beta) < 1e-12:
            loss = -torch.log(g_y)
        else:
            # Integral term: ∫ f^{1+β}
            integral = density_integral(1.0 + beta)

            # Point evaluation term: (1 + 1/β) * f(y)^β
            point_term = (1.0 + 1.0 / beta) * g_y.pow(beta)

            loss = integral - point_term

        results[f"dpd_beta_{beta}"] = loss.mean().item()

    return results

@force_precision(torch.float64)
def compute_energy_score_histogram_corrected(
        probas: torch.Tensor, 
        bin_mids: torch.Tensor, 
        bin_widths: torch.Tensor, 
        y: torch.Tensor, 
        betas: list = [0.2, 0.5, 1.0, 1.5, 2.0]
    ) -> dict:
        """
        Computes the Energy Score with exact uniform interval-correction.
        At beta=1.0, this mathematically equals the exact continuous CRPS.

        Runs in float64 (see ``force_precision``): term1 - term2 is a difference
        of large, nearly-equal values whose float32 cancellation can drive the
        (non-negative) energy score / CRPS below zero. The per-sample clamp below
        is a final guard restoring the mathematical guarantee score >= 0.
        """
        device = probas.device
        n_samples, n_bins = probas.shape
        shared = (bin_mids.ndim == 1)
        
        mids_ext = bin_mids[None, :] if shared else bin_mids
        widths_ext = bin_widths[None, :] if shared else bin_widths
        
        # Define bin edges for the exact integral
        left_edges = mids_ext - widths_ext / 2.0
        right_edges = mids_ext + widths_ext / 2.0
        
        # Distance from edges to target y
        u_l = left_edges - y[:, None]
        u_r = right_edges - y[:, None]

        results = {}

        # Non-negativity (and the clamp(min=0) guard below) requires ||·||^beta
        # to be conditionally negative definite, which holds only for beta in
        # (0, 2].  Outside this range the energy score can be legitimately
        # negative and clamping would corrupt it — reject all betas up-front.
        for beta in betas:
            if not (0.0 < beta <= 2.0):
                raise ValueError(f"Energy score beta must lie in (0, 2]; got beta={beta}.")

        eps = 100 * torch.finfo(probas.dtype).eps
        # A bin with (near-)zero width is a Dirac point mass at its midpoint,
        # not a uniform slab.  Detect these once (β-independent) so we can use
        # the correct point-mass distance instead of the degenerate integral.
        zero_width = widths_ext <= eps

        for beta in betas:
            # ---- Term 1: E|X - y|^beta ----
            # For a uniform bin this is the exact integral of |x - y|^beta.
            # For a zero-width (point-mass) bin the integral is 0/0; the correct
            # limit is the point-mass value |mid - y|^beta.
            numerator = u_r * u_r.abs().pow(beta) - u_l * u_l.abs().pow(beta)
            integral_d = numerator / (widths_ext.clamp(min=eps) * (beta + 1.0))
            # Point-mass distance for zero-width bins: |mid - y|^beta.
            if zero_width.any():
                point_d = (mids_ext - y[:, None]).abs().pow(beta)
                expected_d = torch.where(zero_width, point_d, integral_d)
            else:
                expected_d = integral_d
            term1 = (probas * expected_d).sum(dim=-1)

            # ---- Term 2: 0.5 * E|X - X'|^beta ----
            if shared:
                D = (bin_mids[:, None] - bin_mids[None, :]).abs()
                if beta != 1.0:
                    D = D.pow(beta)
                
                # Diagonal Correction (The Histogram Spirit Fix)
                diag_corr = (2.0 * bin_widths.pow(beta)) / ((beta + 1.0) * (beta + 2.0))
                D.diagonal().copy_(diag_corr)
                
                term2 = 0.5 * torch.einsum("si,ij,sj->s", probas, D, probas)
            else:
                # The per-sample pairwise term materialises a (chunk, n_bins,
                # n_bins) tensor; with the fine grids some models emit (n_bins
                # in the thousands) a fixed chunk of 256 rows overflows GPU
                # memory. Size the chunk so that intermediate stays within a
                # fixed element budget (>=1 row so progress is guaranteed).
                elem_budget = 64_000_000  # ~0.5 GiB in float64 for the (chunk,n_bins,n_bins) tensor
                chunk_size = max(1, min(256, elem_budget // max(1, n_bins * n_bins)))
                # Write the result in place and lets each chunk's temporaries be freed
                # before the next iteration allocates.
                term2 = torch.empty(n_samples, dtype=probas.dtype, device=device)
                for i in range(0, n_samples, chunk_size):
                    end = min(i + chunk_size, n_samples)
                    p_c = probas[i:end]
                    m_c = bin_mids[i:end]
                    w_c = bin_widths[i:end]

                    # (chunk, n_bins, n_bins) — the single largest allocation.
                    Dc = (m_c.unsqueeze(2) - m_c.unsqueeze(1)).abs()
                    if beta != 1.0:
                        Dc = Dc.pow(beta)

                    # Overwrite the diagonal via a view (no index tensor alloc).
                    d_corr = (2.0 * w_c.pow(beta)) / ((beta + 1.0) * (beta + 2.0))
                    Dc.diagonal(dim1=1, dim2=2).copy_(d_corr)

                    # einsum has no out=; copy the (chunk,) result into the
                    # preallocated buffer, then free Dc before the next chunk.
                    torch.mul(torch.einsum("ci,cij,cj->c", p_c, Dc, p_c),
                              0.5, out=term2[i:end])
                    del Dc

            # Average over samples (clamp per-sample: energy score / CRPS is
            # non-negative by definition; any sub-zero value is numerical error).
            results[f"energy_score_beta_{beta}"] = (term1 - term2).clamp(min=0).mean().item()

        return results

def compute_scoring_rules(dist: DistributionPrediction, y_true: np.ndarray) -> dict:
    """Compute all distributional scoring rules from a DistributionPrediction using PyTorch.

    Returns keys: crps, log_score, sharpness, dispersion,
                  coverage_90, interval_score_90,
                  coverage_95, interval_score_95,
                  crls,
                  wcrps_left, wcrps_right, wcrps_center,
                  energy_score_beta_{0.5,1.0,1.5,2.0}.
    """
    # Compute every scoring rule in float64 (enforced by @force_precision on
    # _compute_scoring_rules_torch).  Several rules form differences of large,
    # nearly-equal terms (variance E[X²] - E[X]², CRPS term1 - term2,
    # CDE ∫g² - 2g(y)); float32 cancellation there breaks guarantees such as
    # variance >= 0 / CRPS >= 0.
    # Build tensors on the compute device *first*, then let force_precision
    # upcast in-place
    _device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probas     = torch.as_tensor(dist.probas,        device=_device)
    bin_edges  = torch.as_tensor(dist.bin_edges,     device=_device)
    bin_mids   = torch.as_tensor(dist.bin_midpoints, device=_device)
    y          = torch.as_tensor(np.array(y_true, dtype=float), device=_device)  # np.array copies -> writable tensor
    shared     = bin_edges.ndim == 1

    logger.debug(
        "compute_scoring_rules: n_samples=%d  n_bins=%d  shared=%s",
        probas.shape[0], probas.shape[1], shared,
    )

    t0 = time.perf_counter()
    result = _compute_scoring_rules_torch(probas, bin_edges, bin_mids, y, shared)
    logger.debug("  torch backend      %.4fs (device=%s)",
                 time.perf_counter() - t0,
                 "cuda" if torch.cuda.is_available() else "cpu")
    return result



# ---------------------------------------------------------------------------
# PyTorch (GPU) implementation helpers
# ---------------------------------------------------------------------------

@force_precision(torch.float64)
def _interval(alpha, cdf, bin_edges, y, n_samples, n_bins, device, shared, y_bin, ns_idx):
    """Compute interval score and coverage for a given alpha level.
    
    Parameters
    ----------
    alpha : float
        Confidence level (e.g., 0.10 for 90% CI, 0.05 for 95% CI)
    cdf : torch.Tensor
        Cumulative distribution function (n_samples, n_bins)
    bin_edges : torch.Tensor
        Bin edges (n_bins+1,) or (n_samples, n_bins+1)
    y : torch.Tensor
        Target values (n_samples,)
    n_samples, n_bins, device, shared, y_bin, ns_idx : context variables
    
    Returns
    -------
    interval_score : float
        Mean interval score across samples
    coverage : float
        Empirical coverage (fraction of samples where y is in interval)
    """
    lower_q, upper_q = alpha / 2.0, 1.0 - alpha / 2.0
    if shared:
        n_e = len(bin_edges)
        # uint8 (1 byte/elem) is bit-identical to long (8 bytes/elem) for
        # argmax on a 0/1 tensor; saves 8× on the (n_samples, n_bins) bool cast.
        idx_l = (cdf >= lower_q).to(torch.uint8).argmax(dim=1).clamp(max=n_e - 1)
        idx_u = ((cdf >= upper_q).to(torch.uint8).argmax(dim=1) + 1).clamp(max=n_e - 1)
        lows  = bin_edges[idx_l]
        highs = bin_edges[idx_u]
    else:
        q_l = torch.full((n_samples, 1), lower_q, device=device)
        q_u = torch.full((n_samples, 1), upper_q, device=device)
        idx_l = torch.searchsorted(cdf.contiguous(), q_l).squeeze(1).clamp(0, n_bins - 1)
        idx_u = (torch.searchsorted(cdf.contiguous(), q_u).squeeze(1) + 1).clamp(0, n_bins - 1)
        lows  = bin_edges[ns_idx, idx_l]
        highs = bin_edges[ns_idx, idx_u]
    
    cov = ((y >= lows) & (y <= highs)).float().mean().item()
    sc  = ((highs - lows)
            + (2.0 / alpha) * (lows  - y).clamp(min=0)
            + (2.0 / alpha) * (y - highs).clamp(min=0))
    return sc.mean().item(), cov


@force_precision(torch.float64)
def compute_quantile_wcrps(cdf, bin_mids, y, n_samples, n_bins, device, shared):
    """Compute quantile-weighted CRPS with three weighting schemes.
    
    Quantile-Weighted CRPS (Gneiting & Ranjan 2011, Eq. 17):
        qwCRPS_v(F, y) = 2 ∫₀¹ ρ_α(y, q_α) v(α) dα
    
    where ρ_α(y, q) = (I[y ≤ q] − α)(q − y) is the pinball/check function.
    
    Weight functions (Table 1, Gneiting & Ranjan 2011):
        left-tail:  v(α) = (1−α)²      (emphasizes underprediction)
        right-tail: v(α) = α²          (emphasizes overprediction)
        center:     v(α) = α(1−α)      (balanced)
    
    Returns
    -------
    dict with keys: wcrps_left, wcrps_right, wcrps_center
    """
    alphas_qw = torch.linspace(0.01, 0.99, 99, device=device)   # (A,)
    d_alpha   = 1.0 / (len(alphas_qw) + 1)                       # ≈ 0.01

    # Invert the CDF: for each sample i and level α_j find the smallest bin k
    # with cdf[i, k] >= α_j.  Expand alphas to (n_samples, A) so searchsorted
    # can match the (n_samples, n_bins) cdf row-by-row.
    alphas_expanded = alphas_qw[None, :].expand(n_samples, -1).contiguous()  # (n_samples, A)
    idx_q = torch.searchsorted(cdf.contiguous(), alphas_expanded).clamp(0, n_bins - 1)

    if shared:
        q_a = bin_mids[idx_q]                    # (n_samples, A)
    else:
        q_a = torch.gather(bin_mids, 1, idx_q)   # (n_samples, A)

    # Pinball loss per sample and quantile level: 2(I[y ≤ q_α] − α)(q_α − y)
    pinball = (
        2.0
        * ((y[:, None] <= q_a).float() - alphas_qw[None, :])
        * (q_a - y[:, None])
    )                                                              # (n_samples, A)

    v_left   = (1.0 - alphas_qw).pow(2)                          # (A,)
    v_right  = alphas_qw.pow(2)
    v_center = alphas_qw * (1.0 - alphas_qw)

    wcrps_left   = (pinball * v_left[None, :]).sum(dim=-1).mean().item() * d_alpha
    wcrps_right  = (pinball * v_right[None, :]).sum(dim=-1).mean().item() * d_alpha
    wcrps_center = (pinball * v_center[None, :]).sum(dim=-1).mean().item() * d_alpha
    
    return {
        "wcrps_left":   wcrps_left,
        "wcrps_right":  wcrps_right,
        "wcrps_center": wcrps_center,
    }


@force_precision(torch.float64)
def compute_crls(cdf, bin_widths, y_bin, n_bins, device, eps, shared):
    """Compute Continuous Ranked Logarithmic Score (CRLS).
    
    CRLS = -sum_k w_k * [I(k>=target)*log(CDF_k) + I(k<target)*log(1-CDF_k)]
    
    This is bin-width-weighted cross-entropy between predicted CDF and the
    target step-function CDF (point mass at y). Formula from finetuned_regressor.py.

    Notes
    -----
    **The ``eps`` clamp is an implicit regularizer, not just a numerical guard.**
    The integration domain is *not* truncated -- every bin still contributes.
    What ``cdf.clamp(eps, 1 - eps)`` does is *winsorize the integrand*: it caps
    the per-bin penalty at ``-log(eps)``.  With the production
    ``eps = 100 * finfo(float64).eps ~ 2.2e-14`` that ceiling is ~31.4 nats.
    Three consequences follow, and they are properties of the reported metric:

    1. **Saturation.**  A prediction that is confidently wrong beyond ``eps``
       scores identically to one wrong at 1e-300, so CRLS has no resolving power
       in precisely the regime an unclamped log score punishes most.
    2. **Grid-extent dependence.**  The worst case is
       ``sum_k w_k * (-log eps) = (z_max - z_min) * 31.4``, i.e. the penalty
       ceiling grows with the total width of the support.  CRLS values are not
       directly comparable across models whose bin grids span different ranges.

    Separately (and independent of the clamp): ``cdf`` is a raw ``cumsum`` of the
    PMF, so if ``sum_k p_k = 1 - delta`` every bin at or above the target absorbs
    ``-log(1 - delta) ~ delta``, biasing CRLS upward by ``~delta *`` grid width.
    Normalise ``probas`` upstream if that matters.
    
    Parameters
    ----------
    cdf : torch.Tensor
        Cumulative distribution function (n_samples, n_bins)
    bin_widths : torch.Tensor
        Bin widths, ``(n_bins,)`` if ``shared`` else ``(n_samples, n_bins)``.
        Broadcast to ``(1, n_bins)`` internally when shared.
    y_bin : torch.Tensor
        Bin index of target value (n_samples,)
    n_bins : int
        Number of bins
    device : torch.device
        Computation device
    eps : float
        CDF clamp; also sets the per-bin penalty ceiling at ``-log(eps)``.
        See Notes.
    shared : bool
        Whether grid is shared or per-sample
    
    Returns
    -------
    float
        Mean CRLS across samples
    """
    bw          = bin_widths[None, :] if shared else bin_widths  # (1|n, n_bins)
    bin_idx    = torch.arange(n_bins, device=device)[None, :]  # (1, n_bins)
    target_cdf = (bin_idx >= y_bin[:, None]).to(cdf.dtype)     # (n_samples, n_bins)
    cdf_c      = cdf.clamp(eps, 1 - eps)
    crls_bins  = target_cdf * (-torch.log(cdf_c)) + (1 - target_cdf) * (-torch.log1p(-cdf_c))
    crls       = (crls_bins * bw).sum(dim=-1).mean().item()
    return crls


@force_precision(torch.float64)
def compute_pit_ks(probas, cdf, bin_edges, bin_widths, y_bin, y, shared, ns_idx):
    """Compute PIT values and the Kolmogorov-Smirnov p-value vs. Uniform(0, 1).

    Probability Integral Transform (Dawid 1984; Diebold et al. 1998):
        p_t = F_t(x_t)
    If the predictive distributions F_t are ideal and continuous, the PIT
    values {p_t} are i.i.d. Uniform(0, 1).  We test that null hypothesis with
    a one-sample Kolmogorov-Smirnov test.

    For a histogram predictive density we treat each bin as having a uniform
    density (piecewise-linear CDF), so for y in bin k_y:
        F(y) = cdf_{k_y - 1} + p_{k_y} * (y - left_edge_{k_y}) / w_{k_y}
    Values outside the support are clamped to [0, 1].

    Returns
    -------
    dict with keys:
        pit_ks_stat : float    KS statistic (sup |F_emp(p) - p|)
        pit_ks_pvalue : float  Two-sided p-value vs. Uniform(0, 1)
    """
    eps = 100 * torch.finfo(probas.dtype).eps

    # Probability mass and width of the bin containing each y
    p_y = probas.gather(1, y_bin.unsqueeze(1)).squeeze(1)
    if shared:
        w_y = bin_widths[y_bin]
        left_y = bin_edges[y_bin]
        support_lo = bin_edges[0]
        support_hi = bin_edges[-1]
    else:
        w_y = bin_widths.gather(1, y_bin.unsqueeze(1)).squeeze(1)
        left_y = bin_edges.gather(1, y_bin.unsqueeze(1)).squeeze(1)
        support_lo = bin_edges[:, 0]
        support_hi = bin_edges[:, -1]

    # Cumulative mass strictly below the y-bin
    cdf_prev = cdf[ns_idx, y_bin] - p_y
    frac = ((y - left_y) / w_y.clamp(min=eps)).clamp(0.0, 1.0)
    pit = (cdf_prev + p_y * frac).clamp(0.0, 1.0)

    # y outside support -> clamp PIT to 0 / 1.
    # Scalar constants avoid allocating zeros_like / ones_like tensors.
    pit = torch.where(y <= support_lo, 0.0, pit)
    pit = torch.where(y >= support_hi, 1.0, pit)

    pit_np = pit.detach().cpu().numpy().astype(np.float64)
    ks = stats.kstest(pit_np, "uniform")
    return {
        "pit_ks_stat":   float(ks.statistic),
        "pit_ks_pvalue": float(ks.pvalue),
    }


@force_precision(torch.float64)
def unified_bin_density(probas, bin_widths, shared, eps):
    """Piecewise-constant density on bin grid: f_k = p_k / w_k.
    
    Returns
    -------
    f_bins : (n_samples, n_bins) density value of each bin.
    w_eff : (1, n_bins) or (n_samples, n_bins) width of each bin.
    """
    w_eff = bin_widths[None, :] if shared else bin_widths    # (1|n, n_bins)
    f_bins = probas / w_eff.clamp(min=eps)

    Z = (f_bins * w_eff).sum(dim=-1, keepdim=True)
    return f_bins / Z.clamp(min=eps), w_eff


def _density_terms(probas, cdf, bin_edges, bin_widths, bw, y, y_bin, shared, eps):
    """Build (g_y, density_integral) pair from unified_bin_density.
    
    Returns
    -------
    g_y : (n_samples,) pointwise density f(y).
    density_integral : callable(power) -> (n_samples,) integral of f^power.
    """
    f_bins, w_eff = unified_bin_density(probas, bin_widths, shared, eps)
    g_y = f_bins.gather(1, y_bin.unsqueeze(1)).squeeze(1)

    def density_integral(power):
        return (f_bins.pow(power) * w_eff).sum(dim=-1)

    return g_y, density_integral


@force_precision(torch.float64)
def compute_cde_loss(probas, bin_widths, g_y, bw, shared, density_integral=None):
    """Compute Continuous Density Estimation (CDE) Loss.
    
    From Izbicki and Lee (2016): "Nonparametric Conditional Density Estimation..."
    First derived 1980 (Rudemo): "Empirical Choice of Histograms and Kernel Density Estimators"
    
    General proper scoring rule for density comparison:
        L(f, g) = ∫∫ (f(z|x) - g(z|x))² dP(x) dz
                = ∫∫ f² dP(x)dz - 2∫∫ f·g dP(x)dz + ∫∫ g² dP(x)dz
    
    For scoring rules, drop constants independent of g:
        L_CDE(f, g) = ∫∫ g² dP(x)dz - 2∫∫ f·g dP(x)dz
    
    With empirical target f (point mass at y):
        ∫ g² dz  = ∫ [g(z)]² dz        (second moment of g over support)
        ∫ f·g dz = g(y)                 (density of g evaluated at y)
    
    Discretized form (on grid with bin widths w_k and grid PMF p_k):
        ∫ g² dz  ≈  ∑_k (p_k/w_k)² · w_k = ∑_k p_k² / w_k    (exact bin masses/widths)
        g(y)     ≈  p_{k_y} / w_{k_y}                        (forward difference)

    Relationship to DPD
    -------------------
    The CDE (integrated-squared-error / L²) loss is *identical* to the Density
    Power Divergence score at β = 1:
        S_{β=1}(g, y) = ∫ g(z)^{1+β} dz - (1 + 1/β) g(y)^β
                      = ∫ g² dz - 2 g(y).
    The production path therefore does not call this function at all — it reads
    ``cde_loss`` straight off ``dpd_beta_1.0`` so the two cannot drift apart.
    Kept as the standalone, directly readable form of the rule.

    Parameters
    ----------
    probas : torch.Tensor
        Probability masses per bin (n_samples, n_bins)
    bin_widths : torch.Tensor
        Bin widths (n_bins,) or (n_samples, n_bins)
    g_y : torch.Tensor
        Pointwise predictive density at the target, f(y) (n_samples,).
    bw : torch.Tensor
        Broadcast-ready bin widths
    shared : bool
        Whether grid is shared or per-sample
    density_integral : callable(power) -> (n_samples,) tensor, optional
        ``∫ f(t)^{power} dt`` per sample; must come from the same density as
        ``g_y``.  Defaults to the histogram density ``∑_k p_k² / w_k``.

    Returns
    -------
    float
        Mean CDE loss across samples
    """
    eps = 100*torch.finfo(probas.dtype).eps
    if density_integral is None:
        term1 = (probas.pow(2) / bw.clamp(min=eps)).sum(dim=-1)  # ∫ g² dz
    else:
        term1 = density_integral(2.0)                            # ∫ g² dz
    term2 = 2.0 * g_y.clamp(min=eps)                         # 2·g(y)
    cde_loss = (term1 - term2).mean().item()
    return cde_loss


@force_precision(torch.float64)
def _compute_scoring_rules_torch(probas, bin_edges, bin_mids, y, shared):
    """All scoring rules computed on GPU (or CPU) via PyTorch tensors.

    Note: `probas` are PMF values (probability mass per bin), i.e. for each
    sample the entries satisfy ∑_k p_k = 1 and represent P(z ∈ bin_k).
    To obtain a density at a bin midpoint divide by the bin width:
    density_k = p_k / w_k. Integrating densities over the grid then
    recovers 1: ∑_k density_k * w_k = 1.

    Inputs are float64 tensors (upcast by ``@force_precision``); here we only
    move them onto the compute device.
    """
    # Tensors are already on the target device (moved before the
    # force_precision upcast in compute_scoring_rules); .to(device) is a
    # no-op here but kept for safety when the function is called directly.
    device = probas.device

    n_samples, n_bins = probas.shape
    ns_idx = torch.arange(n_samples, device=device)

    bin_widths = torch.diff(bin_edges, dim=-1)           # (n_bins,) or (n_samples, n_bins)
    bw = bin_widths[None, :] if shared else bin_widths   # broadcast-ready

    cdf = torch.cumsum(probas, dim=-1)                   # (n_samples, n_bins)
    eps = 100*torch.finfo(probas.dtype).eps

    mids = bin_mids[None, :] if shared else bin_mids     # broadcast-ready

    # ---- bin index of each y (reused by log_score, CRLS) ----
    if shared:
        y_bin = torch.searchsorted(bin_edges[1:].contiguous(), y).clamp(0, n_bins - 1)
    else:
        y_bin = torch.searchsorted(
            bin_edges[:, 1:].contiguous(), y.unsqueeze(1)
        ).squeeze(1).clamp(0, n_bins - 1)

    # ---- Quantile-Weighted CRPS (Gneiting & Ranjan 2011, Eq. 17) ----
    qwcrps_result = compute_quantile_wcrps(cdf, bin_mids, y, n_samples, n_bins, device, shared)
    wcrps_left   = qwcrps_result["wcrps_left"]
    wcrps_right  = qwcrps_result["wcrps_right"]
    wcrps_center = qwcrps_result["wcrps_center"]

    # ---- DPD scores, incl. log_score (β=0) and cde_loss (β=1) ----
    # Reading all three off one call keeps them exactly consistent.  The density
    # terms (``f_bins`` / ``g_y`` / the ``density_integral`` closure) are each a
    # full (n_samples, n_bins) tensor; build and consume them inside this scope
    # so they are freed on return instead of being pinned alive through the
    # interval, energy-score and PIT sections below.
    def _dpd_block():
        # One piecewise-constant density on the bin grid supplies both terms, so
        # the two-term rules (cde_loss, dpd_beta_*) are self-consistent and hence
        # proper for it.  See ``unified_bin_density``.
        g_y, density_integral = _density_terms(
            probas, cdf, bin_edges, bin_widths, bw, y, y_bin, shared, eps
        )
        return compute_dpd_scores(probas, bin_widths, g_y,
                                  betas=sorted({*DPD_BETAS, 0.0, 1.0}),
                                  shared=shared,
                                  density_integral=density_integral)

    all_dpd = _dpd_block()
    dpd_scores = {f"dpd_beta_{b}": all_dpd[f"dpd_beta_{b}"] for b in DPD_BETAS}
    log_score = all_dpd["dpd_beta_0.0"]
    cde_loss = all_dpd["dpd_beta_1.0"]

    # ---- Sharpness & Dispersion (Tran et al. 2020) ----
    # Sharpness: mean of per-sample predictive std.  Dispersion: std of the
    # per-sample predictive std.  Scoped so the (n_samples, n_bins) products
    # ``probas * mids`` / ``probas * mids²`` are freed as soon as the two scalars
    # are read out.
    def _sharpness_dispersion():
        mean_  = (probas * mids).sum(dim=-1)
        var_   = ((probas * mids.pow(2)).sum(dim=-1) - mean_.pow(2)).clamp(min=0)
        std_per_sample = var_.sqrt()                          # (n_samples,)
        # Use unbiased=False to avoid torch warning when n_samples is small
        return std_per_sample.mean().item(), std_per_sample.std(unbiased=False).item()

    sharpness, dispersion = _sharpness_dispersion()

    # ---- Interval scores (shared path: vectorised; non-shared: searchsorted) ----
    # Coverage levels (%) from COVERAGE_LEVELS; significance alpha = 1 - level/100.
    interval_results = {}

    for cov_level in COVERAGE_LEVELS:
        alpha = 1.0 - cov_level / 100.0
        is_alpha, cov_alpha = _interval(alpha, cdf, bin_edges, y, n_samples, n_bins, device, shared, y_bin, ns_idx)
        interval_results[f"coverage_{cov_level}"] = cov_alpha
        interval_results[f"interval_score_{cov_level}"] = is_alpha

    # Every beta is computed independently inside
    # ``compute_energy_score_histogram_corrected`` (its per-beta result does not
    # depend on which other betas are requested), so a single batched call is
    # bit-identical to the per-beta calls -- and CRPS is exactly the β=1.0 energy
    # score, so we read it off here instead of paying for a second full pass over
    # the (chunk, n_bins, n_bins) distance matrices.
    energy_all = compute_energy_score_histogram_corrected(
        probas, bin_mids, bin_widths, y, betas=ENERGY_BETAS
    )
    energy_scores = [energy_all[f"energy_score_beta_{beta}"] for beta in ENERGY_BETAS]
    crps = energy_all["energy_score_beta_1.0"]

    # ---- CRLS ----
    crls = compute_crls(cdf, bin_widths, y_bin, n_bins, device, eps, shared)

    # ---- PIT KS test (Dawid 1984; Diebold et al. 1998) ----
    pit_ks = compute_pit_ks(probas, cdf, bin_edges, bin_widths, y_bin, y, shared, ns_idx)

    return {
        "crps":              crps,
        "log_score":         log_score,
        "sharpness":         sharpness,
        "dispersion":        dispersion,
        **interval_results,
        "crls":              crls,
        "cde_loss":          cde_loss,
        "pit_ks_stat":       pit_ks["pit_ks_stat"],
        "pit_ks_pvalue":     pit_ks["pit_ks_pvalue"],
        "wcrps_left":        wcrps_left,
        "wcrps_right":       wcrps_right,
        "wcrps_center":      wcrps_center,
        **{f"energy_score_beta_{b}": v for b, v in zip(ENERGY_BETAS, energy_scores)},
        **dpd_scores,
    }

"""Pure statistical primitives ported from peak_calling.R, with no BED/model
coupling: a Laplace noise model (used to derive the significance floor
`min_score`), a genome-wide neighboring-point autocorrelation matrix, and a
multivariate-Laplace tail-probability integral (the per-summit p-value used
before BH-FDR adjustment in pydreg.peaks).
"""

import math
import time

import numba
import numpy as np
from scipy.stats import _qmvnt


def qlaplace(p, m=0.0, s=1.0):
    """Quantile function of a Laplace(m, s) distribution (R's rmutil::qlaplace)."""
    p = np.asarray(p, dtype=float)
    return np.where(p < 0.5, m + s * np.log(2 * p), m - s * np.log(2 * (1 - p)))


def get_laplace_sigma(ypred, y=None):
    """Estimates the scale (sigma) of a Laplace noise model target = ypred + z,
    z ~ Laplace(0, sigma), from residuals ypred - y (y defaults to all zeros --
    used on the negative-score tail, which is assumed pure noise around 0)."""
    ypred = np.asarray(ypred, dtype=float)
    y = np.zeros_like(ypred) if y is None else np.asarray(y, dtype=float)

    valid = ~(np.isnan(ypred) | np.isnan(y))
    diff = ypred[valid] - y[valid]
    n = diff.shape[0]
    if n == 0:
        return np.nan

    std = np.sqrt(2) * (np.sum(np.abs(diff)) / n)
    outlier = np.abs(diff) > 5 * std
    if np.any(outlier):
        mae = np.sum(np.abs(diff)[~outlier])
        denom = n - np.sum(outlier)
    else:
        mae = np.sum(np.abs(diff))
        denom = n
    if denom == 0:
        return np.nan
    return mae / denom


def get_laplace_quantile(sigma, p=0.05):
    """The (1 - p/2) quantile of a Laplace(0, sigma) distribution -- used as
    the significance floor `min_score` (e.g. p=0.001 for a 99.9th-percentile
    style threshold on the noise model)."""
    return float(qlaplace(1 - p / 2, 0.0, sigma))


def build_cormat(starts, scores, dist=20, order=5):
    """One genome-wide order x order Toeplitz covariance matrix of
    neighboring-point score autocorrelation, from peak_calling.R's
    build_cormat(). `starts`/`scores` are the position/score columns of the
    (densified) informative-position table.

    Only pairs of positions exactly `dist` bp apart (i.e. adjacent points in
    the 10bp-densified table, 2 rows apart) are used to estimate the lag-1..4
    autocorrelation `rho`; every other such pair is then dropped so the
    correlation samples don't overlap. `sigma2` (the outlier-truncated
    variance of ALL scores) scales the correlation into a covariance."""
    starts = np.asarray(starts, dtype=float)
    scores = np.asarray(scores, dtype=float)
    n = starts.shape[0]

    gap = starts[2:] - starts[:-2]
    pair_idx = np.nonzero(gap == dist)[0] + 2
    cor_scores = scores[pair_idx][0::2]

    m = cor_scores.shape[0]
    rho = [1.0]
    for lag in range(1, 5):
        rho.append(np.corrcoef(cor_scores[lag:], cor_scores[: m - lag])[0, 1])
    rho = np.array(rho)

    truncated = scores[np.abs(scores) <= 0.5]
    sigma2 = np.var(truncated, ddof=1)

    idx = np.arange(order)
    lag_mat = np.abs(idx[:, None] - idx[None, :])
    return sigma2 * rho[lag_mat]


def _r_seq(from_, to, by):
    # Mirrors R's seq(from, to, by): count via round(), not cumulative float
    # addition, to avoid arange()'s off-by-one drift at endpoints.
    n = round((to - from_) / by) + 1
    return from_ + by * np.arange(n)


# The z-integration grid is a pure constant (independent of x/cor_mat) --
# computed once, lazily, and reused by every pmv_laplace call in the process.
_Z_GRID = None
_Z_WIDTHS = None
# R's mvtnorm::pmvnorm() (called with no `algorithm=` override by peak_calling.R's
# pmvLaplace()) runs on mvtnorm's own GenzBretz(maxpts=25000, abseps=1e-3, releps=0)
# default -- confirmed from mvtnorm's R source (R/mvt.R) and its algorithms.Rd doc.
# SciPy's own unset defaults (maxpts=1e6*dim, abseps=releps=1e-5) are 200x/100x
# tighter than that -- i.e. far MORE precise than what the R reference this is
# ported from ever actually computed, which is why leaving these unset was both
# unfaithful and (consequently) the dominant cost in peak calling. releps has no
# effect on SciPy's actual d>=3 dispatch either way, so R's releps=0 is moot.
_PMV_CDF_MAXPTS = 25000
_PMV_CDF_EPS = 1e-3
_pmv_cdf_version = 0


def set_pmv_laplace_cdf_options(maxpts=25000, eps=1e-3):
    """Set SciPy multivariate-normal CDF controls used inside pmv_laplace().

    maxpts=25000, eps=1e-3 (for both abseps and releps) are R's own
    mvtnorm::pmvnorm()/GenzBretz() defaults -- the exact precision the
    pretrained model's p-values were produced under, not an approximation.
    Lower values trade fidelity for further speed; higher values exceed
    what R's own reference implementation ever computed.
    """
    global _PMV_CDF_MAXPTS, _PMV_CDF_EPS, _pmv_cdf_version
    if maxpts is not None:
        maxpts = int(maxpts)
        if maxpts < 1:
            raise ValueError(f"pmv_laplace_cdf_maxpts must be >= 1, got {maxpts}")
    eps = float(eps)
    if eps <= 0:
        raise ValueError(f"pmv_laplace_cdf_eps must be > 0, got {eps}")
    _PMV_CDF_MAXPTS = maxpts
    _PMV_CDF_EPS = eps
    _pmv_cdf_version += 1


def get_pmv_laplace_cdf_options():
    return {"maxpts": _PMV_CDF_MAXPTS, "eps": _PMV_CDF_EPS}


def _z_grid():
    global _Z_GRID, _Z_WIDTHS
    if _Z_GRID is None:
        z = np.concatenate(
            [
                [1e-100],
                10.0 ** np.arange(-20, -1, 1),
                _r_seq(0.02, 1.0, 0.04),
                _r_seq(1.0, 10.0, 0.2),
                _r_seq(10.5, 100.0, 0.5),
                10.0 ** np.arange(3, 21, 1),
                [1e100],
            ]
        )
        _Z_GRID = z
        _Z_WIDTHS = z[1:] - z[:-1]
    return _Z_GRID, _Z_WIDTHS


# SciPy's Genz-Bretz CDF integration (scipy.stats._qmvnt, used internally by
# multivariate_normal.cdf for d>=3) builds a QMC lattice (_cbc_lattice) on
# every box-probability evaluation, but that lattice depends only on
# (n_dim, n_qmc_samples) -- and for this codebase's fixed order-5 cor_mat /
# maxpts=25000 usage, scipy's adaptive point-growth loop (_qauto) always
# converges on its first try with an identical n_qmc_samples, verified
# empirically (see docs/PERF_LOG.md) to be a single reused constant across
# every eval, in every call, in a whole process's run -- so this is a pure
# performance win with zero numerical risk: it only reuses exactly the
# value SciPy would otherwise have recomputed, never approximates anything,
# and never touches the randomized QMC shift/kernel evaluation itself
# (scipy's `_qmvn` draws that fresh, per eval, from its own RNG state,
# entirely independent of this cached deterministic setup step) -- so
# pmv_laplace's existing run-to-run QMC noise is unaffected.
#
# _permuted_cholesky (the other per-eval setup step, box-reordering +
# Cholesky factorization) was tried the same way and DELIBERATELY DROPPED:
# although its *output* is highly repetitive (empirically only 2-4 distinct
# results across a whole 291-point z-grid), its *inputs* (the box bounds
# themselves, continuously rescaled by 1/sqrt(z0) per z-grid point) are
# essentially always unique -- so caching keyed on the literal input values
# almost never hits (measured: 290/291 calls still recomputed) despite the
# real redundancy in the output. Safely exploiting that redundancy would
# require cheaply detecting which permutation a given box falls into
# *without* running the expensive pivot-selection algorithm -- i.e. partially
# reimplementing scipy's private algorithm, not just wrapping it in a cache.
# That's real reimplementation risk (the ~7-9% minority of evals that use a
# different permutation are not numerically negligible -- an incorrect
# quantized/approximate cache key could silently misclassify some of them),
# better scoped to a from-scratch implementation (where this logic gets
# reimplemented deliberately and validated end-to-end) than bolted on here
# as a "safe wrapper."
#
# _cbc_lattice is a private SciPy API (`scipy.stats._qmvnt`, underscore-
# prefixed). Saving the original below fails loudly (AttributeError at
# import time) if a future SciPy version renames/removes it, rather than
# silently reverting to uncached-but-still-correct behavior.
_orig_cbc_lattice = _qmvnt._cbc_lattice

_cbc_lattice_cache = {}


def _cached_cbc_lattice(n_dim, n_qmc_samples):
    key = (n_dim, n_qmc_samples)
    cached = _cbc_lattice_cache.get(key)
    if cached is None:
        cached = _orig_cbc_lattice(n_dim, n_qmc_samples)
        _cbc_lattice_cache[key] = cached
    q, n_qmc_samples_actual = cached
    # Return a copy: the lattice array must never be mutated in place by a
    # caller, or every future cache hit would silently return corrupted data.
    return q.copy(), n_qmc_samples_actual


_qmvnt._cbc_lattice = _cached_cbc_lattice


# _permuted_cholesky (box-reordering + Cholesky factorization, run once per
# _qmvn call before the QMC kernel) is pure Python in SciPy -- unlike
# _qmvn_inner (the QMC kernel itself), which is already compiled Cython and
# for which a numba port was tried and found no win (see docs/PERF_LOG.md).
# Now that the small-start adaptive sample schedule above has shrunk the
# kernel's own cost dramatically, this previously-minor (~5-8%) setup step
# is proportionally dominant (~42% of pmv_laplace's time, measured). A numba
# port has real room to win here specifically because the original was never
# compiled at all. This is a line-for-line translation of SciPy's own
# `_permuted_cholesky` (read directly from the installed source) -- a
# deterministic pivoted-Cholesky algorithm with no randomization, so its
# output must match SciPy's own to near machine precision, not just "within
# QMC noise" (validated: bit-exact, max diff 0.0, across 200 random cases;
# see docs/PERF_LOG.md and the accompanying test).
@numba.njit(cache=True, inline="always")
def _ndtr(x):
    # inline="always": numba doesn't inline njit-to-njit calls by default,
    # and this runs inside _permuted_cholesky_numba's innermost loop --
    # measured ~4-5% faster this way (1.9us -> ~1.8us/call), bit-exact
    # either way since inlining is purely a compilation hint.
    return 0.5 * (1.0 + math.erf(x / 1.4142135623730951))


@numba.njit(cache=True)
def _permuted_cholesky_numba(covar, low, high, tol=1e-10):
    cho = covar.copy()
    new_lo = low.copy()
    new_hi = high.copy()
    n = cho.shape[0]

    dc = np.zeros(n)
    for i in range(n):
        dc[i] = math.sqrt(max(cho[i, i], 0.0))
        if dc[i] == 0.0:
            dc[i] = 1.0
    for i in range(n):
        new_lo[i] /= dc[i]
        new_hi[i] /= dc[i]
    for i in range(n):
        for j in range(n):
            cho[i, j] /= dc[j]
            cho[i, j] /= dc[i]

    y = np.zeros(n)
    sqtp = math.sqrt(2 * math.pi)
    for k in range(n):
        epk = (k + 1) * tol
        im = k
        ck = 0.0
        dem = 1.0
        lo_m = 0.0
        hi_m = 0.0
        for i in range(k, n):
            if cho[i, i] > tol:
                ci = math.sqrt(cho[i, i])
                s = 0.0
                if i > 0:
                    for j in range(k):
                        s += cho[i, j] * y[j]
                lo_i = (new_lo[i] - s) / ci
                hi_i = (new_hi[i] - s) / ci
                de = _ndtr(hi_i) - _ndtr(lo_i)
                if de <= dem:
                    ck = ci
                    dem = de
                    lo_m = lo_i
                    hi_m = hi_i
                    im = i
        if im > k:
            cho[im, im] = cho[k, k]
            for j in range(k):
                cho[im, j], cho[k, j] = cho[k, j], cho[im, j]
            for j in range(im + 1, n):
                cho[j, im], cho[j, k] = cho[j, k], cho[j, im]
            for j in range(k + 1, im):
                cho[j, k], cho[im, j] = cho[im, j], cho[j, k]
            new_lo[k], new_lo[im] = new_lo[im], new_lo[k]
            new_hi[k], new_hi[im] = new_hi[im], new_hi[k]
        if ck > epk:
            cho[k, k] = ck
            for j in range(k + 1, n):
                cho[k, j] = 0.0
            for i in range(k + 1, n):
                cho[i, k] /= ck
                for j in range(k + 1, i + 1):
                    cho[i, j] -= cho[i, k] * cho[j, k]
            if abs(dem) > tol:
                y[k] = (math.exp(-lo_m * lo_m / 2) - math.exp(-hi_m * hi_m / 2)) / (
                    sqtp * dem
                )
            else:
                y[k] = (lo_m + hi_m) / 2
                if lo_m < -10:
                    y[k] = hi_m
                elif hi_m > 10:
                    y[k] = lo_m
            for j in range(k + 1):
                cho[k, j] /= ck
            new_lo[k] /= ck
            new_hi[k] /= ck
        else:
            for i in range(k, n):
                cho[i, k] = 0.0
            y[k] = (new_lo[k] + new_hi[k]) / 2
    return cho, new_lo, new_hi


# Private SciPy API, same "fail loudly if renamed/removed" reasoning as
# _orig_cbc_lattice above.
_orig_permuted_cholesky = _qmvnt._permuted_cholesky
_qmvnt._permuted_cholesky = _permuted_cholesky_numba


# SciPy's own public adaptive driver (_qmvnt._qauto, used internally by
# multivariate_normal.cdf) always starts its search at
# min(maxpts, n_dim*1000) samples -- for this codebase's fixed 5-dim
# cor_mat, a hardcoded floor of ~5000 raw samples (n_qmc_samples~=701 per
# batch x 10 batches) *per box*, regardless of how far that box is from the
# 50/50 boundary. R's actual algorithm (mvtnorm's Fortran MVKBRV, read
# directly from mvtnorm's mvt.f) has no such floor: it grows through a fine
# table of lattice sizes starting near ~100-200 points for a problem this
# size, and stops as soon as the error target is met -- which is why R
# converges in a few hundred samples for most boxes while forcing SciPy's
# public API into the same target precision costs ~7000+ regardless.
#
# _qmvn_adaptive below reuses SciPy's own trusted `_qmvn` kernel (lattice
# construction, Cholesky reordering, randomized QMC evaluation) completely
# unchanged -- it only changes *how many* samples are requested before the
# first convergence check, mirroring the *shape* of R's schedule (start
# small, grow only as needed) without reusing any of R's actual lattice-
# generator data. Same stopping condition as SciPy's own _qauto (est_error
# <= abseps), so the same final-precision guarantee holds regardless of
# starting point. Validated empirically (docs/PERF_LOG.md) across 25 random
# order-5 cor_mat/box cases: results agree with the SciPy-floor approach to
# within ordinary QMC noise (max diff ~5e-5) while cutting wall-clock time
# per pmv_laplace call by ~4.3x.
_QAUTO_START = 150


def _qmvn_adaptive(cor_mat, low, high, rng, maxpts, abseps, n_batches=10):
    """Box probability P(low < X < high), X ~ N(0, cor_mat), via SciPy's
    private randomized-QMC kernel (_qmvnt._qmvn) driven by our own small-
    start adaptive loop (see _QAUTO_START above) instead of SciPy's public
    _qauto. Returns (prob, n_calls) -- n_calls counts how many times the
    underlying kernel was invoked (usually 1, occasionally more for boxes
    that need extra growth rounds to hit abseps), used for profiling. Only
    valid for cor_mat.shape[0] >= 3 (this codebase's build_cormat always
    produces order=5)."""
    n_samples = 0
    n_calls = 0
    mi = _QAUTO_START
    prob = 0.0
    est_error = 1.0
    while est_error > abseps and n_samples < maxpts:
        mi = round(np.sqrt(2) * mi)
        pi, ei, ni = _qmvnt._qmvn(mi, cor_mat, low, high, rng=rng, n_batches=n_batches)
        n_samples += ni
        n_calls += 1
        wt = 1.0 / (1 + (ei / est_error) ** 2)
        prob += wt * (pi - prob)
        est_error = np.sqrt(wt) * ei
    return prob, n_calls


_pmv_laplace_profile = {"calls": 0, "seconds": 0.0, "cdf_evals": 0}


def reset_pmv_laplace_profile():
    _pmv_laplace_profile["calls"] = 0
    _pmv_laplace_profile["seconds"] = 0.0
    _pmv_laplace_profile["cdf_evals"] = 0


def get_pmv_laplace_profile():
    return dict(_pmv_laplace_profile)


def pmv_laplace(x, cor_mat):
    """Tail probability of a multivariate-Laplace null (covariance cor_mat)
    inside the symmetric box [-|x|, |x|]^d, from peak_calling.R's pmvLaplace().
    A multivariate Laplace is a Gaussian variance-mixture (Laplace =
    sqrt(Z)*N(0,Sigma), Z~Exp(1)), so this numerically integrates the
    Gaussian box probability (scipy's multivariate_normal.cdf with
    lower_limit, matching R's mvtnorm::pmvnorm) over a log-then-linear grid
    of Z values.

    Faithfully reproduces a genuine bug in the original R: it computes both a
    left-endpoint Riemann sum (`p_max`) and a right-endpoint one (`p_min`)
    and returns `mean(p.max, p.min)` -- but R's `mean(x, trim)` binds the
    second positional argument to `trim`, not to a second value to average,
    so for two scalars this silently returns `p.max` alone, never averaging
    in `p.min` (verified empirically: `mean(5, 3)` returns `5` in R). The
    pretrained model's expected p-values were produced by this exact
    (unaveraged) function, so this returns `p_max` only, not the mean --
    not a bug to fix here.

    Perf note: the z-integration grid and SciPy's QMC lattice construction
    (see the `_cbc_lattice` caching above) are cached across calls -- a whole
    call_peaks run shares one cor_mat, and the grid/lattice are pure
    constants -- since this runs once per peak summit, genome-wide. The up
    to 291 box-CDF evaluations themselves are driven by `_qmvn_adaptive`'s
    small-start schedule (see its docstring) rather than SciPy's public
    `multivariate_normal.cdf`, which forces every evaluation through a
    ~5000-sample floor regardless of how easy the box is -- this was the
    dominant cost in peak calling and the main reason it ran much slower
    than R's own pmvnorm(), which has no such floor. `cdf_evals` in the
    profile now counts actual underlying `_qmvn` kernel invocations
    (usually one per box, occasionally more), not a fixed 291 -- a more
    direct measure of real work than the previous batched-call version's
    count."""
    t0 = time.perf_counter()
    cdf_evals = 0
    try:
        x = np.asarray(x, dtype=float)
        abs_x = np.abs(x)
        rng = np.random.default_rng()

        p_norm, n_calls = _qmvn_adaptive(
            cor_mat, -abs_x, abs_x, rng, _PMV_CDF_MAXPTS, _PMV_CDF_EPS
        )
        cdf_evals += n_calls
        if p_norm > 0.99:
            return p_norm

        z, widths = _z_grid()
        p0 = np.empty_like(z)
        for i, z0 in enumerate(z):
            scaled = abs_x / np.sqrt(z0)
            pi, n_calls = _qmvn_adaptive(
                cor_mat, -scaled, scaled, rng, _PMV_CDF_MAXPTS, _PMV_CDF_EPS
            )
            p0[i] = pi * np.exp(-z0)
            cdf_evals += n_calls
        p_max = min(float(np.sum(widths * p0[:-1])), 1.0)
        return p_max
    finally:
        _pmv_laplace_profile["calls"] += 1
        _pmv_laplace_profile["seconds"] += time.perf_counter() - t0
        _pmv_laplace_profile["cdf_evals"] += cdf_evals

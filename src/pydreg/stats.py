"""Pure statistical primitives ported from peak_calling.R, with no BED/model
coupling: a Laplace noise model (used to derive the significance floor
`min_score`), a genome-wide neighboring-point autocorrelation matrix, and a
multivariate-Laplace tail-probability integral (the per-summit p-value used
before BH-FDR adjustment in pydreg.peaks).
"""

import numpy as np
import time
from scipy.stats import multivariate_normal


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

    std = np.sqrt(2) * (np.sum(np.abs(diff)) / n)
    outlier = np.abs(diff) > 5 * std
    if np.any(outlier):
        mae = np.sum(np.abs(diff)[~outlier])
        denom = n - np.sum(outlier)
    else:
        mae = np.sum(np.abs(diff))
        denom = n
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


# Single-slot cache: a whole call_peaks run reuses one genome-wide cor_mat
# object, so identity comparison against the last-seen array is sufficient
# (and avoids rebuilding the frozen multivariate_normal on every one of the
# thousands of per-summit pmv_laplace calls in a run).
_mvn_cache = {"cor_mat": None, "mvn": None}
_pmv_laplace_profile = {"calls": 0, "seconds": 0.0}


def _cached_mvn(cor_mat):
    if _mvn_cache["cor_mat"] is not cor_mat:
        d = cor_mat.shape[0]
        _mvn_cache["mvn"] = multivariate_normal(mean=np.zeros(d), cov=cor_mat, allow_singular=True)
        _mvn_cache["cor_mat"] = cor_mat
    return _mvn_cache["mvn"]


def reset_pmv_laplace_profile():
    _pmv_laplace_profile["calls"] = 0
    _pmv_laplace_profile["seconds"] = 0.0


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

    Perf note: the frozen `multivariate_normal` (keyed on cor_mat's identity)
    and the z-integration grid are cached across calls -- a whole call_peaks
    run shares one cor_mat, and the grid is a pure constant -- since this
    runs once per peak summit, genome-wide."""
    t0 = time.perf_counter()
    try:
        x = np.asarray(x, dtype=float)
        mvn = _cached_mvn(cor_mat)

        abs_x = np.abs(x)
        p_norm = float(mvn.cdf(abs_x, lower_limit=-abs_x))
        if p_norm > 0.99:
            return p_norm

        z, widths = _z_grid()
        p0 = np.array(
            [
                float(mvn.cdf(abs_x / np.sqrt(z0), lower_limit=-abs_x / np.sqrt(z0)))
                * np.exp(-z0)
                for z0 in z
            ]
        )
        p_max = min(float(np.sum(widths * p0[:-1])), 1.0)
        return p_max
    finally:
        _pmv_laplace_profile["calls"] += 1
        _pmv_laplace_profile["seconds"] += time.perf_counter() - t0

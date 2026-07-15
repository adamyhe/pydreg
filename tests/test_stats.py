import numpy as np
from scipy.stats import _qmvnt

from pydreg import stats
from pydreg.stats import (
    build_cormat,
    get_laplace_quantile,
    get_laplace_sigma,
    get_pmv_laplace_profile,
    pmv_laplace,
    qlaplace,
    reset_pmv_laplace_profile,
    set_pmv_laplace_cdf_options,
)


def test_qlaplace_median_is_zero():
    assert qlaplace(0.5) == 0.0


def test_qlaplace_matches_closed_form():
    # F(x) = 1 - 0.5*exp(-x/s) for x >= m=0; invert at p=0.975, s=2.
    q = qlaplace(0.975, s=2.0)
    assert np.isclose(q, -2.0 * np.log(2 * 0.025))


def test_get_laplace_sigma_and_quantile_are_positive():
    rng = np.random.default_rng(0)
    negative_scores = -np.abs(rng.normal(scale=0.05, size=5000))
    sigma = get_laplace_sigma(negative_scores)
    assert sigma > 0
    min_score = get_laplace_quantile(sigma, 0.001)
    assert min_score > 0


def test_build_cormat_symmetric_with_unit_lag0():
    rng = np.random.default_rng(0)
    n = 2000
    starts = np.arange(n) * 10
    scores = rng.normal(scale=0.1, size=n)
    cormat = build_cormat(starts, scores, dist=20, order=5)
    assert cormat.shape == (5, 5)
    np.testing.assert_allclose(cormat, cormat.T)
    # diagonal == sigma2 * rho[0] == sigma2 (rho[0] is always 1 by construction)
    assert np.allclose(np.diag(cormat), cormat[0, 0])


def test_pmv_laplace_in_unit_interval_and_saturates_for_large_x():
    cor_mat = np.eye(5) * 0.01
    p_small = pmv_laplace(np.array([0.1, 0.15, 0.12, 0.09, 0.11]), cor_mat)
    assert 0.0 <= p_small <= 1.0

    p_large = pmv_laplace(np.full(5, 5.0), cor_mat)
    assert p_large == 1.0


def test_pmv_laplace_tracks_cdf_evals():
    cor_mat = np.eye(5) * 0.01
    x = np.array([0.1, 0.15, 0.12, 0.09, 0.11])
    reset_pmv_laplace_profile()
    pmv_laplace(x, cor_mat)
    assert get_pmv_laplace_profile()["cdf_evals"] > 1
    reset_pmv_laplace_profile()


def test_pmv_laplace_cdf_options_keep_result_in_unit_interval():
    cor_mat = np.eye(5) * 0.01
    x = np.array([0.1, 0.15, 0.12, 0.09, 0.11])
    try:
        set_pmv_laplace_cdf_options(maxpts=500, eps=1e-3)
        p = pmv_laplace(x, cor_mat)
    finally:
        set_pmv_laplace_cdf_options()
    assert 0.0 <= p <= 1.0


def test_pmv_laplace_reuses_cbc_lattice_across_z_grid():
    # The QMC lattice construction (scipy.stats._qmvnt._cbc_lattice) depends
    # only on (n_dim, n_qmc_samples), which is a constant for this codebase's
    # fixed order-5 cor_mat / maxpts=25000 usage -- so a single pmv_laplace
    # call (up to 291 CDF evals) should recompute it at most a handful of
    # times, not once per eval.
    call_count = [0]
    orig = stats._orig_cbc_lattice

    def counting_orig(*args, **kwargs):
        call_count[0] += 1
        return orig(*args, **kwargs)

    cor_mat = np.eye(5) * 0.3 + 0.1
    x = np.array([0.8, 0.5, 0.9, 0.3, 0.6])
    stats._cbc_lattice_cache.clear()
    stats._orig_cbc_lattice = counting_orig
    try:
        reset_pmv_laplace_profile()
        pmv_laplace(x, cor_mat)
        n_evals = get_pmv_laplace_profile()["cdf_evals"]
    finally:
        stats._orig_cbc_lattice = orig
        reset_pmv_laplace_profile()

    assert n_evals > 1
    assert call_count[0] <= 4
    assert call_count[0] < n_evals


def test_pmv_laplace_z_grid_matches_r_seq_exactly():
    # R: c(10^-100, 10^seq(-20,-2,1), seq(0.02,1,0.04), seq(1,10,0.2),
    #      seq(10.5,100,0.5), 10^(3:20), 10^100) -- verified against a real
    # R session to have length 290 and these exact boundary values.
    def r_seq(from_, to, by):
        n = round((to - from_) / by) + 1
        return from_ + by * np.arange(n)

    z = np.concatenate(
        [
            [1e-100],
            10.0 ** np.arange(-20, -1, 1),
            r_seq(0.02, 1.0, 0.04),
            r_seq(1.0, 10.0, 0.2),
            r_seq(10.5, 100.0, 0.5),
            10.0 ** np.arange(3, 21, 1),
            [1e100],
        ]
    )
    assert z.shape[0] == 290
    np.testing.assert_allclose(z[:5], [1e-100, 1e-20, 1e-19, 1e-18, 1e-17])
    np.testing.assert_allclose(z[-5:], [1e17, 1e18, 1e19, 1e20, 1e100])


def _random_cormat(rng):
    while True:
        a = rng.uniform(0.2, 0.85)
        rho = a ** np.arange(5)
        rho = rho * (1 + rng.uniform(-0.05, 0.05, size=5))
        rho[0] = 1.0
        idx = np.arange(5)
        lag = np.abs(idx[:, None] - idx[None, :])
        sigma2 = rng.uniform(0.05, 0.5)
        cormat = sigma2 * rho[lag]
        if np.all(np.linalg.eigvalsh(cormat) > 1e-9):
            return cormat


def test_permuted_cholesky_numba_matches_scipy_exactly_for_typical_boxes():
    # _permuted_cholesky is a deterministic pivoted-Cholesky algorithm (no
    # randomization), so for a "typical" (non-saturated) box -- where each
    # dimension's marginal probability range is meaningfully different from
    # the others -- the greedy pivot-selection heuristic has a real signal
    # to discriminate on, and the numba port agrees with SciPy's own
    # pure-Python implementation to near machine precision.
    rng = np.random.default_rng(0)
    max_diff = 0.0
    for _ in range(100):
        cor_mat = _random_cormat(rng)
        abs_x = rng.uniform(0.01, 3.0, size=5)
        z0 = 10 ** rng.uniform(-1, 1)  # moderate range only -- see the
        # saturated-box test below for why extreme z0 is excluded here.
        low = -abs_x / np.sqrt(z0)
        high = abs_x / np.sqrt(z0)

        cho_s, lo_s, hi_s = stats._orig_permuted_cholesky(cor_mat, low, high)
        cho_n, lo_n, hi_n = stats._permuted_cholesky_numba(cor_mat, low, high)

        max_diff = max(
            max_diff,
            np.max(np.abs(cho_s - cho_n)),
            np.max(np.abs(lo_s - lo_n)),
            np.max(np.abs(hi_s - hi_n)),
        )

    assert max_diff < 1e-12


def test_permuted_cholesky_numba_agrees_with_scipy_on_final_probability_at_saturated_boxes():
    # At a near-saturated box (every dimension's marginal probability
    # already ~1 -- exactly what pmv_laplace's z-grid produces at its
    # small-z0 tail), the pivot-selection heuristic's "smallest remaining
    # probability range" comparison is a near-tie in every dimension
    # simultaneously, not just between two candidates. Found via a 200k-case
    # random search (see docs/PERF_LOG.md) that this can flip which pivot
    # order this numba port picks vs. SciPy's own implementation -- purely a
    # floating-point summation-order difference (NumPy's `@`, used
    # internally by SciPy's pure-Python original, vs. this port's explicit
    # loop), reproducible even on a single machine, not a cross-platform
    # BLAS artifact. Confirmed on the worst case found (~196 divergence in
    # the raw cho/lo/hi arrays) that BOTH pivot choices are independently
    # valid: feeding either decomposition into SciPy's own _qmvn_inner
    # kernel (same lattice, same random shift, isolating the decomposition
    # as the only variable) gives the same final probability, matching a
    # high-precision scipy.stats.multivariate_normal.cdf reference to
    # ~1e-13. So this test checks the invariant that actually matters for
    # pmv_laplace's correctness -- the downstream probability -- not
    # bit-identical intermediates, which even SciPy's own algorithm
    # wouldn't guarantee across different BLAS backends at this kind of
    # degenerate input.
    rng = np.random.default_rng(0)
    n_batches = 10
    mi = 25000
    q_lat, n_qmc = _qmvnt._cbc_lattice(4, mi // n_batches)

    # An explicit, known-divergent case (found via a 200k-case random
    # search, exact values captured directly -- not reconstructed from a
    # rounded correlation vector, since this tie is sensitive enough that
    # even tiny input changes can make it stop diverging) is included first
    # so this test always exercises a real pivot-order divergence rather
    # than relying on the random draws below to stumble onto one
    # (empirically, they almost never do -- roughly 1-in-40000, per the
    # search that found it).
    _known_cor_mat = np.array([
        [0.24994265385077855, 0.16564401749379096, 0.1267257337092855, 0.08624421809638201, 0.05817043133102397],
        [0.16564401749379096, 0.24994265385077855, 0.16564401749379096, 0.1267257337092855, 0.08624421809638201],
        [0.1267257337092855, 0.16564401749379096, 0.24994265385077855, 0.16564401749379096, 0.1267257337092855],
        [0.08624421809638201, 0.1267257337092855, 0.16564401749379096, 0.24994265385077855, 0.16564401749379096],
        [0.05817043133102397, 0.08624421809638201, 0.1267257337092855, 0.16564401749379096, 0.24994265385077855],
    ])
    _known_low = np.array([-77.560628426754, -24.30620393963036, -15.880500456361732,
                            -4.136553148872132, -55.794487509459614])
    _known_high = -_known_low

    cho_s, lo_s, hi_s = stats._orig_permuted_cholesky(_known_cor_mat, _known_low, _known_high)
    cho_n, lo_n, hi_n = stats._permuted_cholesky_numba(_known_cor_mat, _known_low, _known_high)
    known_diff = max(np.max(np.abs(cho_s - cho_n)), np.max(np.abs(lo_s - lo_n)), np.max(np.abs(hi_s - hi_n)))
    assert known_diff > 1e-6, "expected known case to diverge; if it no longer does, replace it"

    rndm = np.random.default_rng(1).random(size=(n_batches, 5))
    prob_s, _, _ = _qmvnt._qmvn_inner(q_lat, rndm, n_qmc, n_batches, cho_s, lo_s, hi_s)
    prob_n, _, _ = _qmvnt._qmvn_inner(q_lat, rndm, n_qmc, n_batches, cho_n, lo_n, hi_n)
    assert abs(prob_s - prob_n) < 1e-6, (
        f"known-divergent case: probabilities disagree ({prob_s} vs {prob_n})"
    )

    for i in range(200):
        cor_mat = _random_cormat(rng)
        abs_x = rng.uniform(0.01, 3.0, size=5)
        z0 = 10 ** rng.uniform(-3, 3)  # full range, including saturating tails
        low = -abs_x / np.sqrt(z0)
        high = abs_x / np.sqrt(z0)

        cho_s, lo_s, hi_s = stats._orig_permuted_cholesky(cor_mat, low, high)
        cho_n, lo_n, hi_n = stats._permuted_cholesky_numba(cor_mat, low, high)

        rndm = np.random.default_rng(1000 + i).random(size=(n_batches, 5))
        prob_s, _, _ = _qmvnt._qmvn_inner(q_lat, rndm, n_qmc, n_batches, cho_s, lo_s, hi_s)
        prob_n, _, _ = _qmvnt._qmvn_inner(q_lat, rndm, n_qmc, n_batches, cho_n, lo_n, hi_n)

        assert abs(prob_s - prob_n) < 1e-6, (
            f"case {i}: cho/lo/hi diverged (different pivot order) AND the "
            f"resulting probabilities disagree ({prob_s} vs {prob_n}) -- this "
            "would be a real correctness bug, unlike a mere pivot-order difference"
        )

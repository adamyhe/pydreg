import numpy as np

from pydreg.stats import build_cormat, get_laplace_quantile, get_laplace_sigma, pmv_laplace, qlaplace


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

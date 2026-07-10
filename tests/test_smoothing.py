import numpy as np

from pydreg.smoothing import deriv, fastsmooth, segmented_smooth


def test_deriv_central_difference():
    a = np.array([1.0, 3.0, 6.0, 10.0, 15.0])
    np.testing.assert_allclose(deriv(a), [2.0, 2.5, 3.5, 4.5, 5.0])


def test_fastsmooth_matches_r_docstring_example():
    # From peak_calling_ext.R's own docstring examples.
    Y = np.array([1, 1, 1, 10, 10, 10, 1, 1, 1, 1], dtype=float)
    np.testing.assert_allclose(fastsmooth(Y, 3, 1, 0), [0, 1, 4, 7, 10, 7, 4, 1, 1, 0])
    np.testing.assert_allclose(fastsmooth(Y, 3, 1, 1), [1, 1, 4, 7, 10, 7, 4, 1, 1, 1])


def test_segmented_smooth_scalar_width_matches_fastsmooth():
    rng = np.random.default_rng(0)
    y = rng.normal(size=50)
    np.testing.assert_allclose(segmented_smooth(y, 4, 2), fastsmooth(y, 4, 2))

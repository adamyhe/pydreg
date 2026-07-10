import numpy as np

from pydreg import rfsplit


def test_find_rf_peaks_detects_two_separated_peaks(rf_model):
    x = np.arange(0, 2000, 10)
    y = np.zeros_like(x, dtype=float)
    y += 0.8 * np.exp(-((x - 500) ** 2) / (2 * 80**2))
    y += 0.9 * np.exp(-((x - 1500) ** 2) / (2 * 80**2))
    y += np.random.default_rng(0).normal(scale=0.01, size=len(x))

    cor_mat = np.eye(5) * 0.01
    result = rfsplit.find_rf_peaks(rf_model, x, y, amp_threshold=0.05, smoothwidth=4, cor_mat=cor_mat)

    assert result is not None
    assert len(result) == 2
    np.testing.assert_allclose(result["smooth_mode"], [490.0, 1490.0])
    assert result["score"].iloc[0] > 0.7
    assert result["score"].iloc[1] > 0.8
    assert (result["prob"] < 0.01).all()


def test_find_rf_peaks_returns_none_with_no_signal(rf_model):
    x = np.arange(0, 2000, 10)
    y = np.random.default_rng(0).normal(scale=0.01, size=len(x))
    result = rfsplit.find_rf_peaks(rf_model, x, y, amp_threshold=0.5, smoothwidth=4, cor_mat=np.eye(5))
    assert result is None

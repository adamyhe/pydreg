import numpy as np
import pybigtools
import pytest

from pydreg import features, io


@pytest.fixture
def integer_bigwig_pair(tmp_path):
    """A bigWig pair with integer-valued signal (Poisson counts), matching
    dREG's actual input contract (unnormalized point-mode read counts --
    see CLAUDE.md). Unlike the continuous-Gaussian synthetic_bigwig_pair
    fixture used elsewhere, this is what extract_features_batch's
    cumsum-based binning is bit-identical against: summing exact integers
    in float64 has no rounding error regardless of summation order, but
    summing arbitrary (non-integer) floats can differ in the last bit
    between cumsum-then-subtract and reshape-then-sum -- a real but
    practically irrelevant distinction, since real bigWig inputs to dREG
    are always integer read counts (verified on real chr21 data, see
    docs/PERF_LOG.md)."""
    rng = np.random.default_rng(1)
    chrom_size = 100_000
    plus = rng.poisson(0.02, size=chrom_size).astype(float)
    minus = -rng.poisson(0.02, size=chrom_size).astype(float)
    x = np.arange(chrom_size)
    plus += np.round(6 * np.exp(-((x - 50200) ** 2) / (2 * 150**2)))
    minus -= np.round(5 * np.exp(-((x - 49800) ** 2) / (2 * 150**2)))

    paths = {}
    for strand, vals in (("plus", plus), ("minus", minus)):
        path = str(tmp_path / f"{strand}.bw")
        bw = pybigtools.open(path, "w")
        intervals = []
        i = 0
        while i < chrom_size:
            if vals[i] != 0:
                j = i
                while j < chrom_size and vals[j] == vals[i]:
                    j += 1
                intervals.append(("chr1", i, j, float(vals[i])))
                i = j
            else:
                i += 1
        bw.write({"chr1": chrom_size}, intervals)
        paths[strand] = path

    return paths["plus"], paths["minus"]


def _naive_batch(bw_plus, bw_minus, chrom, centers, window_sizes, half_n_windows):
    """Reference implementation: one extract_features() call per position,
    no shared-fetch batching -- what extract_features_batch used to do."""
    rows = [
        features.extract_features(bw_plus, bw_minus, chrom, int(c), window_sizes, half_n_windows)
        for c in centers
    ]
    return np.stack(rows)


def test_extract_features_batch_matches_naive_per_position(integer_bigwig_pair):
    plus_path, minus_path = integer_bigwig_pair
    bw_plus = io.open_bigwig(plus_path)
    bw_minus = io.open_bigwig(minus_path)

    window_sizes = [10, 25, 50]
    half_n_windows = [10, 10, 10]
    centers = np.array([49800, 49850, 49900, 50000, 50100, 50200, 60000])

    naive = _naive_batch(bw_plus, bw_minus, "chr1", centers, window_sizes, half_n_windows)
    batched = features.extract_features_batch(
        bw_plus, bw_minus, "chr1", centers, window_sizes, half_n_windows
    )
    np.testing.assert_array_equal(naive, batched)


def test_extract_features_batch_handles_unsorted_input(integer_bigwig_pair):
    plus_path, minus_path = integer_bigwig_pair
    bw_plus = io.open_bigwig(plus_path)
    bw_minus = io.open_bigwig(minus_path)

    window_sizes = [10, 25, 50]
    half_n_windows = [10, 10, 10]
    sorted_centers = np.array([49800, 49850, 49900, 50000, 50100, 50200, 60000])
    shuffled = sorted_centers[[3, 0, 6, 1, 5, 2, 4]]

    naive = _naive_batch(bw_plus, bw_minus, "chr1", shuffled, window_sizes, half_n_windows)
    batched = features.extract_features_batch(
        bw_plus, bw_minus, "chr1", shuffled, window_sizes, half_n_windows
    )
    np.testing.assert_array_equal(naive, batched)


def test_extract_features_batch_handles_chromosome_edges(integer_bigwig_pair):
    plus_path, minus_path = integer_bigwig_pair
    bw_plus = io.open_bigwig(plus_path)
    bw_minus = io.open_bigwig(minus_path)

    window_sizes = [10, 25, 50]
    half_n_windows = [10, 10, 10]
    # max_dist = max(10*10, 25*10, 50*10) = 500 -- these positions push the
    # shared/naive fetch window past both chromosome boundaries.
    centers = np.array([0, 10, 200, 99_999, 99_800])

    naive = _naive_batch(bw_plus, bw_minus, "chr1", centers, window_sizes, half_n_windows)
    batched = features.extract_features_batch(
        bw_plus, bw_minus, "chr1", centers, window_sizes, half_n_windows
    )
    np.testing.assert_array_equal(naive, batched)


def test_extract_features_batch_splits_wide_clusters(monkeypatch, integer_bigwig_pair):
    """Forces _MAX_SHARED_FETCH_WIDTH small enough that a handful of widely
    spaced centers must fall into separate clusters, exercising the
    multi-cluster path on a tiny fixture."""
    plus_path, minus_path = integer_bigwig_pair
    bw_plus = io.open_bigwig(plus_path)
    bw_minus = io.open_bigwig(minus_path)

    window_sizes = [10, 25, 50]
    half_n_windows = [10, 10, 10]
    centers = np.array([1000, 2000, 50000, 51000, 90000])

    monkeypatch.setattr(features, "_MAX_SHARED_FETCH_WIDTH", 500)

    naive = _naive_batch(bw_plus, bw_minus, "chr1", centers, window_sizes, half_n_windows)
    batched = features.extract_features_batch(
        bw_plus, bw_minus, "chr1", centers, window_sizes, half_n_windows
    )
    np.testing.assert_array_equal(naive, batched)

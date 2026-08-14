import numpy as np
import pybigtools
import pytest

from pydreg import features, infp


@pytest.fixture
def integer_bigwig_pair(tmp_path):
    """A bigWig pair with integer-valued signal (Poisson counts), matching
    dREG's actual input contract (unnormalized point-mode read counts --
    see CLAUDE.md). Unlike the continuous-Gaussian synthetic_bigwig_pair
    fixture used elsewhere, this is what extract_features_batch's
    cumsum-based *summation* is bit-identical against: summing exact
    integers in float64 has no rounding error regardless of summation
    order, but summing arbitrary (non-integer) floats can differ in the
    last bit between cumsum-then-subtract and reshape-then-sum -- a real
    but practically irrelevant distinction, since real bigWig inputs to
    dREG are always integer read counts (verified on real chr21 data, see
    docs/PERF_LOG.md). That guarantee covers the summation step only, not
    the logistic-scale step downstream of it -- _binned_sums_batch's numba
    kernel can differ from the naive per-position path by ~1 ULP there
    (numba's np.exp() lowering vs NumPy's own), which is why the
    comparisons below use assert_allclose(atol=1e-12), not
    assert_array_equal."""
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
    bw_plus = pybigtools.open(plus_path)
    bw_minus = pybigtools.open(minus_path)

    window_sizes = [10, 25, 50]
    half_n_windows = [10, 10, 10]
    centers = np.array([49800, 49850, 49900, 50000, 50100, 50200, 60000])

    naive = _naive_batch(bw_plus, bw_minus, "chr1", centers, window_sizes, half_n_windows)
    batched = features.extract_features_batch(
        bw_plus, bw_minus, "chr1", centers, window_sizes, half_n_windows
    )
    # atol, not assert_array_equal: the cumsum-difference summation itself
    # is exact for integer input (see integer_bigwig_pair's docstring), but
    # _binned_sums_batch's logistic-scale step now runs through numba's
    # np.exp() lowering (LLVM-compiled), which isn't guaranteed bit-for-bit
    # identical to NumPy's own np.exp() ufunc -- both are correctly rounded
    # to within ~1 ULP, they just don't have to agree on which way a
    # borderline case rounds. Confirmed on real CI: a real ~1-ULP
    # (6.9e-18 absolute, 2.3e-16 relative) mismatch surfaced on the Python
    # 3.11 runner specifically (not reproducible on this dev machine, nor
    # on the 3.12/3.13 runners) -- exactly the signature of two distinct
    # transcendental-function implementations, not a real algorithmic bug.
    # See docs/PERF_LOG.md.
    np.testing.assert_allclose(naive, batched, atol=1e-12)


def test_extract_features_batch_handles_unsorted_input(integer_bigwig_pair):
    plus_path, minus_path = integer_bigwig_pair
    bw_plus = pybigtools.open(plus_path)
    bw_minus = pybigtools.open(minus_path)

    window_sizes = [10, 25, 50]
    half_n_windows = [10, 10, 10]
    sorted_centers = np.array([49800, 49850, 49900, 50000, 50100, 50200, 60000])
    shuffled = sorted_centers[[3, 0, 6, 1, 5, 2, 4]]

    naive = _naive_batch(bw_plus, bw_minus, "chr1", shuffled, window_sizes, half_n_windows)
    batched = features.extract_features_batch(
        bw_plus, bw_minus, "chr1", shuffled, window_sizes, half_n_windows
    )
    # atol, not assert_array_equal: the cumsum-difference summation itself
    # is exact for integer input (see integer_bigwig_pair's docstring), but
    # _binned_sums_batch's logistic-scale step now runs through numba's
    # np.exp() lowering (LLVM-compiled), which isn't guaranteed bit-for-bit
    # identical to NumPy's own np.exp() ufunc -- both are correctly rounded
    # to within ~1 ULP, they just don't have to agree on which way a
    # borderline case rounds. Confirmed on real CI: a real ~1-ULP
    # (6.9e-18 absolute, 2.3e-16 relative) mismatch surfaced on the Python
    # 3.11 runner specifically (not reproducible on this dev machine, nor
    # on the 3.12/3.13 runners) -- exactly the signature of two distinct
    # transcendental-function implementations, not a real algorithmic bug.
    # See docs/PERF_LOG.md.
    np.testing.assert_allclose(naive, batched, atol=1e-12)


def test_extract_features_batch_handles_chromosome_edges(integer_bigwig_pair):
    plus_path, minus_path = integer_bigwig_pair
    bw_plus = pybigtools.open(plus_path)
    bw_minus = pybigtools.open(minus_path)

    window_sizes = [10, 25, 50]
    half_n_windows = [10, 10, 10]
    # max_dist = max(10*10, 25*10, 50*10) = 500 -- these positions push the
    # shared/naive fetch window past both chromosome boundaries.
    centers = np.array([0, 10, 200, 99_999, 99_800])

    naive = _naive_batch(bw_plus, bw_minus, "chr1", centers, window_sizes, half_n_windows)
    batched = features.extract_features_batch(
        bw_plus, bw_minus, "chr1", centers, window_sizes, half_n_windows
    )
    # atol, not assert_array_equal: the cumsum-difference summation itself
    # is exact for integer input (see integer_bigwig_pair's docstring), but
    # _binned_sums_batch's logistic-scale step now runs through numba's
    # np.exp() lowering (LLVM-compiled), which isn't guaranteed bit-for-bit
    # identical to NumPy's own np.exp() ufunc -- both are correctly rounded
    # to within ~1 ULP, they just don't have to agree on which way a
    # borderline case rounds. Confirmed on real CI: a real ~1-ULP
    # (6.9e-18 absolute, 2.3e-16 relative) mismatch surfaced on the Python
    # 3.11 runner specifically (not reproducible on this dev machine, nor
    # on the 3.12/3.13 runners) -- exactly the signature of two distinct
    # transcendental-function implementations, not a real algorithmic bug.
    # See docs/PERF_LOG.md.
    np.testing.assert_allclose(naive, batched, atol=1e-12)


def test_extract_features_batch_splits_wide_clusters(monkeypatch, integer_bigwig_pair):
    """Forces _MAX_SHARED_FETCH_WIDTH small enough that a handful of widely
    spaced centers must fall into separate clusters, exercising the
    multi-cluster path on a tiny fixture."""
    plus_path, minus_path = integer_bigwig_pair
    bw_plus = pybigtools.open(plus_path)
    bw_minus = pybigtools.open(minus_path)

    window_sizes = [10, 25, 50]
    half_n_windows = [10, 10, 10]
    centers = np.array([1000, 2000, 50000, 51000, 90000])

    monkeypatch.setattr(features, "_MAX_SHARED_FETCH_WIDTH", 500)

    naive = _naive_batch(bw_plus, bw_minus, "chr1", centers, window_sizes, half_n_windows)
    batched = features.extract_features_batch(
        bw_plus, bw_minus, "chr1", centers, window_sizes, half_n_windows
    )
    # atol, not assert_array_equal: the cumsum-difference summation itself
    # is exact for integer input (see integer_bigwig_pair's docstring), but
    # _binned_sums_batch's logistic-scale step now runs through numba's
    # np.exp() lowering (LLVM-compiled), which isn't guaranteed bit-for-bit
    # identical to NumPy's own np.exp() ufunc -- both are correctly rounded
    # to within ~1 ULP, they just don't have to agree on which way a
    # borderline case rounds. Confirmed on real CI: a real ~1-ULP
    # (6.9e-18 absolute, 2.3e-16 relative) mismatch surfaced on the Python
    # 3.11 runner specifically (not reproducible on this dev machine, nor
    # on the 3.12/3.13 runners) -- exactly the signature of two distinct
    # transcendental-function implementations, not a real algorithmic bug.
    # See docs/PERF_LOG.md.
    np.testing.assert_allclose(naive, batched, atol=1e-12)


def test_build_clusters_splits_on_density_not_just_absolute_span():
    # max_dist=500 => 2*max_dist+1=1001 is the density threshold. Two dense
    # groups (50bp apart internally) separated by a ~40,000bp gap -- far
    # more than 1001 apart, but nowhere near _MAX_SHARED_FETCH_WIDTH
    # (5,000,000) -- must now split into 2 clusters instead of 1, since
    # merging them would fetch ~40,000bp that no query point needs.
    max_dist = 500
    sorted_centers = np.array([10_000, 10_050, 10_100, 50_000, 50_050])
    clusters = features._build_clusters(sorted_centers, max_dist)
    assert clusters == [(0, 3), (3, 5)]


def test_build_clusters_still_merges_within_density_threshold():
    max_dist = 500
    # consecutive gaps here (50, 1000) are both <= 2*500+1=1001 -- one
    # cluster, same as the old span-only rule would have given.
    sorted_centers = np.array([10_000, 10_050, 11_050])
    clusters = features._build_clusters(sorted_centers, max_dist)
    assert clusters == [(0, 3)]


def test_build_clusters_respects_max_shared_fetch_width_backstop(monkeypatch):
    # A long chain of points each exactly at the density threshold apart
    # (so the density rule alone would keep merging them forever) must
    # still split once accumulated span exceeds _MAX_SHARED_FETCH_WIDTH.
    max_dist = 500
    step = 2 * max_dist + 1
    monkeypatch.setattr(features, "_MAX_SHARED_FETCH_WIDTH", step * 3)
    sorted_centers = np.arange(0, step * 10, step)
    clusters = features._build_clusters(sorted_centers, max_dist)
    assert len(clusters) > 1
    for start_i, end_i in clusters:
        span = sorted_centers[end_i - 1] - sorted_centers[start_i]
        assert span <= step * 3


def test_cap_workers_for_memory_reduces_for_several_large_clusters(monkeypatch):
    monkeypatch.setattr(features, "_CLUSTER_BYTES_PER_BP", 1)
    monkeypatch.setattr(features, "_MAX_CONCURRENT_CLUSTER_BYTES", 2_000)
    max_dist = 0
    # 4 clusters, each spanning 1000bp (1000 bytes/cluster at 1 byte/bp) --
    # budget only fits 2 of them concurrently.
    sorted_centers = np.array([0, 999, 2000, 2999, 4000, 4999, 6000, 6999])
    clusters = [(0, 2), (2, 4), (4, 6), (6, 8)]
    assert features._cap_workers_for_memory(clusters, sorted_centers, max_dist, 4) == 2


def test_cap_workers_for_memory_does_not_over_restrict_for_one_huge_cluster(monkeypatch):
    # One huge cluster among many tiny ones should only cost as much as
    # that one cluster needs, not restrict every worker to fit the largest.
    monkeypatch.setattr(features, "_CLUSTER_BYTES_PER_BP", 1)
    monkeypatch.setattr(features, "_MAX_CONCURRENT_CLUSTER_BYTES", 1_100)
    max_dist = 0
    sorted_centers = np.array([0, 999, 2000, 2001, 3000, 3001, 4000, 4001])
    # cluster 0: span 1000 (huge); clusters 1-3: span 2 each (tiny)
    clusters = [(0, 2), (2, 4), (4, 6), (6, 8)]
    # 4 workers: worst case is the 1000-span cluster plus the 3 largest of
    # the rest (2 each) = 1006 <= 1100 -- still fits, no reduction needed.
    assert features._cap_workers_for_memory(clusters, sorted_centers, max_dist, 4) == 4


def test_cap_workers_for_memory_never_reduces_below_one(monkeypatch):
    monkeypatch.setattr(features, "_CLUSTER_BYTES_PER_BP", 1)
    monkeypatch.setattr(features, "_MAX_CONCURRENT_CLUSTER_BYTES", 1)
    max_dist = 0
    sorted_centers = np.array([0, 999, 2000, 2999])
    clusters = [(0, 2), (2, 4)]
    assert features._cap_workers_for_memory(clusters, sorted_centers, max_dist, 2) == 1


def test_cap_workers_for_memory_is_a_noop_for_a_single_requested_worker():
    # requested_workers <= 1 short-circuits -- no spans computed, nothing
    # to reduce.
    clusters = [(0, 2)]
    sorted_centers = np.array([0, 999])
    assert features._cap_workers_for_memory(clusters, sorted_centers, 0, 1) == 1


def test_extract_features_batch_with_extra_readers_matches_single_threaded(
    integer_bigwig_pair,
):
    plus_path, minus_path = integer_bigwig_pair
    window_sizes = [10, 25, 50]
    half_n_windows = [10, 10, 10]
    # max_dist=500 (density threshold 1001) -- 4 groups spaced 20,000bp
    # apart form 4 independent clusters, more than the 2 reader pairs
    # below, exercising the round-robin multi-cluster-per-worker path.
    # Kept within integer_bigwig_pair's 100,000bp chromosome.
    centers = np.concatenate(
        [base + np.arange(3) * 50 for base in [5_000, 25_000, 45_000, 65_000]]
    )

    bw_plus = pybigtools.open(plus_path)
    bw_minus = pybigtools.open(minus_path)
    single_threaded = features.extract_features_batch(
        bw_plus, bw_minus, "chr1", centers, window_sizes, half_n_windows
    )

    bw_plus2 = pybigtools.open(plus_path)
    bw_minus2 = pybigtools.open(minus_path)
    extra_readers = [(pybigtools.open(plus_path), pybigtools.open(minus_path))]
    parallel = features.extract_features_batch(
        bw_plus2,
        bw_minus2,
        "chr1",
        centers,
        window_sizes,
        half_n_windows,
        extra_readers=extra_readers,
    )
    np.testing.assert_array_equal(single_threaded, parallel)


def test_extract_features_batch_extra_readers_exceeding_cluster_count(
    integer_bigwig_pair,
):
    # More reader pairs than clusters must not error -- n_workers is
    # capped at len(clusters), excess reader pairs simply go unused.
    plus_path, minus_path = integer_bigwig_pair
    window_sizes = [10, 25, 50]
    half_n_windows = [10, 10, 10]
    centers = np.array([49800, 49850, 49900])  # one dense cluster

    bw_plus = pybigtools.open(plus_path)
    bw_minus = pybigtools.open(minus_path)
    reference = features.extract_features_batch(
        bw_plus, bw_minus, "chr1", centers, window_sizes, half_n_windows
    )

    bw_plus2 = pybigtools.open(plus_path)
    bw_minus2 = pybigtools.open(minus_path)
    extra_readers = [
        (pybigtools.open(plus_path), pybigtools.open(minus_path)) for _ in range(4)
    ]
    result = features.extract_features_batch(
        bw_plus2,
        bw_minus2,
        "chr1",
        centers,
        window_sizes,
        half_n_windows,
        extra_readers=extra_readers,
    )
    np.testing.assert_array_equal(reference, result)


def test_extract_features_batch_memory_cap_still_matches_uncapped_result(
    monkeypatch, integer_bigwig_pair
):
    # Force the memory cap to bind (4 clusters, 4 reader pairs available,
    # but a budget that only fits 1 cluster's worth of buffers) and confirm
    # the result is still identical -- the cap changes worker count, never
    # what gets computed.
    monkeypatch.setattr(features, "_MAX_CONCURRENT_CLUSTER_BYTES", 1)
    plus_path, minus_path = integer_bigwig_pair
    window_sizes = [10, 25, 50]
    half_n_windows = [10, 10, 10]
    centers = np.concatenate(
        [base + np.arange(3) * 50 for base in [5_000, 25_000, 45_000, 65_000]]
    )

    bw_plus = pybigtools.open(plus_path)
    bw_minus = pybigtools.open(minus_path)
    reference = features.extract_features_batch(
        bw_plus, bw_minus, "chr1", centers, window_sizes, half_n_windows
    )

    bw_plus2 = pybigtools.open(plus_path)
    bw_minus2 = pybigtools.open(minus_path)
    extra_readers = [
        (pybigtools.open(plus_path), pybigtools.open(minus_path)) for _ in range(3)
    ]
    result = features.extract_features_batch(
        bw_plus2,
        bw_minus2,
        "chr1",
        centers,
        window_sizes,
        half_n_windows,
        extra_readers=extra_readers,
    )
    np.testing.assert_array_equal(reference, result)


def test_extract_features_handles_contig_present_only_in_plus_bigwig(tmp_path):
    plus_path = str(tmp_path / "plus.bw")
    minus_path = str(tmp_path / "minus.bw")

    bw = pybigtools.open(plus_path, "w")
    bw.write(
        {"chrUn_gl000233": 5000},
        [("chrUn_gl000233", 100, 300, 5.0)],
    )
    bw = pybigtools.open(minus_path, "w")
    bw.write({"chr1": 5000}, [("chr1", 10, 11, -1.0)])

    bw_plus = pybigtools.open(plus_path)
    bw_minus = pybigtools.open(minus_path)
    positions = infp.get_informative_positions(bw_plus, bw_minus)

    assert set(positions["chrom"]) == {"chrUn_gl000233"}

    X = features.extract_features_batch(
        bw_plus,
        bw_minus,
        "chrUn_gl000233",
        positions["start"].to_numpy(),
        [10],
        [1],
    )

    assert X.shape == (len(positions), 4)
    np.testing.assert_allclose(X[:, 2:], 0.01)

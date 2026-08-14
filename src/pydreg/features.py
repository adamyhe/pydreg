"""Multi-scale genomic feature extraction, ported from read_genomic_data.R +
src/read_genomic_data.c. Produces the raw (Stage A) feature vector that
pydreg.models.DREGModel.predict() expects as its X_raw input -- DREGModel
itself applies a second, independent z-score normalization (Stage B) on top
of this, then RBF kernel eval, then y un-scaling; nothing about that changes
here.

Layout, per position: for each zoom i (window_sizes[i]=W, half_n_windows[i]=H)
there are 2H non-overlapping, contiguous, W-bp-wide bins with NO separate
center bin -- H bins to the left of center (farthest to nearest) and H bins
to the right (nearest to farthest). All zooms' forward-strand bins come
first (concatenated in zoom order), then all zooms' reverse-strand bins --
NOT interleaved per zoom. Each zoom/strand's 2H bins are then independently
logistic-scaled (see `_logistic_scale`) before being placed in the vector.

This module depends only on pydreg.io's read helpers, never on
pydreg.models -- the zoom configuration (window_sizes/half_n_windows) is a
property of a specific trained model (DREGModel exposes it after loading),
passed in explicitly here rather than imported.
"""

import logging
from concurrent.futures import ThreadPoolExecutor

import numba
import numpy as np

from . import io

logger = logging.getLogger(__name__)

VAL_AT_MIN = 0.01
_ALPHA_LN99 = np.log(1 / VAL_AT_MIN - 1)  # ln(99)


def max_dist_from_center(window_sizes, half_n_windows):
    return int(np.max(np.asarray(window_sizes) * np.asarray(half_n_windows)))


def _logistic_scale(bins):
    """Per-zoom, per-strand logistic scaling (scale_genomic_data_strand_sep
    in the C source). Operates on already-non-negative bin sums -- both
    strands' raw signal is abs()'d at fetch time (see extract_features/
    _extract_features_cluster), matching the C reference's
    `bigwig_readi(..., abs=1, ...)` read call. See docs/PLANNING.md for the
    full sourced trace."""
    true_max = np.max(bins)
    scale_max = 1.0 if true_max == 0 else 0.05 * true_max
    alpha = _ALPHA_LN99 / scale_max
    return 1.0 / (1.0 + np.exp(-alpha * (bins - scale_max)))


def extract_features(bw_plus, bw_minus, chrom, center, window_sizes, half_n_windows):
    """Extracts the 2*sum(2*half_n_windows) length feature vector (e.g. 360
    for the pretrained dREG model) for a single genomic position `center`.

    Known perf note carried over from the R/C original: this does one raw
    fetch per position (width 2*max_dist+1, e.g. 200,001 bp for the
    pretrained model's max_dist=100,000). The C implementation batches
    nearby centers into shared larger bigWig queries (merge_adjacent_range)
    to cut down I/O; this Python port does not do that optimization yet
    (per docs/PLANNING.md, v1 is single-process/unoptimized batching) --
    worth revisiting if per-position fetch overhead dominates on real runs."""
    window_sizes = np.asarray(window_sizes, dtype=int)
    half_n_windows = np.asarray(half_n_windows, dtype=int)
    max_dist = max_dist_from_center(window_sizes, half_n_windows)

    def strand_vector(raw):
        left_full = raw[:max_dist]  # positions center-max_dist .. center-1, far->near
        right_full = raw[
            max_dist + 1 :
        ]  # positions center+1 .. center+max_dist, near->far

        blocks = []
        for W, H in zip(window_sizes, half_n_windows):
            span = W * H
            left_bins = left_full[-span:].reshape(H, W).sum(axis=1)
            right_bins = right_full[:span].reshape(H, W).sum(axis=1)
            zoom_bins = np.concatenate([left_bins, right_bins])
            blocks.append(_logistic_scale(zoom_bins))
        return np.concatenate(blocks)

    # abs() matches the C reference's bigwig_readi(..., abs=1, ...) read call
    # (read_genomic_data.c:414-415) -- both strands are absolute-valued per
    # base pair at read time, before any binning. Must be applied here, to
    # the raw per-bp buffer, not to the summed bins: sum(abs(x)) != abs(sum(x)).
    raw_fwd = np.abs(
        io.fetch_raw(bw_plus, chrom, center - max_dist, center + max_dist + 1)
    )
    raw_rev = np.abs(
        io.fetch_raw(bw_minus, chrom, center - max_dist, center + max_dist + 1)
    )
    return np.concatenate([strand_vector(raw_fwd), strand_vector(raw_rev)])


@numba.njit(cache=True, parallel=True)
def _binned_sums_batch_numba(csum, offsets, window_sizes, half_n_windows, n_features):
    """Fused per-position/per-zoom core of _binned_sums_batch: for each
    center, computes every zoom's W-bp bin sums directly from `csum` via
    scalar indexing (no (n, H+1)-shaped gather-index arrays, no separate
    gather-then-subtract-then-logistic-scale passes -- see
    _binned_sums_batch's docstring for why cumsum differences are exact
    here in the first place). Confirmed bit-identical (max_abs_diff=0.0)
    against the prior NumPy fancy-indexing implementation across randomized
    inputs at realistic problem sizes -- see docs/PERF_LOG.md. 2-4x faster
    in isolation before parallelization, and -- since this is the majority
    (60-80%+) of per-cluster feature-extraction time at realistic
    informative-position densities, not just of this one function -- a real
    win specifically for the `cupy` backend, where predict is fast enough
    that extraction is already the dominant, unhidden cost for at least the
    gap-filled-positions step on real hardware (TITAN Xp *and* A100, not
    just top-end cards -- see docs/OPTIMIZATION.md's "Real measurements"
    table).

    parallel=True + prange over positions: each position's output row is
    computed independently of every other (no cross-position reduction, no
    shared mutable state besides each row's own slice of `out`), so this is
    embarrassingly parallel and order-independent -- confirmed bit-identical
    (max_abs_diff=0.0) against the sequential version across randomized
    inputs. A further ~3-4x on top of the sequential jit on this session's
    10-core Apple M4; see docs/PERF_LOG.md."""
    n = offsets.shape[0]
    n_zooms = window_sizes.shape[0]
    out = np.empty((n, n_features), dtype=np.float64)
    for i in numba.prange(n):
        off = offsets[i]
        col = 0
        for z in range(n_zooms):
            W = window_sizes[z]
            H = half_n_windows[z]
            span = W * H
            zoom_bins = np.empty(2 * H, dtype=np.float64)
            true_max = 0.0

            base_left = off - span
            for h in range(H):
                v = csum[base_left + W * (h + 1)] - csum[base_left + W * h]
                zoom_bins[h] = v
                true_max = max(true_max, v)

            base_right = off + 1
            for h in range(H):
                v = csum[base_right + W * (h + 1)] - csum[base_right + W * h]
                zoom_bins[H + h] = v
                true_max = max(true_max, v)

            scale_max = 1.0 if true_max == 0.0 else 0.05 * true_max
            alpha = _ALPHA_LN99 / scale_max
            for h in range(2 * H):
                out[i, col + h] = 1.0 / (
                    1.0 + np.exp(-alpha * (zoom_bins[h] - scale_max))
                )
            col += 2 * H
    return out


def _binned_sums_batch(csum, offsets, window_sizes, half_n_windows):
    """csum: length-(buf_len+1) cumulative sum (csum[0]=0) of one strand's
    shared raw buffer for a cluster of centers. offsets: (n,) each center's
    index within that buffer. Returns (n, sum(2*half_n_windows)) zoom-binned
    + logistic-scaled features for this strand, in zoom order.

    Computes each W-bp bin's sum as a cumsum difference instead of a
    per-position reshape+sum -- O(n*H) work per zoom regardless of window
    width W, instead of O(n*W*H), which is what makes this tractable for
    wide zooms (e.g. W=5000, H=20 => a 100,000-sample window per position)
    without materializing an (n_centers, W*H) gather array. Exact for
    dREG's actual input domain: cumsum-then-subtract and reshape-then-sum
    are bit-identical when summing exact integers in float64 (no rounding
    error regardless of summation order) -- true of real bigWig inputs to
    dREG, which are always unnormalized point-mode read counts (see
    CLAUDE.md), and verified bit-for-bit against the per-position
    reshape+sum path on real chr21 data (see docs/PERF_LOG.md). Not
    bit-identical (though still numerically equivalent to float precision)
    for arbitrary non-integer input, since cumsum's summation order differs
    from reshape+sum's -- not a real-world concern given dREG's input
    contract, but worth noting if this is ever fed non-count data.

    The actual cumsum-difference + logistic-scale arithmetic is
    _binned_sums_batch_numba, a numba-jitted fused kernel -- see its
    docstring for why (a real, measured win, not a speculative one)."""
    n_features = int(2 * np.sum(half_n_windows))
    window_sizes = np.asarray(window_sizes, dtype=np.int64)
    half_n_windows = np.asarray(half_n_windows, dtype=np.int64)
    offsets = np.asarray(offsets, dtype=np.int64)
    return _binned_sums_batch_numba(
        csum, offsets, window_sizes, half_n_windows, n_features
    )


# Safety cap on one shared raw-fetch buffer's genomic span, in case a run of
# points each exactly 2*max_dist+1 apart (the density threshold below) ever
# produces a pathologically long chain -- not the primary clustering driver
# (see extract_features_batch's density-aware splitting), just a backstop
# against unbounded memory for that edge case.
_MAX_SHARED_FETCH_WIDTH = 5_000_000

# Empirical per-bp cost of one cluster's transient working set inside
# _extract_features_cluster: two raw per-bp fetch buffers (forward/reverse
# strand) plus their two cumulative-sum buffers, all float64 (~32 bytes/bp),
# rounded up for np.abs()'s own transient copy and general slack. Used only
# to bound *concurrency* (how many clusters' buffers can be alive at once),
# not to predict exact memory use.
_CLUSTER_BYTES_PER_BP = 48

# Total budget for concurrently in-flight cluster buffers across every
# extraction worker thread combined, independent of --cores. Measured
# directly (not assumed) that letting --cores alone decide worker count
# multiplies extraction's peak memory by roughly the core count for sparse
# position sets (density-aware clustering's own target case): 16-way
# clustered extraction on a synthetic sparse dataset used ~2.7x the memory
# of single-threaded for identical data, and real production runs showed a
# proportional ~1.2-1.3x whole-pipeline RSS increase after this session's
# parallel-extraction work -- see docs/PERF_LOG.md. A fixed (not
# --cores-scaled) budget keeps that ceiling roughly constant regardless of
# how many cores a run requests; deliberately not exposed as a CLI flag
# yet -- revisit if a real workload needs tuning it.
_MAX_CONCURRENT_CLUSTER_BYTES = 512 * 1024 * 1024


def _extract_features_cluster(
    bw_plus, bw_minus, chrom, cluster_centers, max_dist, window_sizes, half_n_windows
):
    lo = int(cluster_centers[0]) - max_dist
    hi = int(cluster_centers[-1]) + max_dist + 1
    offsets = (cluster_centers - lo).astype(np.int64)

    # abs() before cumsum, matching the C reference's bigwig_readi(...,
    # abs=1, ...) read call -- see extract_features's comment above.
    raw_fwd = np.abs(io.fetch_raw(bw_plus, chrom, lo, hi))
    raw_rev = np.abs(io.fetch_raw(bw_minus, chrom, lo, hi))
    csum_fwd = np.concatenate([[0.0], np.cumsum(raw_fwd)])
    csum_rev = np.concatenate([[0.0], np.cumsum(raw_rev)])

    fwd = _binned_sums_batch(csum_fwd, offsets, window_sizes, half_n_windows)
    rev = _binned_sums_batch(csum_rev, offsets, window_sizes, half_n_windows)
    return np.concatenate([fwd, rev], axis=1)


def _build_clusters(sorted_centers, max_dist):
    """Groups sorted_centers into shared-fetch clusters: extends the
    current cluster to include the next point only when doing so adds no
    wasted fetch span -- i.e. only when that point's own
    [center-max_dist, center+max_dist+1) window would already overlap (or
    touch) the previous point's, which happens exactly when consecutive
    centers are within 2*max_dist+1 of each other. Every byte a resulting
    cluster's shared buffer covers is therefore needed by at least one
    point's own window; nothing is fetched "just because it was on the way"
    to a distant next point. _MAX_SHARED_FETCH_WIDTH remains a backstop cap
    on total cluster span, not the primary splitting rule.

    This matters most for sparse position sets (e.g. peaks.find_gap_infp's
    gap-filled points, which exist specifically to fill sparse gaps between
    dense informative regions): previously, capping only on
    _MAX_SHARED_FETCH_WIDTH (5,000,000bp) meant a handful of genuinely
    isolated points scattered across a chromosome would still get merged
    into a few multi-megabase clusters, most of whose span no query point
    actually needed -- turning a handful of tiny per-position fetches into
    a few enormous ones. Dense position sets (adjacent points 10-50bp
    apart, e.g. the bulk informative-position scan) are unaffected: their
    consecutive gaps are already far under 2*max_dist+1, so they cluster
    exactly as before. See docs/PERF_LOG.md for the real-hardware numbers
    that motivated this.

    Returns a list of (start_i, end_i) index ranges into sorted_centers."""
    n = sorted_centers.shape[0]
    clusters = []
    start_i = 0
    while start_i < n:
        end_i = start_i + 1
        while (
            end_i < n
            and (sorted_centers[end_i] - sorted_centers[end_i - 1])
            <= 2 * max_dist + 1
            and (sorted_centers[end_i] - sorted_centers[start_i])
            <= _MAX_SHARED_FETCH_WIDTH
        ):
            end_i += 1
        clusters.append((start_i, end_i))
        start_i = end_i
    return clusters


def _cap_workers_for_memory(clusters, sorted_centers, max_dist, requested_workers):
    """Reduces `requested_workers` (already min(len(reader_pairs),
    len(clusters))) so that the worst-case simultaneous memory footprint of
    that many concurrently in-flight cluster buffers stays under
    _MAX_CONCURRENT_CLUSTER_BYTES. Uses the `requested_workers` LARGEST
    clusters' combined span for that worst case, not just the single
    largest one or an average -- clusters are assigned round-robin (see
    extract_features_batch), so nothing guarantees the biggest clusters
    land one-per-worker rather than piling onto a few; assuming they could
    all end up running concurrently is the safe bound. This means a
    handful of huge clusters among many tiny ones correctly reduces worker
    count by only as much as those few clusters actually require, rather
    than penalizing every worker for the single largest cluster in the
    whole batch.

    Never reduces below 1 -- this bounds concurrency, not whether the work
    happens at all."""
    if requested_workers <= 1:
        return requested_workers
    spans = sorted(
        (
            int(sorted_centers[end_i - 1]) - int(sorted_centers[start_i]) + 2 * max_dist + 1
            for start_i, end_i in clusters
        ),
        reverse=True,
    )
    for n in range(requested_workers, 1, -1):
        if sum(spans[:n]) * _CLUSTER_BYTES_PER_BP <= _MAX_CONCURRENT_CLUSTER_BYTES:
            return n
    return 1


def extract_features_batch(
    bw_plus, bw_minus, chrom, centers, window_sizes, half_n_windows, extra_readers=None
):
    """Same as extract_features(), for an array of centers on one
    chromosome. Returns (n_centers, n_features).

    Unlike calling extract_features() once per center, this fetches one
    shared raw buffer per strand for a whole cluster of nearby centers
    (see _build_clusters) instead of re-fetching an overlapping
    ~2*max_dist-wide window per position -- adjacent informative positions
    are frequently 10-50bp apart while max_dist can be ~100,000bp, so this
    is the batching this module's docstring flagged as still missing
    relative to the C original's merge_adjacent_range. Input order need not
    be sorted; this sorts internally and restores the original order
    before returning.

    extra_readers: optional list of additional (bw_plus, bw_minus) reader
    pairs, each independently opened from the same underlying bigWig files
    (pybigtools readers aren't safe for concurrent use from multiple
    threads -- see docs/PERF_LOG.md -- so a shared pool of *distinct*
    reader objects is required, not just more threads). When given,
    clusters are statically partitioned round-robin across
    1+len(extra_readers) worker threads, each thread owning exactly one
    reader pair for the threads's entire lifetime (never handed to another
    concurrently-running task, so there's no risk of two threads touching
    the same reader at once). Falls back to the original single-threaded
    loop when omitted or when there are fewer clusters than reader pairs
    (no point spinning up threads that would sit idle). Each worker's own
    numba thread count is reduced to numba.get_num_threads() // n_workers
    -- numba.set_num_threads() is thread-local (confirmed directly, not
    assumed), so this only affects that worker's own calls -- to avoid
    n_workers x numba's own thread count oversubscribing real cores, the
    same class of problem peaks.py's BLAS thread-pinning already solves
    for peak-calling workers.

    Worker count is also capped by _cap_workers_for_memory, independent of
    (and potentially lower than) 1+len(extra_readers): letting --cores
    alone decide worker count means every additional reader also means
    another cluster's fetch/cumsum buffers alive at once, which -- for
    sparse position sets with genuinely large clusters -- measurably
    multiplies extraction's peak memory by close to the worker count
    itself, not just its wall time. See docs/PERF_LOG.md for the real
    numbers behind this cap."""
    window_sizes = np.asarray(window_sizes, dtype=int)
    half_n_windows = np.asarray(half_n_windows, dtype=int)
    max_dist = max_dist_from_center(window_sizes, half_n_windows)
    centers = np.asarray(centers, dtype=np.int64)

    order = np.argsort(centers, kind="stable")
    sorted_centers = centers[order]
    n = sorted_centers.shape[0]
    n_features = 2 * int(np.sum(2 * half_n_windows))
    out = np.empty((n, n_features), dtype=np.float64)

    clusters = _build_clusters(sorted_centers, max_dist)
    reader_pairs = [(bw_plus, bw_minus)] + list(extra_readers or [])
    requested_workers = min(len(reader_pairs), len(clusters))
    n_workers = _cap_workers_for_memory(clusters, sorted_centers, max_dist, requested_workers)
    if n_workers < requested_workers:
        max_span = max(
            (int(sorted_centers[end_i - 1]) - int(sorted_centers[start_i]) + 2 * max_dist + 1)
            for start_i, end_i in clusters
        )
        logger.info(
            "extract_features_batch: memory cap reduced workers %d -> %d for %d "
            "clusters (largest span %d bp, ~%.0fMB) to stay under the %.0fMB "
            "concurrent-cluster budget",
            requested_workers,
            n_workers,
            len(clusters),
            max_span,
            max_span * _CLUSTER_BYTES_PER_BP / 1024**2,
            _MAX_CONCURRENT_CLUSTER_BYTES / 1024**2,
        )

    if n_workers <= 1:
        for start_i, end_i in clusters:
            cluster = sorted_centers[start_i:end_i]
            out[order[start_i:end_i]] = _extract_features_cluster(
                bw_plus, bw_minus, chrom, cluster, max_dist, window_sizes, half_n_windows
            )
        return out

    numba_threads_per_worker = max(1, numba.get_num_threads() // n_workers)

    def _run_worker(reader_pair, owned_clusters):
        bwp, bwm = reader_pair
        numba.set_num_threads(numba_threads_per_worker)
        computed = []
        for start_i, end_i in owned_clusters:
            cluster = sorted_centers[start_i:end_i]
            computed.append(
                (
                    start_i,
                    end_i,
                    _extract_features_cluster(
                        bwp, bwm, chrom, cluster, max_dist, window_sizes, half_n_windows
                    ),
                )
            )
        return computed

    owned = [[] for _ in range(n_workers)]
    for i, cluster_range in enumerate(clusters):
        owned[i % n_workers].append(cluster_range)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [
            pool.submit(_run_worker, reader_pairs[w], owned[w]) for w in range(n_workers)
        ]
        for future in futures:
            for start_i, end_i, result in future.result():
                out[order[start_i:end_i]] = result

    return out

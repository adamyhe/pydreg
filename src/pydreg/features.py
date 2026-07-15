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

import numpy as np

from . import io

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
        right_full = raw[max_dist + 1 :]  # positions center+1 .. center+max_dist, near->far

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
    raw_fwd = np.abs(io.fetch_raw(bw_plus, chrom, center - max_dist, center + max_dist + 1))
    raw_rev = np.abs(io.fetch_raw(bw_minus, chrom, center - max_dist, center + max_dist + 1))
    return np.concatenate([strand_vector(raw_fwd), strand_vector(raw_rev)])


def _logistic_scale_batch(bins):
    """Vectorized _logistic_scale, applied independently per row: bins is
    (n, 2H), and each row is scaled using only that row's own max -- the
    same per-position formula as _logistic_scale, just computed for many
    positions via broadcasting (no cross-row reduction, so this is exactly
    the same elementwise arithmetic as calling _logistic_scale row-by-row,
    not an approximation)."""
    true_max = np.max(bins, axis=1, keepdims=True)
    scale_max = np.where(true_max == 0, 1.0, 0.05 * true_max)
    alpha = _ALPHA_LN99 / scale_max
    return 1.0 / (1.0 + np.exp(-alpha * (bins - scale_max)))


def _binned_sums_batch(csum, offsets, window_sizes, half_n_windows):
    """csum: length-(buf_len+1) cumulative sum (csum[0]=0) of one strand's
    shared raw buffer for a cluster of centers. offsets: (n,) each center's
    index within that buffer. Returns (n, sum(2*half_n_windows)) zoom-binned
    + logistic-scaled features for this strand, in zoom order.

    Computes each W-bp bin's sum as a cumsum difference instead of a
    per-position reshape+sum -- O(n*H) work per zoom regardless of window
    width W, instead of O(n*W*H), which is what makes vectorizing across a
    batch of positions tractable for wide zooms (e.g. W=5000, H=20 => a
    100,000-sample window per position) without materializing an
    (n_centers, W*H) gather array. Exact for dREG's actual input domain:
    cumsum-then-subtract and reshape-then-sum are bit-identical when
    summing exact integers in float64 (no rounding error regardless of
    summation order) -- true of real bigWig inputs to dREG, which are
    always unnormalized point-mode read counts (see CLAUDE.md), and
    verified bit-for-bit against the per-position reshape+sum path on real
    chr21 data (see docs/PERF_LOG.md). Not bit-identical (though still
    numerically equivalent to float precision) for arbitrary non-integer
    input, since cumsum's summation order differs from reshape+sum's --
    not a real-world concern given dREG's input contract, but worth noting
    if this is ever fed non-count data."""
    blocks = []
    for W, H in zip(window_sizes, half_n_windows):
        span = int(W) * int(H)
        w = np.arange(H + 1)
        left_edges = offsets[:, None] - span + W * w[None, :]
        left_bins = csum[left_edges[:, 1:]] - csum[left_edges[:, :-1]]
        right_edges = offsets[:, None] + 1 + W * w[None, :]
        right_bins = csum[right_edges[:, 1:]] - csum[right_edges[:, :-1]]
        zoom_bins = np.concatenate([left_bins, right_bins], axis=1)
        blocks.append(_logistic_scale_batch(zoom_bins))
    return np.concatenate(blocks, axis=1)


# Caps the genomic span of one shared raw-fetch buffer per cluster (not the
# number of positions in it -- see extract_features_batch); keeps a single
# fetch from ballooning to chromosome width if a batch's positions are
# spread out, at the cost of falling back to a second shared fetch for the
# remainder.
_MAX_SHARED_FETCH_WIDTH = 5_000_000


def _extract_features_cluster(bw_plus, bw_minus, chrom, cluster_centers, max_dist, window_sizes, half_n_windows):
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


def extract_features_batch(bw_plus, bw_minus, chrom, centers, window_sizes, half_n_windows):
    """Same as extract_features(), for an array of centers on one
    chromosome. Returns (n_centers, n_features).

    Unlike calling extract_features() once per center, this fetches one
    shared raw buffer per strand for a whole cluster of nearby centers
    (clustered by sorted position, capped at _MAX_SHARED_FETCH_WIDTH bp
    span) instead of re-fetching an overlapping ~2*max_dist-wide window per
    position -- adjacent informative positions are frequently 10-50bp apart
    while max_dist can be ~100,000bp, so this is the batching this module's
    docstring flagged as still missing relative to the C original's
    merge_adjacent_range. Input order need not be sorted; this sorts
    internally and restores the original order before returning."""
    window_sizes = np.asarray(window_sizes, dtype=int)
    half_n_windows = np.asarray(half_n_windows, dtype=int)
    max_dist = max_dist_from_center(window_sizes, half_n_windows)
    centers = np.asarray(centers, dtype=np.int64)

    order = np.argsort(centers, kind="stable")
    sorted_centers = centers[order]
    n = sorted_centers.shape[0]
    n_features = 2 * int(np.sum(2 * half_n_windows))
    out = np.empty((n, n_features), dtype=np.float64)

    start_i = 0
    while start_i < n:
        end_i = start_i + 1
        while end_i < n and (sorted_centers[end_i] - sorted_centers[start_i]) <= _MAX_SHARED_FETCH_WIDTH:
            end_i += 1
        cluster = sorted_centers[start_i:end_i]
        out[order[start_i:end_i]] = _extract_features_cluster(
            bw_plus, bw_minus, chrom, cluster, max_dist, window_sizes, half_n_windows
        )
        start_i = end_i

    return out

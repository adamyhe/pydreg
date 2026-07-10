"""Pure-numpy smoothing/derivative primitives ported from peak_calling_ext.R
(originally T. C. O'Haver's classic boxcar-smoothing routines). Used by
pydreg.rfsplit's find_rf_peaks() to pre-smooth the score profile and its
derivative before peak/valley detection. Kept separate from rfsplit.py for
isolated unit-testability against known input/output vectors.
"""

import numpy as np


def deriv(a):
    """First derivative via 2-point central difference (T. C. O'Haver, 1988)."""
    a = np.asarray(a, dtype=float)
    n = a.shape[0]
    d = np.zeros(n)
    d[0] = a[1] - a[0]
    d[-1] = a[-1] - a[-2]
    d[1:-1] = (a[2:] - a[:-2]) / 2
    return d


def sa(Y, smoothwidth, ends=0):
    """Sliding-average (boxcar) smooth of width `smoothwidth`.

    Assumes smoothwidth >= 2 (halfw = round(w/2) >= 1); R's implementation
    has a degenerate edge case at w=1 (assigning to R's nonexistent index 0,
    silently a no-op there) that isn't specially replicated here since the
    real pipeline only ever calls this with smoothwidth=4."""
    Y = np.asarray(Y, dtype=float)
    L = Y.shape[0]
    w = round(smoothwidth)
    halfw = round(w / 2)

    cs = np.concatenate([[0.0], np.cumsum(Y)])
    moving_sum = cs[w:] - cs[:-w]  # moving_sum[k] = sum(Y[k : k+w]), k = 0..L-w

    s = np.zeros(L)
    start_idx = halfw - 1
    end_idx = start_idx + moving_sum.shape[0]
    s[start_idx:end_idx] = moving_sum
    smooth_y = s / w

    if ends == 1:
        startpoint = (smoothwidth + 1) / 2
        smooth_y[0] = (Y[0] + Y[1]) / 2
        for k in range(2, int(np.floor(startpoint)) + 1):
            smooth_y[k - 1] = np.mean(Y[: 2 * k - 1])
            smooth_y[L - k] = np.mean(Y[L - 2 * k + 1 : L])
        smooth_y[L - 1] = (Y[L - 1] + Y[L - 2]) / 2

    return smooth_y


def fastsmooth(Y, w, type=1, ends=0):
    """type: 1=rectangular (1 boxcar pass), 2=triangular (2 passes),
    3/4=pseudo-Gaussian (3/4 passes of the same width), 5=multiple-width
    (4 passes of different widths: 1.6w, 1.4w, 1.2w, w)."""
    if type == 1:
        return sa(Y, w, ends)
    if type == 2:
        return sa(sa(Y, w, ends), w, ends)
    if type == 3:
        return sa(sa(sa(Y, w, ends), w, ends), w, ends)
    if type == 4:
        return sa(sa(sa(sa(Y, w, ends), w, ends), w, ends), w, ends)
    if type == 5:
        return sa(
            sa(
                sa(sa(Y, round(1.6 * w), ends), round(1.4 * w), ends),
                round(1.2 * w),
                ends,
            ),
            w,
            ends,
        )
    raise ValueError(f"unknown smooth type {type}")


def segmented_smooth(y, smoothwidths, type=1, ends=0):
    """Divides y into len(smoothwidths) equal-length segments and smooths
    each with fastsmooth() at that segment's width. The real pipeline always
    calls this with a scalar smoothwidths (1 segment spanning all of y), for
    which this degenerates to plain fastsmooth(y, smoothwidths, type, ends).

    Faithfully replicates a quirk in the original R: if the final segment's
    end index would overflow len(y) (only possible with >1 segment whose
    lengths don't evenly divide len(y)), that segment's contribution is
    silently dropped (left as 0) rather than clamped -- the R code's `break`
    fires before the assignment for that segment runs."""
    y = np.asarray(y, dtype=float)
    n = y.shape[0]
    widths = np.atleast_1d(smoothwidths)
    num_segments = widths.shape[0]
    seg_length = round(n / num_segments)

    smoothed = np.zeros(n)
    for seg in range(num_segments):
        smooth_segment = fastsmooth(y, widths[seg], type, ends)
        start_idx = seg * seg_length
        end_idx = start_idx + seg_length
        if end_idx > n:
            break
        smoothed[start_idx:end_idx] = smooth_segment[start_idx:end_idx]
    return smoothed

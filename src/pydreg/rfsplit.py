"""Per-broad-peak local-maxima detection and RF-based merge/split decision,
ported from peak_calling_rf.R's find_rf_peaks()/split_peak(). Called once per
broad peak (independently -- see pydreg.peaks) with that peak's own
positions/scores, the genome-wide `cor_mat` from pydreg.stats.build_cormat,
and the DREGPeakSplitForest model.

Two upstream quirks are replicated faithfully here, not "fixed" -- the
pretrained pipeline's expected behavior was produced by this exact code:

- The "prob == -1" sentinel for regions narrower than 5 points: no p-value is
  computed at all, and `original_mode`/`centroid` are literal 0, not derived
  quantities (see the R source's own narrow-region branch).
- split_peak()'s ST==2 (merge) branch reads `rp[i, "vally"]` -- a typo for
  "valley" that doesn't exist as a column. Indexing a data.frame by a
  nonexistent column name returns NULL in R (verified empirically, not
  assumed), and `c(existing, a, b, NULL)` silently drops the NULL argument.
  So merged regions' valley position is silently never added to `center`
  here (only start/stop are) -- not a crash, just a silent omission,
  reproduced as-is below.
"""

import numpy as np
import pandas as pd

from . import smoothing, stats


def find_rf_peaks(model, x, y, amp_threshold, smoothwidth, cor_mat, smoothtype=2):
    """x, y: 1-D arrays of positions/scores for one broad peak (already
    restricted to that peak's span). model: a DREGPeakSplitForest. Returns a
    DataFrame with columns start, stop, score, prob, smooth_mode,
    original_mode, centroid (prob == -1 sentinel for regions <5 points wide
    -- no p-value computed), or None if no local maxima exceed amp_threshold.

    SlopeThreshold from the R signature is dropped: it's only ever used
    there to decide how many times to replicate AmpThreshold/smoothwidth
    (via NROW(SlopeThreshold)), and every real call site passes a scalar,
    making that replication a no-op -- so this only supports (and the real
    pipeline only ever needs) scalar amp_threshold/smoothwidth."""
    x = np.asarray(x, dtype=float)
    y_org = np.asarray(y, dtype=float)
    n = x.shape[0]
    smoothwidth = round(smoothwidth)

    seg_width = smoothwidth if n > smoothwidth else 3
    y = smoothing.segmented_smooth(y_org, seg_width, smoothtype)
    if smoothwidth > 1:
        d = smoothing.segmented_smooth(smoothing.deriv(y), seg_width, smoothtype)
    else:
        d = smoothing.deriv(y)

    m = np.arange(1, n)
    is_peak = (d[m] < 0) & (d[m - 1] > 0) & (y[m] > amp_threshold)
    peak_loci = m[is_peak]
    if peak_loci.shape[0] == 0:
        return None

    rows = []
    K = peak_loci.shape[0]
    if K > 1:
        LI, RI = peak_loci[:-1], peak_loci[1:]
        for li, ri in zip(LI, RI):
            vi = li + int(np.argmin(y[li : ri + 1]))
            rows.append(
                {
                    "dist": x[ri] - x[li], "LI": li, "RI": ri, "VI": vi,
                    "start": x[li], "stop": x[ri], "valley": x[vi],
                    "LS": y[li], "RS": y[ri], "VS": y[vi], "ST": 0,
                }
            )

    first_pl, last_pl = int(peak_loci[0]), int(peak_loci[-1])
    left_row = {
        "dist": x[first_pl] - x[0], "LI": 0, "RI": first_pl, "VI": 0,
        "start": x[0], "stop": x[first_pl], "valley": x[0],
        "LS": y[0], "RS": y[first_pl], "VS": y[0], "ST": -1,
    }
    right_row = {
        "dist": x[-1] - x[last_pl], "LI": last_pl, "RI": n - 1, "VI": n - 1,
        "start": x[last_pl], "stop": x[-1], "valley": x[-1],
        "LS": y[last_pl], "RS": y[-1], "VS": y[-1], "ST": 1,
    }
    rp = [left_row, *rows, right_row]

    rp = _split_peak(model, rp)
    if rp is None or len(rp) == 0:
        return None

    out_rows = []
    for region in rp:
        li, ri = int(region["LI"]), int(region["RI"])
        window = y[li : ri + 1]
        above = np.nonzero(window > amp_threshold)[0]
        if above.shape[0] == 0:
            continue
        i_left = li + int(above.min())
        i_right = li + int(above.max())
        i_peak = i_left + int(np.argmax(y[i_left : i_right + 1]))

        w_left = x[i_peak] - x[i_left]
        w_right = x[i_right] - x[i_peak]
        if w_left > 2 * w_right and w_right > 300:
            w_left = 2 * w_right
        if 2 * w_left < w_right and w_left > 300:
            w_right = 2 * w_left

        if i_right - i_left < 5:
            # y.p, R's re-derived argmax over this same [i_left,i_right]
            # window, is always equal to i_peak (identical window) -- reuse it.
            out_rows.append(
                dict(
                    start=x[i_left], stop=x[i_right],
                    score=float(np.max(y_org[i_left : i_right + 1])),
                    prob=-1.0, smooth_mode=x[i_peak], original_mode=0.0, centroid=0.0,
                )
            )
            continue

        i_sample = _sample_indices(y, i_left, i_right, i_peak)
        pv = np.nan
        if i_sample is not None and not np.any(np.isnan(y[i_sample])):
            pv = stats.pmv_laplace(y[i_sample], cor_mat)

        peak_window = y[i_left : i_right + 1]
        weights = np.arange(1, i_right - i_left + 2, dtype=float)
        y_centroid = np.sum(peak_window * weights) / np.sum(peak_window)
        x_wc = round(y_centroid / (i_right - i_left) * (x[i_right] - x[i_left])) + x[i_left]

        original_mode = x[i_left + int(np.argmax(y_org[i_left : i_right + 1]))]
        out_rows.append(
            dict(
                start=x[i_peak] - w_left + 10, stop=x[i_peak] + w_right - 10,
                score=float(np.max(y_org[i_left : i_right + 1])),
                prob=1 - pv, smooth_mode=x[i_peak], original_mode=original_mode, centroid=x_wc,
            )
        )

    if not out_rows:
        return None
    return pd.DataFrame(out_rows)


def _sample_indices(y, i_left, i_right, i_peak):
    """The 5 representative points used for the pmv_laplace p-value test,
    ported from find_rf_peaks() itself (peak_calling_rf.R:83-116) -- NOT
    from the disabled find_peaks() (peak_calling.R:576-584), which uses a
    different search space (global array bounds, not [i_left,i_right]) and
    a different selection rule (maximize sum, not first-fit).

    i_right - i_left < 9: sort the 5 highest-y points in [i_left,i_right] by
    position. Otherwise: 5 fixed offset patterns anchored at i_peak, spaced
    2bp apart; return the first pattern fully contained in [i_left,i_right]
    (R's `x %in% i.left:i.right`, an integer-range containment test), or
    None if none fit (R leaves the sample as NA in this case; the caller's
    NaN check produces the same "no p-value" outcome either way)."""
    if i_right - i_left < 9:
        window = np.arange(i_left, i_right + 1)
        top5 = np.argsort(-y[window], kind="stable")[:5]
        return np.sort(window[top5])

    patterns = (
        i_peak + np.array([-4, -2, 0, 2, 4]),
        i_peak + np.array([-2, 0, 2, 4, 6]),
        i_peak + np.array([-6, -4, -2, 0, 2]),
        i_peak + np.array([-8, -6, -4, -2, 0]),
        i_peak + np.array([0, 2, 4, 6, 8]),
    )
    for pattern in patterns:
        if np.all(pattern >= i_left) and np.all(pattern <= i_right):
            return pattern
    return None


def _split_peak(model, rp):
    for idx, row in enumerate(rp, start=1):
        row["IDX"] = idx
        _update_split_features(row)

    feature_cols = ("dist", "LD", "RD", "LS", "RS", "maxy", "d1", "d2", "d3", "dr")
    while any(row["ST"] == 0 for row in rp):
        active = [row for row in rp if row["ST"] == 0]
        newdata = np.array([[row[col] for col in feature_cols] for row in active], dtype=float)
        pred = model.predict(newdata)
        for row, pred_i in zip(active, pred):
            row["ST"] = 3 if pred_i > 0.5 else 2

        adjacent_merge = [
            idx for idx in range(len(rp) - 1)
            if rp[idx]["ST"] == 2 and rp[idx + 1]["ST"] == 2
        ]
        if not adjacent_merge:
            break

        idx = adjacent_merge[0]
        idx1 = idx + 1
        row = rp[idx]
        row1 = rp[idx1]
        row1["ST"] = -2
        row["ST"] = 0
        row["stop"] = row1["stop"]
        row["RI"] = row1["RI"]
        row["RS"] = row1["RS"]
        # R's which.min(c(a, b)) picks the first element on ties, i.e. a's
        # position when a <= b -- hence <=, not <, here.
        take_idx1 = row1["VS"] <= row["VS"]
        row["VI"] = row1["VI"] if take_idx1 else row["VI"]
        row["valley"] = row1["valley"] if take_idx1 else row["valley"]
        row["VS"] = min(row1["VS"], row["VS"])
        _update_split_features(row)

        rp = [row for row in rp if row["ST"] != -2]
        for idx, row in enumerate(rp, start=1):
            row["IDX"] = idx

    for row in rp:
        row["IDX"] *= 2

    split_positions = [idx for idx, row in enumerate(rp) if row["ST"] == 3]
    if split_positions:
        split_set = set(split_positions)
        new_rows = []
        for i in split_positions:
            row = rp[i]
            top = row.copy()
            top["ST"] = 1
            top["stop"] = row["valley"]
            top["RS"] = row["VS"]
            top["RI"] = row["VI"]

            bottom = row.copy()
            bottom["ST"] = -1
            bottom["IDX"] = row["IDX"] + 1
            bottom["LI"] = row["VI"]
            bottom["LS"] = row["VS"]
            bottom["start"] = row["valley"]

            new_rows.append(top)
            new_rows.append(bottom)

        rp = [row for i, row in enumerate(rp) if i not in split_set] + new_rows
        rp = sorted(rp, key=lambda row: row["IDX"])

    return _collapse_regions(rp)


def _update_split_features(row):
    row["dist"] = row["stop"] - row["start"]
    row["LD"] = row["valley"] - row["start"]
    row["RD"] = row["stop"] - row["valley"]
    row["maxy"] = max(row["LS"], row["RS"])
    row["d1"] = abs(row["LS"] - row["RS"])
    row["d2"] = min(row["LS"], row["RS"]) - row["VS"]
    row["d3"] = row["VS"]
    row["dr"] = row["d2"] / (row["d1"] + row["VS"])


def _collapse_regions(rp):
    rpeak_rows = []
    LI = LS = start = None
    center = center_s = center_i = []

    for row in rp:
        st = row["ST"]
        if st == -1:
            LI, LS, start = row["LI"], row["LS"], row["start"]
            center = [row["start"], row["stop"]]
            center_s = [row["LS"], row["RS"]]
            center_i = [row["LI"], row["RI"]]
        elif st == 2:
            # Faithful to the R typo `rp[i, "vally"]` (nonexistent column ->
            # NULL -> silently dropped by c()): the valley position is NOT
            # appended here, only start/stop.
            center = center + [row["start"], row["stop"]]
            center_s = center_s + [row["LS"], row["RS"], row["VS"]]
            center_i = center_i + [row["LI"], row["RI"], row["VI"]]
        elif st == 1:
            RI, RS, stop = row["RI"], row["RS"], row["stop"]
            center = center + [row["start"], row["stop"]]
            center_s = center_s + [row["LS"], row["RS"]]
            center_i = center_i + [row["LI"], row["RI"]]

            # `center` can be shorter than center_s/center_i (see the ST==2
            # comment above): if argmax lands past the end of `center`,
            # replicate R's out-of-bounds vector indexing, which returns NA
            # rather than erroring (verified empirically, not assumed).
            best = int(np.argmax(center_s))
            peak_center = center[best] if best < len(center) else np.nan
            rpeak_rows.append(
                dict(
                    LI=LI, RI=RI, PI=center_i[best],
                    start=start, stop=stop, center=peak_center,
                    LS=LS, RS=RS, VS=max(center_s),
                )
            )
            LI = LS = start = None
            center, center_s, center_i = [], [], []

    if not rpeak_rows:
        return None
    return rpeak_rows

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
                dict(
                    dist=x[ri] - x[li], LI=li, RI=ri, VI=vi,
                    start=x[li], stop=x[ri], valley=x[vi],
                    LS=y[li], RS=y[ri], VS=y[vi], ST=0,
                )
            )

    first_pl, last_pl = int(peak_loci[0]), int(peak_loci[-1])
    left_row = dict(
        dist=x[first_pl] - x[0], LI=0, RI=first_pl, VI=0,
        start=x[0], stop=x[first_pl], valley=x[0],
        LS=y[0], RS=y[first_pl], VS=y[0], ST=-1,
    )
    right_row = dict(
        dist=x[-1] - x[last_pl], LI=last_pl, RI=n - 1, VI=n - 1,
        start=x[last_pl], stop=x[-1], valley=x[-1],
        LS=y[last_pl], RS=y[-1], VS=y[-1], ST=1,
    )
    rp = pd.DataFrame([left_row, *rows, right_row])

    rp = _split_peak(model, rp)
    if rp is None or len(rp) == 0:
        return None

    out_rows = []
    for _, region in rp.iterrows():
        li, ri = int(region["LI"]), int(region["RI"])
        window = y[li : ri + 1]
        above = np.nonzero(window > amp_threshold)[0]
        if above.shape[0] == 0:
            continue
        i_left = li + int(above.min())
        i_right = li + int(above.max())
        i_peak = li + int(np.argmax(window))

        w_left = x[i_peak] - x[i_left]
        w_right = x[i_right] - x[i_peak]
        if w_left > 2 * w_right and w_right > 300:
            w_left = 2 * w_right
        if 2 * w_left < w_right and w_left > 300:
            w_right = 2 * w_left

        if i_right - i_left < 5:
            y_p = i_left + int(np.argmax(y[i_left : i_right + 1]))
            out_rows.append(
                dict(
                    start=x[i_left], stop=x[i_right],
                    score=float(np.max(y_org[i_left : i_right + 1])),
                    prob=-1.0, smooth_mode=x[y_p], original_mode=0.0, centroid=0.0,
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
    """The 5 representative points used for the pmv_laplace p-value test."""
    if i_right - i_left < 9:
        window_idx = np.arange(i_left, i_right + 1)
        order = np.argsort(-y[window_idx], kind="stable")
        return np.sort(window_idx[order[:5]])

    patterns = [
        [i_peak - 4, i_peak - 2, i_peak, i_peak + 2, i_peak + 4],
        [i_peak - 2, i_peak, i_peak + 2, i_peak + 4, i_peak + 6],
        [i_peak - 6, i_peak - 4, i_peak - 2, i_peak, i_peak + 2],
        [i_peak - 8, i_peak - 6, i_peak - 4, i_peak - 2, i_peak],
        [i_peak, i_peak + 2, i_peak + 4, i_peak + 6, i_peak + 8],
    ]
    for pattern in patterns:
        if all(i_left <= p <= i_right for p in pattern):
            return np.array(pattern)
    return None


def _split_peak(model, rp):
    rp = rp.reset_index(drop=True)
    rp["IDX"] = np.arange(1, len(rp) + 1)
    rp["LD"] = rp["valley"] - rp["start"]
    rp["RD"] = rp["stop"] - rp["valley"]
    rp["maxy"] = rp[["LS", "RS"]].max(axis=1)
    rp["d1"] = (rp["LS"] - rp["RS"]).abs()
    rp["d2"] = rp[["LS", "RS"]].min(axis=1) - rp["VS"]
    rp["d3"] = rp["VS"]
    rp["dr"] = rp["d2"] / (rp["d1"] + rp["VS"])

    feature_cols = ["dist", "LD", "RD", "LS", "RS", "maxy", "d1", "d2", "d3", "dr"]
    while (rp["ST"] == 0).any():
        newdata = rp.loc[rp["ST"] == 0, feature_cols].to_numpy()
        pred = model.predict(newdata)
        rp.loc[rp["ST"] == 0, "ST"] = np.where(pred > 0.5, 3, 2)

        st = rp["ST"].to_numpy()
        adjacent_merge = np.nonzero((st[:-1] == 2) & (st[1:] == 2))[0]
        if adjacent_merge.shape[0] == 0:
            break

        idx = int(adjacent_merge[0])
        idx1 = idx + 1
        rp.loc[idx1, "ST"] = -2
        rp.loc[idx, "ST"] = 0
        rp.loc[idx, "stop"] = rp.loc[idx1, "stop"]
        rp.loc[idx, "RI"] = rp.loc[idx1, "RI"]
        rp.loc[idx, "RS"] = rp.loc[idx1, "RS"]
        # R's which.min(c(a, b)) picks the first element on ties, i.e. a's
        # position when a <= b -- hence <=, not <, here.
        take_idx1 = rp.loc[idx1, "VS"] <= rp.loc[idx, "VS"]
        rp.loc[idx, "VI"] = rp.loc[idx1, "VI"] if take_idx1 else rp.loc[idx, "VI"]
        rp.loc[idx, "valley"] = rp.loc[idx1, "valley"] if take_idx1 else rp.loc[idx, "valley"]
        rp.loc[idx, "LD"] = rp.loc[idx, "valley"] - rp.loc[idx, "start"]
        rp.loc[idx, "RD"] = rp.loc[idx, "stop"] - rp.loc[idx, "valley"]
        rp.loc[idx, "dist"] = rp.loc[idx, "stop"] - rp.loc[idx, "start"]
        rp.loc[idx, "VS"] = min(rp.loc[idx1, "VS"], rp.loc[idx, "VS"])
        ls, rs = rp.loc[idx, "LS"], rp.loc[idx, "RS"]
        rp.loc[idx, "d1"] = max(ls, rs) - min(ls, rs)
        rp.loc[idx, "d2"] = min(ls, rs) - rp.loc[idx, "VS"]
        rp.loc[idx, "d3"] = rp.loc[idx, "VS"]
        rp.loc[idx, "dr"] = rp.loc[idx, "d2"] / (rp.loc[idx, "d1"] + rp.loc[idx, "d3"])
        rp.loc[idx, "maxy"] = max(ls, rs)

        rp = rp[rp["ST"] != -2].reset_index(drop=True)
        rp["IDX"] = np.arange(1, len(rp) + 1)

    rp["IDX"] = rp["IDX"] * 2
    split_positions = np.nonzero((rp["ST"] == 3).to_numpy())[0]
    if split_positions.shape[0] > 0:
        new_rows = []
        for i in split_positions:
            row = rp.iloc[i]
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

        keep = rp.iloc[[i for i in range(len(rp)) if i not in set(split_positions.tolist())]]
        rp = pd.concat([keep, pd.DataFrame(new_rows)], ignore_index=True)
        rp = rp.sort_values("IDX", kind="stable").reset_index(drop=True)

    return _collapse_regions(rp)


def _collapse_regions(rp):
    rpeak_rows = []
    LI = LS = start = None
    center = center_s = center_i = []

    for _, row in rp.iterrows():
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
    return pd.DataFrame(rpeak_rows)

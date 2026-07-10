"""BED-DataFrame-shaped peak-calling operations, ported from peak_calling.R.
Depends only on pydreg.stats and pydreg.rfsplit -- never on pydreg.io/
features/models/backend directly. The two steps that need to re-score newly
generated candidate positions (get_dense_infp's gap-filling and
densification) take a `score_fn(bed_df) -> np.ndarray` callback instead of
importing a scoring backend themselves; pydreg.pipeline supplies the real
one (wiring together bigwig readers, feature extraction, and the chosen
DREGModel backend).

Several upstream R bugs are replicated faithfully here, not fixed -- the
pretrained pipeline's expected behavior was produced by this exact code
(see docs/PLANNING.md for the full list and reasoning):

- merge_broad_peak: the separation-index sentinel is only ever prepended
  (0), never appended, so the LAST contiguous group of rows per chromosome
  (after the final big gap, or the whole chromosome if no gap >= `join`
  exists at all) is always silently dropped -- never becomes a broad peak.
  Groups of exactly one row are also filtered out.
- find_gap_infp: relies on R's `a:b` colon operator, which counts DOWN and
  never returns empty when a > b (e.g. `1:0` is `[1, 0]`, not `[]`) -- see
  `_r_colon` below.
"""

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from . import rfsplit, stats


def _r_colon(a, b):
    """R's `a:b`: ascending if a<=b, descending if a>b -- never empty."""
    if a <= b:
        return np.arange(a, b + 1)
    return np.arange(a, b - 1, -1)


def merge_broad_peak(pred_bed, threshold, join=500):
    """pred_bed: DataFrame with columns chrom, start, end, score (any
    names, first 4 columns used positionally, matching the R original's
    column-index access)."""
    chrom_col, start_col, end_col, score_col = pred_bed.columns[:4]
    pred_bed = pred_bed[pred_bed[score_col] >= threshold].copy()
    if len(pred_bed) == 0:
        return None

    pred_bed[start_col] = pred_bed[start_col] - 50
    pred_bed[end_col] = pred_bed[end_col] + 50
    pred_bed = pred_bed.sort_values([chrom_col, start_col], kind="stable")

    peak_rows = []
    for chrom, chr_rows in pred_bed.groupby(chrom_col, sort=False):
        starts = chr_rows[start_col].to_numpy()
        ends = chr_rows[end_col].to_numpy()

        gap_after = np.nonzero(starts[1:] - ends[:-1] >= join)[0] + 1  # 1-indexed rows
        separe_idx = np.concatenate([[0], gap_after])

        group_start = separe_idx[:-1] + 1  # 1-indexed, inclusive
        group_end = separe_idx[1:]
        keep = group_start != group_end
        group_start, group_end = group_start[keep], group_end[keep]

        for gs, ge in zip(group_start, group_end):
            peak_rows.append(dict(chrom=chrom, start=starts[gs - 1], end=ends[ge - 1]))

    if not peak_rows:
        return None
    return pd.DataFrame(peak_rows)


def get_broadpeak_summary(infp_bed, threshold=0):
    """min/max/mean/sum/std/count of informative-position scores within
    each merged broad-peak interval -- replaces the one bedmap call in the
    original R. Peaks are non-overlapping and both tables are sorted, so
    this is a per-chromosome searchsorted grouping, not a general interval
    join."""
    peak_bed = merge_broad_peak(infp_bed, threshold)
    if peak_bed is None or len(peak_bed) == 0:
        return None

    chrom_col, start_col, end_col, score_col = infp_bed.columns[:4]
    infp_sorted = infp_bed.sort_values([chrom_col, start_col], kind="stable")

    rows = []
    for chrom, chr_peaks in peak_bed.groupby("chrom", sort=False):
        chr_infp = infp_sorted[infp_sorted[chrom_col] == chrom]
        infp_starts = chr_infp[start_col].to_numpy()
        infp_scores = chr_infp[score_col].to_numpy()

        # Batch both searchsorted calls across all of this chromosome's
        # peaks at once (was 2 calls per peak); the per-peak reduction
        # itself stays a plain slice + scalar numpy call (unchanged
        # numerics) since window sizes are ragged.
        peak_starts = chr_peaks["start"].to_numpy()
        peak_ends = chr_peaks["end"].to_numpy()
        lo_arr = np.searchsorted(infp_starts, peak_starts, side="left")
        hi_arr = np.searchsorted(infp_starts, peak_ends, side="left")

        for start, end, lo, hi in zip(peak_starts, peak_ends, lo_arr, hi_arr):
            scores = infp_scores[lo:hi]
            rows.append(
                dict(
                    chrom=chrom, start=start, end=end,
                    min=scores.min(), max=scores.max(), mean=scores.mean(),
                    sum=scores.sum(),
                    stdev=scores.std(ddof=1) if scores.shape[0] > 1 else 0.0,
                    count=scores.shape[0],
                )
            )
    return pd.DataFrame(rows)


def find_gap_infp(dreg_pred, threshold=0.2):
    """Fills 50bp-spaced candidate points into gaps (>50bp) between
    consecutive informative positions, near whichever side of the gap has a
    promising score, so downstream peak-splitting has a well-sampled
    profile through candidate regions. dreg_pred: DataFrame with columns
    chrom, start, end, score (first 4 columns used positionally)."""
    chrom_col, start_col, end_col, score_col = dreg_pred.columns[:4]
    dreg_pred = dreg_pred.sort_values([chrom_col, start_col], kind="stable")
    existing = set(zip(dreg_pred[chrom_col], dreg_pred[start_col]))

    new_rows = []
    for chrom, chr_rows in dreg_pred.groupby(chrom_col, sort=False):
        starts = chr_rows[start_col].to_numpy()
        scores = chr_rows[score_col].to_numpy()
        if starts.shape[0] < 2:
            continue
        dist = starts[1:] - starts[:-1]

        for j in np.nonzero(dist > 50)[0]:
            gap = dist[j]
            positions = []

            if scores[j] > threshold:
                r_maxpos = (
                    int(np.floor((gap - 50) / 50))
                    if gap < 500
                    else int(np.ceil((scores[j] - threshold) / 0.05))
                )
                positions.append(starts[j] + _r_colon(1, r_maxpos) * 50)

            if scores[j + 1] > threshold:
                r_maxpos = (
                    int(np.floor((gap - 50) / 50))
                    if gap < 500
                    else int(np.ceil((scores[j + 1] - threshold) / 0.05))
                )
                positions.append(starts[j + 1] - _r_colon(r_maxpos, 1) * 50)

            if not positions:
                continue
            for p in np.unique(np.concatenate(positions)):
                new_rows.append(dict(chrom=chrom, start=int(p)))

    if not new_rows:
        return None
    gap_bed = pd.DataFrame(new_rows).drop_duplicates(subset=["chrom", "start"])
    existing_idx = (
        pd.MultiIndex.from_tuples(list(existing), names=["chrom", "start"])
        if existing
        else pd.MultiIndex.from_arrays([[], []], names=["chrom", "start"])
    )
    gap_idx = pd.MultiIndex.from_frame(gap_bed[["chrom", "start"]])
    gap_bed = gap_bed[~gap_idx.isin(existing_idx)]
    if len(gap_bed) == 0:
        return None
    gap_bed["end"] = gap_bed["start"] + 1
    return gap_bed.reset_index(drop=True)


def get_dense_infp(infp_bed, score_fn, threshold=0.05):
    """Orchestrates the full densification pipeline: derives the Laplace
    noise-model significance floor `min_score`, fills and scores gap
    positions, merges into broad peaks, then re-densifies at 10bp
    resolution inside promising broad peaks and merges those in too.

    infp_bed: DataFrame with columns chrom, start, end, score (the
    already-scored informative positions). score_fn(bed_df) -> np.ndarray:
    scores a chrom/start/end DataFrame at its given positions (supplied by
    pydreg.pipeline).

    Returns (dense_infp, peak_broad, min_score); dense_infp has an
    additional `infp` column (1 = original informative position, 0 =
    gap-filled/densified)."""
    chrom_col, start_col, end_col, score_col = infp_bed.columns[:4]

    negative = infp_bed[score_col][infp_bed[score_col] < 0].to_numpy()
    zeros = infp_bed[score_col][infp_bed[score_col] == 0].to_numpy()
    noise = np.concatenate([negative, np.abs(negative), zeros])
    sigma = stats.get_laplace_sigma(noise)
    min_score = stats.get_laplace_quantile(sigma, 0.001)

    gap_bed = find_gap_infp(infp_bed[[chrom_col, start_col, end_col, score_col]], min_score)
    if gap_bed is not None and len(gap_bed) > 0:
        gap_bed = gap_bed[["chrom", "start", "end"]]
        gap_bed.columns = [chrom_col, start_col, end_col]
        gap_bed[score_col] = score_fn(gap_bed)
        gap_bed["infp"] = 0
        infp_flagged = infp_bed.copy()
        infp_flagged["infp"] = 1
        newinfp = pd.concat([gap_bed, infp_flagged], ignore_index=True)
    else:
        newinfp = infp_bed.copy()
        newinfp["infp"] = 1
    newinfp = newinfp.sort_values([chrom_col, start_col], kind="stable").reset_index(drop=True)

    peak_broad = get_broadpeak_summary(
        newinfp[[chrom_col, start_col, end_col, score_col]], threshold=0.05
    )

    dense_infp = _pred_dense_infp(
        peak_broad[peak_broad["max"] >= min_score] if peak_broad is not None else None,
        newinfp, score_fn, chrom_col, start_col, end_col, score_col,
    )
    return dense_infp, peak_broad, min_score


def _pred_dense_infp(dreg_peak, newinfp, score_fn, chrom_col, start_col, end_col, score_col):
    if dreg_peak is None or len(dreg_peak) == 0:
        return newinfp

    new_dense = []
    for chrom, chr_peaks in dreg_peak.groupby("chrom", sort=False):
        existing_starts = newinfp.loc[newinfp[chrom_col] == chrom, start_col].to_numpy()

        # Build every peak's 10bp grid (+ explicit end point) first, then
        # dedup/filter against `existing_starts` once across the whole
        # chromosome instead of a Python set.add() per position.
        all_pos = [
            np.unique(np.concatenate([np.arange(s, e, 10), [e]]))
            for s, e in zip(chr_peaks["start"].to_numpy(), chr_peaks["end"].to_numpy())
        ]
        if not all_pos:
            continue
        candidate = np.unique(np.concatenate(all_pos)).astype(int)
        positions = candidate[~np.isin(candidate, existing_starts)]
        if positions.size > 0:
            new_dense.append(
                pd.DataFrame({chrom_col: chrom, start_col: np.sort(positions)})
            )

    if not new_dense:
        return newinfp

    r_dense = pd.concat(new_dense, ignore_index=True)
    r_dense[end_col] = r_dense[start_col] + 1
    r_dense[score_col] = score_fn(r_dense)
    r_dense = r_dense[r_dense[score_col] > 0.05]
    if len(r_dense) == 0:
        return newinfp

    r_dense["infp"] = 0
    newinfp = pd.concat([newinfp, r_dense], ignore_index=True)
    return newinfp.sort_values([chrom_col, start_col], kind="stable").reset_index(drop=True)


def call_peaks(
    dense_infp, peak_broad, min_score, rf_model,
    smoothwidth=4, pv_adjust="fdr", pv_threshold=0.05, progress=False,
):
    """The find_rf_peaks-calling orchestration from peak_calling.R's
    start_calling(): one genome-wide cor_mat, then an independent call to
    rfsplit.find_rf_peaks() per broad peak whose max score clears
    min_score. R's BLOCKWIDTH block-splitting + snowfall-cluster
    parallelization is pure IPC infrastructure with no algorithmic content
    -- this is a plain per-broad-peak loop (v1, single-process; see
    docs/PLANNING.md).

    dense_infp: DataFrame with columns chrom, start, end, score (+ infp
    flag, unused here). peak_broad: DataFrame from get_broadpeak_summary
    (columns chrom, start, end, min, max, mean, sum, stdev, count).

    progress: show a tqdm progress bar over candidate broad peaks
    (auto-hidden if stdout isn't a terminal).

    Returns (raw_peak, peak_bed): raw_peak has all candidate peaks
    pre-FDR-filter (columns chrom, start, end, score, prob, smooth_mode,
    original_mode, centroid); peak_bed is raw_peak filtered/BH-adjusted to
    significant peaks only (columns chrom, start, end, score, prob,
    center)."""
    chrom_col, start_col, end_col, score_col = dense_infp.columns[:4]
    cor_mat = stats.build_cormat(dense_infp[start_col].to_numpy(), dense_infp[score_col].to_numpy())

    if peak_broad is None or len(peak_broad) == 0:
        return None, None
    candidates = peak_broad[peak_broad["max"] >= min_score]

    dense_sorted = dense_infp.sort_values([chrom_col, start_col], kind="stable")
    raw_rows = []
    pbar = tqdm(
        total=len(candidates), desc="calling peaks", unit="peak",
        disable=None if progress else True,
    )
    for chrom, chr_peaks in candidates.groupby("chrom", sort=False):
        chr_infp = dense_sorted[dense_sorted[chrom_col] == chrom]
        infp_starts = chr_infp[start_col].to_numpy()
        infp_ends = chr_infp[end_col].to_numpy()
        infp_scores = chr_infp[score_col].to_numpy()

        for _, peak in chr_peaks.iterrows():
            lo = np.searchsorted(infp_starts, peak["start"], side="left")
            hi = np.searchsorted(infp_ends, peak["end"], side="right")
            xp, yp = infp_starts[lo:hi], infp_scores[lo:hi]
            if xp.shape[0] <= 3 or yp.max() <= min_score:
                pbar.update(1)
                continue

            result = rfsplit.find_rf_peaks(
                rf_model, xp, yp, amp_threshold=min_score, smoothwidth=smoothwidth, cor_mat=cor_mat
            )
            pbar.update(1)
            if result is None:
                continue
            result = result.copy()
            result.insert(0, chrom_col, chrom)
            raw_rows.append(result)
    pbar.close()

    if not raw_rows:
        return None, None
    raw_peak = pd.concat(raw_rows, ignore_index=True)
    raw_peak.columns = [
        "chr", "start", "end", "score", "prob", "smooth.mode", "original.mode", "centroid",
    ]

    peak_bed = select_sig_peak(raw_peak, pv_adjust, pv_threshold)
    return raw_peak, peak_bed


def select_sig_peak(raw_peak, pv_adjust="fdr", pv_threshold=0.05):
    """BH-FDR (or other statsmodels-supported method) adjustment of raw
    per-peak p-values across ALL candidate peaks genome-wide, dropping the
    prob==-1 sentinel rows (peaks <5 points wide, no p-value computed) both
    before and by construction from the adjustment."""
    from statsmodels.stats.multitest import multipletests

    peak_bed = raw_peak[["chr", "start", "end", "score", "prob", "original.mode"]].copy()
    peak_bed.columns = ["chr", "start", "end", "score", "prob", "center"]
    peak_bed = peak_bed[peak_bed["prob"] != -1]

    # Maps R's p.adjust() method names to statsmodels.multipletests()'s.
    # Only "fdr" (the default, an alias for "BH" in R too) is ever actually
    # used by the real pipeline; the rest are here for completeness.
    method = {
        "fdr": "fdr_bh", "BH": "fdr_bh", "BY": "fdr_by", "bonferroni": "bonferroni",
        "holm": "holm", "hochberg": "simes-hochberg", "hommel": "hommel", "none": None,
    }.get(pv_adjust, pv_adjust)
    if method is not None:
        peak_bed["prob"] = multipletests(peak_bed["prob"].to_numpy(), method=method)[1]

    return peak_bed[peak_bed["prob"] <= pv_threshold].reset_index(drop=True)

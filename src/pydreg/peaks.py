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

import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from . import rfsplit, stats


PEAK_CALLING_BLOCK_WIDTH = 500
_WORKER_STATE = {}
logger = logging.getLogger(__name__)


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
    score_values = infp_bed[score_col].to_numpy()
    finite_scores = score_values[np.isfinite(score_values)]
    if finite_scores.size:
        logger.info(
            "informative score distribution: min=%.4f q01=%.4f median=%.4f q99=%.4f max=%.4f; "
            "negative=%d zero=%d noise=%d sigma=%s",
            np.min(finite_scores), np.quantile(finite_scores, 0.01),
            np.median(finite_scores), np.quantile(finite_scores, 0.99),
            np.max(finite_scores), len(negative), len(zeros), len(noise),
            f"{sigma:.6g}" if np.isfinite(sigma) else "nan",
        )
    else:
        logger.warning("informative scores contain no finite values")

    if not np.isfinite(min_score):
        raise ValueError(
            "could not estimate finite min_score from negative/zero score tail "
            f"(negative={len(negative)}, zero={len(zeros)}); this usually means "
            "the scoring backend or input/model convention is wrong"
        )

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


def _init_peak_worker(
    rf_model,
    min_score,
    smoothwidth,
    cor_mat,
    pmv_laplace_cdf_maxpts=25000,
    pmv_laplace_cdf_eps=1e-3,
):
    _WORKER_STATE["rf_model"] = rf_model
    _WORKER_STATE["min_score"] = min_score
    _WORKER_STATE["smoothwidth"] = smoothwidth
    _WORKER_STATE["cor_mat"] = cor_mat
    stats.set_pmv_laplace_cdf_options(pmv_laplace_cdf_maxpts, pmv_laplace_cdf_eps)


def _call_peak_block(task):
    chrom, peak_starts, peak_ends, infp_starts, infp_ends, infp_scores = task
    rf_model = _WORKER_STATE["rf_model"]
    min_score = _WORKER_STATE["min_score"]
    smoothwidth = _WORKER_STATE["smoothwidth"]
    cor_mat = _WORKER_STATE["cor_mat"]

    t0 = time.perf_counter()
    pmv_before = stats.get_pmv_laplace_profile()
    raw_rows = []
    for peak_start, peak_end in zip(peak_starts, peak_ends):
        lo = np.searchsorted(infp_starts, peak_start, side="left")
        hi = np.searchsorted(infp_ends, peak_end, side="right")
        xp, yp = infp_starts[lo:hi], infp_scores[lo:hi]
        if xp.shape[0] <= 3 or yp.max() <= min_score:
            continue

        result = rfsplit.find_rf_peaks(
            rf_model, xp, yp, amp_threshold=min_score, smoothwidth=smoothwidth, cor_mat=cor_mat
        )
        if result is None:
            continue
        result = result.copy()
        result.insert(0, "chr", chrom)
        raw_rows.append(result)

    pmv_after = stats.get_pmv_laplace_profile()
    profile = {
        "peaks": len(peak_starts),
        "seconds": time.perf_counter() - t0,
        "pmv_calls": pmv_after["calls"] - pmv_before["calls"],
        "pmv_seconds": pmv_after["seconds"] - pmv_before["seconds"],
        "pmv_cdf_evals": pmv_after["cdf_evals"] - pmv_before["cdf_evals"],
    }
    if not raw_rows:
        return None, profile
    return pd.concat(raw_rows, ignore_index=True), profile


def _peak_calling_tasks(dense_sorted, candidates, chrom_col, start_col, end_col, block_width):
    for chrom, chr_peaks in candidates.groupby("chrom", sort=False):
        chr_infp = dense_sorted[dense_sorted[chrom_col] == chrom]
        infp_starts = chr_infp[start_col].to_numpy()
        infp_ends = chr_infp[end_col].to_numpy()
        infp_scores = chr_infp.iloc[:, 3].to_numpy()
        peak_starts_all = chr_peaks["start"].to_numpy()
        peak_ends_all = chr_peaks["end"].to_numpy()

        for start in range(0, len(chr_peaks), block_width):
            end = min(start + block_width, len(chr_peaks))
            block_starts = peak_starts_all[start:end]
            block_ends = peak_ends_all[start:end]

            # Mirror legacy dREG's block split: each worker gets only the
            # dense informative points spanning its 500 broad peaks.
            lo = np.searchsorted(infp_starts, block_starts[0], side="left")
            hi = np.searchsorted(infp_ends, block_ends[-1], side="right")
            yield (
                chrom,
                block_starts,
                block_ends,
                infp_starts[lo:hi],
                infp_ends[lo:hi],
                infp_scores[lo:hi],
            )


def call_peaks(
    dense_infp, peak_broad, min_score, rf_model,
    smoothwidth=4, pv_adjust="fdr", pv_threshold=0.05, progress=False,
    peak_calling_cores=1, peak_calling_block_width=100,
    pmv_laplace_cdf_maxpts=25000, pmv_laplace_cdf_eps=1e-3,
):
    """The find_rf_peaks-calling orchestration from peak_calling.R's
    start_calling(): one genome-wide cor_mat, then an independent call to
    rfsplit.find_rf_peaks() per broad peak whose max score clears
    min_score. When peak_calling_cores > 1, candidate broad peaks are split
    into blocks and processed in worker processes, matching legacy dREG's
    BLOCKWIDTH/snowfall execution model but allowing smaller blocks for
    better load balancing.

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
    if not np.isfinite(min_score):
        raise ValueError(f"min_score must be finite before peak calling, got {min_score}")

    cor_mat = stats.build_cormat(dense_infp[start_col].to_numpy(), dense_infp[score_col].to_numpy())

    if peak_broad is None or len(peak_broad) == 0:
        return None, None
    candidates = peak_broad[peak_broad["max"] >= min_score]
    if len(candidates) == 0:
        return None, None

    dense_sorted = dense_infp.sort_values([chrom_col, start_col], kind="stable")
    block_width = max(1, int(peak_calling_block_width))
    if pmv_laplace_cdf_maxpts is not None:
        pmv_laplace_cdf_maxpts = int(pmv_laplace_cdf_maxpts)
    pmv_laplace_cdf_eps = float(pmv_laplace_cdf_eps)
    tasks = list(_peak_calling_tasks(dense_sorted, candidates, chrom_col, start_col, end_col, block_width))
    logger.info(
        "calling %d broad peaks in %d blocks of up to %d with %d peak-calling core(s) "
        "(pmv_laplace_cdf_maxpts=%s, pmv_laplace_cdf_eps=%g)",
        len(candidates), len(tasks), block_width, peak_calling_cores,
        pmv_laplace_cdf_maxpts, pmv_laplace_cdf_eps,
    )
    raw_rows = []
    completed_results = [None] * len(tasks)
    profile_totals = {
        "peaks": 0,
        "seconds": 0.0,
        "pmv_calls": 0,
        "pmv_seconds": 0.0,
        "pmv_cdf_evals": 0,
    }
    completed_blocks = 0
    last_profile_log = time.perf_counter()
    pbar = tqdm(
        total=len(candidates), desc="calling peaks", unit="peak",
        disable=None if progress else True,
    )
    _init_peak_worker(
        rf_model,
        min_score,
        smoothwidth,
        cor_mat,
        pmv_laplace_cdf_maxpts,
        pmv_laplace_cdf_eps,
    )
    stats.reset_pmv_laplace_profile()

    def collect_profile(profile):
        for key in profile_totals:
            profile_totals[key] += profile[key]

    def maybe_log_progress(force=False):
        nonlocal last_profile_log
        now = time.perf_counter()
        if not force and now - last_profile_log < 60:
            return
        last_profile_log = now
        non_pmv_seconds = max(profile_totals["seconds"] - profile_totals["pmv_seconds"], 0.0)
        logger.info(
            "peak-calling progress: %d/%d blocks, %d peaks profiled, %.2fs block CPU, "
            "%.2fs in %d pmv_laplace call(s) / %d CDF eval(s), %.2fs non-pmv",
            completed_blocks, len(tasks), profile_totals["peaks"], profile_totals["seconds"],
            profile_totals["pmv_seconds"], profile_totals["pmv_calls"],
            profile_totals["pmv_cdf_evals"], non_pmv_seconds,
        )

    if peak_calling_cores and peak_calling_cores > 1 and len(tasks) > 1:
        try:
            with ProcessPoolExecutor(
                max_workers=peak_calling_cores,
                initializer=_init_peak_worker,
                initargs=(
                    rf_model,
                    min_score,
                    smoothwidth,
                    cor_mat,
                    pmv_laplace_cdf_maxpts,
                    pmv_laplace_cdf_eps,
                ),
            ) as pool:
                futures = {pool.submit(_call_peak_block, task): idx for idx, task in enumerate(tasks)}
                for future in as_completed(futures):
                    idx = futures[future]
                    task = tasks[idx]
                    result, profile = future.result()
                    collect_profile(profile)
                    completed_blocks += 1
                    completed_results[idx] = result
                    pbar.update(len(task[1]))
                    maybe_log_progress()
        except (OSError, NotImplementedError) as e:
            logger.warning("parallel peak calling unavailable (%s); falling back to serial", e)
            for idx, task in enumerate(tasks):
                result, profile = _call_peak_block(task)
                collect_profile(profile)
                completed_blocks += 1
                completed_results[idx] = result
                pbar.update(len(task[1]))
                maybe_log_progress()
    else:
        for idx, task in enumerate(tasks):
            result, profile = _call_peak_block(task)
            collect_profile(profile)
            completed_blocks += 1
            completed_results[idx] = result
            pbar.update(len(task[1]))
            maybe_log_progress()
    pbar.close()
    raw_rows = [result for result in completed_results if result is not None]
    non_pmv_seconds = max(profile_totals["seconds"] - profile_totals["pmv_seconds"], 0.0)
    logger.info(
        "peak-calling profile: %.2fs block CPU time, %.2fs in %d pmv_laplace call(s) / "
        "%d CDF eval(s), %.2fs non-pmv",
        profile_totals["seconds"], profile_totals["pmv_seconds"],
        profile_totals["pmv_calls"], profile_totals["pmv_cdf_evals"], non_pmv_seconds,
    )

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

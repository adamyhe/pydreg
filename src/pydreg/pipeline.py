"""Top-level orchestration mirroring run_dREG.R: io -> infp -> features ->
backend-scoring -> peaks -> output writers. Processes query positions in
backend-sized chunks (see docs/PLANNING.md "Batching") -- this module is
the only one that wires pydreg.io/features/backend/models together and
supplies peaks.py's score_fn callback.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import numpy as np
from tqdm.auto import tqdm

from . import backend, features, infp, io, peaks
from .models import DREGModel, DREGPeakSplitForest

logger = logging.getLogger(__name__)


@contextmanager
def _timed(name):
    """Logs how long the wrapped phase took -- cheap instrumentation for
    seeing where a run's wall-clock time actually goes."""
    t0 = time.perf_counter()
    yield
    logger.info("%s done in %.2fs", name, time.perf_counter() - t0)


def _iter_score_chunks(bed_df, chrom_col, start_col, chunk):
    """Yields (chrom, positions, centers) once per scoring chunk, flattened
    across every chromosome group in bed_df -- a flat sequence is what
    _score_positions's prefetch loop wants (one uniform "next chunk"
    boundary, including the last-chunk-of-one-chromosome ->
    first-chunk-of-the-next one), rather than a nested per-chromosome loop."""
    for chrom, group in bed_df.groupby(chrom_col, sort=False):
        positions = group.index.to_numpy()
        centers = group[start_col].to_numpy()
        for start in range(0, centers.shape[0], chunk):
            sl = slice(start, start + chunk)
            yield chrom, positions[sl], centers[sl]


def _score_positions(
    bw_plus, bw_minus, model, scorer, bed_df, chunk, progress=False, desc="scoring"
):
    """Scores every row of bed_df (columns chrom, start, ... positionally)
    and returns scores in the same row order. Groups by chromosome first
    (only peaks.py's gap-fill/densify steps can produce a multi-chromosome
    bed_df; the initial informative-position scan is already per-run).

    Overlaps each chunk's CPU-bound feature extraction (bigWig I/O +
    binning) with the *previous* chunk's scorer.predict() call, via a
    single background thread one chunk ahead -- these two steps were
    previously strictly sequential (extract, then predict, then extract
    the next chunk, ...), which left the GPU backends idle during every
    chunk's extraction. This is scheduling only, not a formula change: the
    same feature-extraction/scoring calls run on the same inputs in the
    same order, just overlapped. Safe with a single background thread
    specifically because it's the *only* thread that ever touches
    bw_plus/bw_minus -- the main thread never reads a bigWig while a
    background extraction is in flight, and a ThreadPoolExecutor with
    max_workers=1 guarantees at most one extraction ever runs at a time
    regardless of how far ahead a chunk gets submitted. The overlap itself
    relies on scorer.predict() releasing the GIL while it blocks on the GPU
    (true for CuPy's device-sync calls) -- on the numpy/sklearn CPU
    backends this prefetch still can't hurt correctness, just may not
    overlap as usefully since there's no GPU wait to hide behind.

    progress: show a tqdm progress bar over positions scored
    (auto-hidden if stdout isn't a terminal).

    Logs accumulated extract_seconds/predict_seconds once at the end (not
    per-chunk -- that would be exactly the kind of noisy progress line
    just demoted to DEBUG elsewhere): extract_seconds is the sum of the
    background thread's own per-chunk timings (nonlocal, but never written
    by more than one thread at a time -- see the prefetch note above);
    predict_seconds is timed on the main thread same as any other call.
    Since the two run concurrently, they don't sum to this call's wall
    time -- that's the whole point, and worth seeing directly rather than
    inferred from GPU-utilization graphs alone."""
    bed_df = bed_df.reset_index(drop=True)
    chrom_col, start_col = bed_df.columns[0], bed_df.columns[1]
    scores = np.empty(len(bed_df))
    extract_seconds = 0.0
    predict_seconds = 0.0

    def extract(item):
        nonlocal extract_seconds
        t0 = time.perf_counter()
        chrom, positions, centers = item
        X = features.extract_features_batch(
            bw_plus, bw_minus, chrom, centers, model.window_sizes, model.half_n_windows
        )
        extract_seconds += time.perf_counter() - t0
        return positions, X

    pbar = tqdm(
        total=len(bed_df), desc=desc, unit="pos", disable=None if progress else True
    )
    chunks = _iter_score_chunks(bed_df, chrom_col, start_col, chunk)
    with ThreadPoolExecutor(max_workers=1) as pool:
        next_item = next(chunks, None)
        future = pool.submit(extract, next_item) if next_item is not None else None
        while future is not None:
            positions, X = future.result()

            next_item = next(chunks, None)
            future = pool.submit(extract, next_item) if next_item is not None else None

            t0 = time.perf_counter()
            scores[positions] = scorer.predict(X)
            predict_seconds += time.perf_counter() - t0
            pbar.update(len(positions))
    pbar.close()
    logger.info(
        "%s: %.2fs extracting features, %.2fs in scorer.predict "
        "(these overlap, so they don't sum to this step's wall time)",
        desc,
        extract_seconds,
        predict_seconds,
    )
    return scores


def _resolve_query_chunk(scorer_backend, query_chunk=None):
    if query_chunk is not None:
        return query_chunk
    return backend.DEFAULT_QUERY_CHUNK[scorer_backend]


def run(
    plus_bw_path,
    minus_bw_path,
    out_prefix,
    backend_name=None,
    smoothwidth=4,
    pv_adjust="fdr",
    pv_threshold=0.05,
    query_chunk=None,
    cupy_sv_chunk=None,
    peak_calling_cores=1,
    peak_calling_block_width=100,
    pmv_laplace_cdf_maxpts=25000,
    pmv_laplace_cdf_eps=1e-3,
    write_outputs=True,
    progress=False,
):
    """Runs the full dREG peak-calling pipeline on a pair of bigWig files
    and (by default) writes the standard output set alongside `out_prefix`.
    backend_name: None ("auto") or one of "cupy"/"sklearn"/"numpy" --
    see pydreg.backend. progress: show tqdm progress bars for the
    informative-position scan, position scoring, and peak calling (off by
    default for library use; pydreg.cli enables it; auto-hidden if stdout
    isn't a terminal regardless). Returns a dict with
    dense_infp/raw_peak/peak_bed/min_score for programmatic use regardless
    of write_outputs."""
    bw_plus = io.open_bigwig(plus_bw_path)
    bw_minus = io.open_bigwig(minus_bw_path)

    model = DREGModel.from_pretrained()
    rf_model = DREGPeakSplitForest.from_pretrained()
    scorer = backend.build_scorer(model, backend_name, cupy_sv_chunk=cupy_sv_chunk)
    chunk = _resolve_query_chunk(scorer.backend, query_chunk)
    logger.info("using %s backend (query_chunk=%d)", scorer.backend, chunk)

    logger.info("scanning informative positions...")
    with _timed("scanning informative positions"):
        infp_bed = infp.get_informative_positions(bw_plus, bw_minus, progress=progress)
    logger.info("%d informative positions found", len(infp_bed))

    logger.info("scoring informative positions...")
    with _timed("scoring informative positions"):
        infp_bed["score"] = _score_positions(
            bw_plus,
            bw_minus,
            model,
            scorer,
            infp_bed,
            chunk,
            progress=progress,
            desc="scoring informative positions",
        )

    def score_fn(bed_df, desc="scoring"):
        return _score_positions(
            bw_plus,
            bw_minus,
            model,
            scorer,
            bed_df,
            chunk,
            progress=progress,
            desc=desc,
        )

    logger.info("densifying and merging into broad peaks...")
    with _timed("densifying and merging into broad peaks"):
        dense_infp, peak_broad, min_score = peaks.get_dense_infp(infp_bed, score_fn)
    logger.info(
        "min_score=%.4f, %d dense positions, %s broad peaks",
        min_score,
        len(dense_infp),
        "0" if peak_broad is None else len(peak_broad),
    )

    logger.info("calling peaks...")
    with _timed("calling peaks"):
        raw_peak, peak_bed = peaks.call_peaks(
            dense_infp,
            peak_broad,
            min_score,
            rf_model,
            smoothwidth=smoothwidth,
            pv_adjust=pv_adjust,
            pv_threshold=pv_threshold,
            progress=progress,
            peak_calling_cores=peak_calling_cores,
            peak_calling_block_width=peak_calling_block_width,
            pmv_laplace_cdf_maxpts=pmv_laplace_cdf_maxpts,
            pmv_laplace_cdf_eps=pmv_laplace_cdf_eps,
        )
    logger.info(
        "%s raw candidate peaks, %s significant",
        "0" if raw_peak is None else len(raw_peak),
        "0" if peak_bed is None else len(peak_bed),
    )

    if write_outputs:
        with _timed("writing outputs"):
            _write_outputs(out_prefix, bw_plus, dense_infp, raw_peak, peak_bed)

    return dict(
        dense_infp=dense_infp, raw_peak=raw_peak, peak_bed=peak_bed, min_score=min_score
    )


def _write_outputs(out_prefix, bw_plus, dense_infp, raw_peak, peak_bed):
    sizes = io.chrom_sizes(bw_plus)
    chrom_col, start_col, end_col = dense_infp.columns[:3]

    infp_out = dense_infp[[chrom_col, start_col, end_col, "score", "infp"]]
    io.write_bed_gz(infp_out, f"{out_prefix}.dREG.infp.bed.gz")
    io.write_bigwig(f"{out_prefix}.dREG.infp.bw", sizes, infp_out, value_col="score")

    if raw_peak is not None and len(raw_peak) > 0:
        io.write_bed_gz(raw_peak, f"{out_prefix}.dREG.raw.peak.bed.gz")

    if peak_bed is not None and len(peak_bed) > 0:
        io.write_bed_gz(peak_bed, f"{out_prefix}.dREG.peak.full.bed.gz")

        score_bed = peak_bed[["chr", "start", "end", "score"]]
        io.write_bed_gz(score_bed, f"{out_prefix}.dREG.peak.score.bed.gz")
        io.write_bigwig(
            f"{out_prefix}.dREG.peak.score.bw", sizes, score_bed, value_col="score"
        )

        prob_bed = peak_bed[["chr", "start", "end", "prob"]].copy()
        prob_bed["prob"] = 1 - prob_bed["prob"]
        io.write_bed_gz(prob_bed, f"{out_prefix}.dREG.peak.prob.bed.gz")
        io.write_bigwig(
            f"{out_prefix}.dREG.peak.prob.bw", sizes, prob_bed, value_col="prob"
        )

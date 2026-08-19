"""Top-level orchestration mirroring run_dREG.R: io -> infp -> features ->
backend-scoring -> peaks -> output writers. Processes query positions in
backend-sized chunks (see docs/PLANNING.md "Batching") -- this module is
the only one that wires pydreg.io/features/backend/models together and
supplies peaks.py's score_fn callback.
"""

import logging
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial

import numba
import numpy as np
import pybigtools
import threadpoolctl
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
    bw_plus,
    bw_minus,
    model,
    scorer,
    bed_df,
    chunk,
    progress=False,
    desc="scoring",
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
    same order, just overlapped. Safe with a single background thread at
    *this* level specifically because it's the only thread here that ever
    touches bw_plus/bw_minus -- the main thread never reads a bigWig while
    a background extraction is in flight, and a ThreadPoolExecutor with
    max_workers=1 guarantees at most one call into this level's extract()
    ever runs at a time regardless of how far ahead a chunk gets submitted.
    The overlap itself relies on scorer.predict() releasing the GIL while
    it blocks on the GPU (true for CuPy's device-sync calls) -- on the
    numpy/sklearn CPU backends this prefetch still can't hurt correctness,
    just may not overlap as usefully since there's no GPU wait to hide
    behind.

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
            bw_plus,
            bw_minus,
            chrom,
            centers,
            model.window_sizes,
            model.half_n_windows,
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


def _load_models(svr_model_path=None, rf_model_path=None):
    model = (
        DREGModel(svr_model_path)
        if svr_model_path is not None
        else DREGModel.from_pretrained()
    )
    rf_model = (
        DREGPeakSplitForest(rf_model_path)
        if rf_model_path is not None
        else DREGPeakSplitForest.from_pretrained()
    )
    return model, rf_model


def run(
    plus_bw_path,
    minus_bw_path,
    out_prefix,
    backend_name=None,
    svr_model_path=None,
    rf_model_path=None,
    smoothwidth=4,
    pv_adjust="fdr",
    pv_threshold=0.05,
    query_chunk=None,
    cupy_sv_chunk=None,
    mlx_sv_chunk=None,
    cores=1,
    peak_calling_block_width=100,
    pmv_laplace_cdf_maxpts=25000,
    pmv_laplace_cdf_eps=1e-3,
    pmv_laplace_tail_tol=0.0,
    write_outputs=True,
    progress=False,
):
    """Runs the full dREG peak-calling pipeline on a pair of bigWig files
    and (by default) writes the standard output set alongside `out_prefix`.
    backend_name: None ("auto") or one of "cupy"/"mlx"/"sklearn"/"numpy" --
    see pydreg.backend. svr_model_path/rf_model_path: optional local
    .safetensors[.zst] model files; omitted paths use the pretrained Hugging
    Face weights. progress: show tqdm progress bars for the informative-
    position scan, position scoring, and peak calling (off by default for
    library use; pydreg.cli enables it; auto-hidden if stdout isn't a terminal
    regardless). Returns a dict with dense_infp/raw_peak/peak_bed/min_score
    for programmatic use regardless of write_outputs.

    cores: one number applied consistently across every parallel stage in
    the pipeline -- peaks.call_peaks's ProcessPoolExecutor (worker
    processes for the final peak-calling stage), numba's thread count for
    the parallelized feature-extraction/informative-position-scanning/
    scoring kernels (features._binned_sums_batch_numba,
    infp._windowed_sums_numba, models._rbf_accumulate), the BLAS thread
    count for the CPU scoring backend's GEMMs (via threadpoolctl -- see
    below), and the worker pool that writes output files concurrently.
    Feature extraction itself (features.extract_features_batch) is
    single-threaded, one reader, regardless of cores -- see that
    function's own docstring for why. Deliberately one knob otherwise, not
    several independently-tunable ones -- a run restricted to N cores in
    one stage but left unrestricted (or serial) in another would both
    undersell available hardware and oversubscribe it, depending on which
    stage you looked at.

    The threadpoolctl call matters on its own: without it, BLAS defaults
    to auto-detecting the machine's core count and ignores `cores`
    entirely, so a run on a shared box that deliberately requests fewer
    cores than the machine has would still have every DREGModel.predict()
    GEMM silently grab every core anyway. Confirmed on real x86/Linux
    hardware (32-core, OpenBLAS via threadpoolctl) that BLAS and numba
    threads genuinely add capacity together here rather than contend for
    it -- max/max was the fastest of every (BLAS threads, numba threads)
    combination tested, not a regression -- so this is a correctness/
    consistency fix for what `cores` means, not a performance workaround;
    see docs/PERF_LOG.md's 2026-08-13 entries for the full sweep. Applied
    process-wide (not as a context manager, matching
    peaks._init_peak_worker's own use of threadpoolctl for the same
    reason) since it should hold for this whole run, not just one call --
    peak-calling's worker processes still independently pin themselves to
    a single BLAS thread each via their own initializer, unaffected by
    this main-process-wide setting."""
    numba.set_num_threads(cores)
    threadpoolctl.threadpool_limits(limits=cores)
    bw_plus = pybigtools.open(plus_bw_path)
    bw_minus = pybigtools.open(minus_bw_path)

    logger.info("loading models...")
    with _timed("loading models"):
        model, rf_model = _load_models(svr_model_path, rf_model_path)
    scorer = backend.build_scorer(
        model, backend_name, cupy_sv_chunk=cupy_sv_chunk, mlx_sv_chunk=mlx_sv_chunk
    )
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
            cores=cores,
            peak_calling_block_width=peak_calling_block_width,
            pmv_laplace_cdf_maxpts=pmv_laplace_cdf_maxpts,
            pmv_laplace_cdf_eps=pmv_laplace_cdf_eps,
            pmv_laplace_tail_tol=pmv_laplace_tail_tol,
        )
    logger.info(
        "%s raw candidate peaks, %s significant",
        "0" if raw_peak is None else len(raw_peak),
        "0" if peak_bed is None else len(peak_bed),
    )

    if write_outputs:
        with _timed("writing outputs"):
            _write_outputs(
                out_prefix, bw_plus, dense_infp, raw_peak, peak_bed, cores=cores
            )

    return {
        "dense_infp": dense_infp,
        "raw_peak": raw_peak,
        "peak_bed": peak_bed,
        "min_score": min_score,
    }


def _write_outputs(out_prefix, bw_plus, dense_infp, raw_peak, peak_bed, cores=1):
    """Writes the standard output set. Informative-position scores are
    written only as `.bw` -- the `.bed.gz` version was dropped: it
    duplicated the same data purely for debugging, was never read back in
    anywhere, and (being by far the largest output, one row per
    informative/gap-filled/densified position genome-wide) dominated this
    step's wall time for a file most runs never actually looked at.

    Every remaining file is independent of every other (distinct source
    DataFrame or a read-only slice of one, distinct output path), and
    neither io.write_bed_gz nor io.write_bigwig mutates its input
    DataFrame in place, so every write below is safe to run concurrently
    with every other, including the `.bed.gz`/`.bw` pairs that share
    score_bed/prob_bed as a read-only source.

    Dispatched across *two* pools, not one -- measured directly (not
    assumed) that the two writers behave oppositely under threading:
    `pysam.tabix_index`'s bgzip compression does release the GIL (~5.6x
    speedup threading 8 concurrent calls), but `pybigtools`' bigWig
    writer does not -- threading 4 concurrent write_bigwig calls measured
    **4x slower** than calling them serially (20.7s vs 5.2s), i.e. real
    lock contention inside its Rust binding, not just "no speedup". A
    `ProcessPoolExecutor` sidesteps that (2.5s for the same 4 files) at
    the cost of pickling each write's DataFrame across a process
    boundary -- cheap here since `io.py` itself imports nothing heavier
    than numpy/pybigtools (confirmed: ~0.2s pool startup, not the seconds
    a fresh numba/sklearn import would cost) and bigWig outputs are small
    now that the large infp `.bed.gz` is gone. `.bed.gz` writes stay on
    threads, which need no such workaround. Falls back to serial bigWig
    writes if process pools are unavailable in the current environment
    (mirrors peaks.call_peaks's own ProcessPoolExecutor fallback). `cores`
    is the same pipeline-wide budget as everywhere else, split between the
    two pools rather than a separate setting."""
    sizes = bw_plus.chroms()
    chrom_col, start_col, end_col = dense_infp.columns[:3]

    infp_out = dense_infp[[chrom_col, start_col, end_col, "score", "infp"]]
    bedgz_tasks = []
    bigwig_tasks = [
        partial(
            io.write_bigwig,
            f"{out_prefix}.dREG.infp.bw",
            sizes,
            infp_out,
            value_col="score",
        ),
    ]

    if raw_peak is not None and len(raw_peak) > 0:
        bedgz_tasks.append(
            partial(io.write_bed_gz, raw_peak, f"{out_prefix}.dREG.raw.peak.bed.gz")
        )

    if peak_bed is not None and len(peak_bed) > 0:
        bedgz_tasks.append(
            partial(io.write_bed_gz, peak_bed, f"{out_prefix}.dREG.peak.full.bed.gz")
        )

        score_bed = peak_bed[["chr", "start", "end", "score"]]
        bedgz_tasks.append(
            partial(io.write_bed_gz, score_bed, f"{out_prefix}.dREG.peak.score.bed.gz")
        )
        bigwig_tasks.append(
            partial(
                io.write_bigwig,
                f"{out_prefix}.dREG.peak.score.bw",
                sizes,
                score_bed,
                value_col="score",
            )
        )

        prob_bed = peak_bed[["chr", "start", "end", "prob"]].copy()
        prob_bed["prob"] = 1 - prob_bed["prob"]
        bedgz_tasks.append(
            partial(io.write_bed_gz, prob_bed, f"{out_prefix}.dREG.peak.prob.bed.gz")
        )
        bigwig_tasks.append(
            partial(
                io.write_bigwig,
                f"{out_prefix}.dREG.peak.prob.bw",
                sizes,
                prob_bed,
                value_col="prob",
            )
        )

    with ThreadPoolExecutor(
        max_workers=max(1, min(cores, len(bedgz_tasks)))
    ) as thread_pool:
        thread_futures = [thread_pool.submit(task) for task in bedgz_tasks]

        try:
            with ProcessPoolExecutor(
                max_workers=max(1, min(cores, len(bigwig_tasks)))
            ) as process_pool:
                process_futures = [process_pool.submit(task) for task in bigwig_tasks]
                for future in process_futures:
                    future.result()
        except (OSError, NotImplementedError) as e:
            logger.warning(
                "parallel bigWig writing unavailable (%s); falling back to serial", e
            )
            for task in bigwig_tasks:
                task()

        for future in thread_futures:
            future.result()

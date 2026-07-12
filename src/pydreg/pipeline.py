"""Top-level orchestration mirroring run_dREG.R: io -> infp -> features ->
backend-scoring -> peaks -> output writers. Processes query positions in
backend-sized chunks (see docs/PLANNING.md "Batching") -- this module is
the only one that wires pydreg.io/features/backend/models together and
supplies peaks.py's score_fn callback.
"""

import logging
import time
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


def _score_positions(bw_plus, bw_minus, model, scorer, bed_df, chunk, progress=False, desc="scoring"):
    """Scores every row of bed_df (columns chrom, start, ... positionally)
    and returns scores in the same row order. Groups by chromosome first
    (only peaks.py's gap-fill/densify steps can produce a multi-chromosome
    bed_df; the initial informative-position scan is already per-run).

    progress: show a tqdm progress bar over positions scored
    (auto-hidden if stdout isn't a terminal)."""
    bed_df = bed_df.reset_index(drop=True)
    chrom_col, start_col = bed_df.columns[0], bed_df.columns[1]
    scores = np.empty(len(bed_df))

    pbar = tqdm(total=len(bed_df), desc=desc, unit="pos", disable=None if progress else True)
    for chrom, group in bed_df.groupby(chrom_col, sort=False):
        positions = group.index.to_numpy()
        centers = group[start_col].to_numpy()
        for start in range(0, centers.shape[0], chunk):
            sl = slice(start, start + chunk)
            X = features.extract_features_batch(
                bw_plus, bw_minus, chrom, centers[sl], model.window_sizes, model.half_n_windows
            )
            scores[positions[sl]] = scorer.predict(X)
            pbar.update(len(centers[sl]))
    pbar.close()
    return scores


def _resolve_query_chunk(scorer_backend, query_chunk=None, cuml_query_chunk=800_000):
    if query_chunk is not None:
        return query_chunk
    if scorer_backend == "cuml" and cuml_query_chunk is not None:
        return cuml_query_chunk
    return backend.DEFAULT_QUERY_CHUNK[scorer_backend]


def _check_minus_strand_sign(
    bw_minus,
    infp_bed,
    flank=1000,
    max_windows=512,
    min_nonzero=50,
    max_positive_fraction=0.99,
):
    """Fail early when the minus bigWig looks positive-signed.

    dREG's pretrained feature scaling expects reverse-strand signal to be
    negative. Positive-signed minus tracks can still pass informative-position
    scanning, then produce an all-positive score distribution with no
    negative/zero noise tail for min_score estimation. Sampling around already
    discovered informative positions catches that convention mismatch before
    the expensive model-scoring pass.
    """
    if infp_bed is None or len(infp_bed) == 0:
        return

    chrom_col, start_col = infp_bed.columns[:2]
    take = min(max_windows, len(infp_bed))
    sample_idx = np.linspace(0, len(infp_bed) - 1, take, dtype=int)

    positive = 0
    negative = 0
    nonzero = 0
    minus_sizes = io.chrom_sizes(bw_minus)
    for _, row in infp_bed.iloc[sample_idx].iterrows():
        chrom = row[chrom_col]
        if chrom not in minus_sizes:
            continue
        center = int(row[start_col])
        values = io.fetch_raw(bw_minus, chrom, center - flank, center + flank + 1)
        finite = values[np.isfinite(values)]
        observed = finite[finite != 0]
        if observed.size == 0:
            continue
        positive += int(np.count_nonzero(observed > 0))
        negative += int(np.count_nonzero(observed < 0))
        nonzero += int(observed.size)

    if nonzero == 0:
        logger.info("minus-strand sign check skipped: no nonzero sampled minus signal")
        return

    positive_fraction = positive / nonzero
    logger.info(
        "minus-strand sign check: %d sampled nonzero bp, %d positive, %d negative",
        nonzero, positive, negative,
    )
    if nonzero >= min_nonzero and negative == 0 and positive_fraction >= max_positive_fraction:
        raise ValueError(
            "minus-strand bigWig appears to be positive-signed "
            f"({positive}/{nonzero} sampled nonzero values > 0, 0 < 0). "
            "pydreg expects the minus-strand bigWig to contain negative-signed "
            "reverse-strand signal, matching legacy dREG preprocessing. "
            "Invert the minus track values or rerun with --no-check-minus-sign "
            "if this dataset is intentionally nonstandard."
        )


def run(
    plus_bw_path,
    minus_bw_path,
    out_prefix,
    backend_name=None,
    smoothwidth=4,
    pv_adjust="fdr",
    pv_threshold=0.05,
    query_chunk=None,
    cuml_query_chunk=800_000,
    peak_calling_cores=1,
    peak_calling_block_width=100,
    pmv_laplace_cdf_maxpts=None,
    pmv_laplace_cdf_eps=1e-5,
    write_outputs=True,
    progress=False,
    check_minus_sign=True,
):
    """Runs the full dREG peak-calling pipeline on a pair of bigWig files
    and (by default) writes the standard output set alongside `out_prefix`.
    backend_name: None ("auto") or one of "cuml"/"sklearn"/"numpy" -- see
    pydreg.backend. progress: show tqdm progress bars for the
    informative-position scan, position scoring, and peak calling (off by
    default for library use; pydreg.cli enables it; auto-hidden if stdout
    isn't a terminal regardless). Returns a dict with
    dense_infp/raw_peak/peak_bed/min_score for programmatic use regardless
    of write_outputs."""
    bw_plus = io.open_bigwig(plus_bw_path)
    bw_minus = io.open_bigwig(minus_bw_path)

    model = DREGModel.from_pretrained()
    rf_model = DREGPeakSplitForest.from_pretrained()
    scorer = backend.build_scorer(model, backend_name)
    chunk = _resolve_query_chunk(scorer.backend, query_chunk, cuml_query_chunk)
    logger.info("using %s backend (query_chunk=%d)", scorer.backend, chunk)

    logger.info("scanning informative positions...")
    with _timed("scanning informative positions"):
        infp_bed = infp.get_informative_positions(bw_plus, bw_minus, progress=progress)
    logger.info("%d informative positions found", len(infp_bed))
    if check_minus_sign:
        _check_minus_strand_sign(bw_minus, infp_bed)

    logger.info("scoring informative positions...")
    with _timed("scoring informative positions"):
        infp_bed["score"] = _score_positions(
            bw_plus, bw_minus, model, scorer, infp_bed, chunk,
            progress=progress, desc="scoring informative positions",
        )

    def score_fn(bed_df):
        return _score_positions(
            bw_plus, bw_minus, model, scorer, bed_df, chunk,
            progress=progress, desc="scoring",
        )

    logger.info("densifying and merging into broad peaks...")
    with _timed("densifying and merging into broad peaks"):
        dense_infp, peak_broad, min_score = peaks.get_dense_infp(infp_bed, score_fn)
    logger.info(
        "min_score=%.4f, %d dense positions, %s broad peaks",
        min_score, len(dense_infp), "0" if peak_broad is None else len(peak_broad),
    )

    logger.info("calling peaks...")
    with _timed("calling peaks"):
        raw_peak, peak_bed = peaks.call_peaks(
            dense_infp, peak_broad, min_score, rf_model,
            smoothwidth=smoothwidth, pv_adjust=pv_adjust, pv_threshold=pv_threshold,
            progress=progress, peak_calling_cores=peak_calling_cores,
            peak_calling_block_width=peak_calling_block_width,
            pmv_laplace_cdf_maxpts=pmv_laplace_cdf_maxpts,
            pmv_laplace_cdf_eps=pmv_laplace_cdf_eps,
        )
    logger.info(
        "%s raw candidate peaks, %s significant",
        "0" if raw_peak is None else len(raw_peak), "0" if peak_bed is None else len(peak_bed),
    )

    if write_outputs:
        with _timed("writing outputs"):
            _write_outputs(out_prefix, bw_plus, dense_infp, raw_peak, peak_bed)

    return dict(dense_infp=dense_infp, raw_peak=raw_peak, peak_bed=peak_bed, min_score=min_score)


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
        io.write_bigwig(f"{out_prefix}.dREG.peak.score.bw", sizes, score_bed, value_col="score")

        prob_bed = peak_bed[["chr", "start", "end", "prob"]].copy()
        prob_bed["prob"] = 1 - prob_bed["prob"]
        io.write_bed_gz(prob_bed, f"{out_prefix}.dREG.peak.prob.bed.gz")
        io.write_bigwig(f"{out_prefix}.dREG.peak.prob.bw", sizes, prob_bed, value_col="prob")

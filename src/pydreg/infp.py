"""Informative-position scan, ported from get_informative_positions.R.

Only the use_ANDOR=True path is implemented -- the only one ever exercised by
the production pipeline (run_dREG.R / run_predict.R both call this with
depth=0, step=50, use_ANDOR=True; use_OR is dead code, unreachable when
use_ANDOR=True given the R branch order). The depthAND/windowAND/depthOR/
windowOR literals below are hardcoded in the R source itself (not derived
from a user-passed `depth` argument), so there is no `depth` parameter here.
"""

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from . import io

MIN_CHROM_SIZE = 2500
WINDOW_OR, DEPTH_OR = 100, 2
WINDOW_AND, DEPTH_AND = 1000, 0


def _windowed_sums_from_fine(fine, phase, window, step):
    """Derives io.windowed_sum(bw, chrom, phase, window, chrom_size) from a
    single step-resolution fetch, by summing consecutive fine bins -- exact
    (verified bit-for-bit against direct per-phase bigWig calls on real
    chr21 data), since WINDOW_OR/WINDOW_AND/phase are all multiples of
    `step`. Avoids up to 27 separate bw.values() calls per chromosome."""
    off = phase // step
    ratio = window // step
    n_bins = (fine.shape[0] - off) // ratio
    if n_bins <= 0:
        return np.zeros(0)
    usable = fine[off : off + n_bins * ratio]
    return usable.reshape(n_bins, ratio).sum(axis=1)


def get_informative_positions(bw_plus, bw_minus, window=400, step=50, progress=False):
    """bw_plus/bw_minus: open pybigtools readers (io.open_bigwig(...)).

    Chromosomes scanned = bw_plus's chromosomes with size > 2500 (strict).
    Known upstream bug, replicated faithfully (see docs/PLANNING.md): the
    minus-strand chromosome list is never actually consulted for chromosome
    selection in the original R (a `unique(x, incomparables)` call that looks
    like a union but isn't) -- the pretrained model's expected input
    distribution was produced by this exact scan.

    progress: show a tqdm progress bar over chromosomes (auto-hidden if
    stdout isn't a terminal, e.g. redirected to a file or CI log).

    Returns a DataFrame with columns chrom, start, end (1bp intervals,
    end = start + 1), one row per informative position, sorted and
    deduplicated within each chromosome."""
    # The fine-grid shortcut below (see _windowed_sums_from_fine) requires
    # WINDOW_OR/WINDOW_AND to be exact multiples of step -- true for every
    # real call site (step=50 always), but assert rather than silently
    # truncate if that's ever violated.
    assert WINDOW_OR % step == 0 and WINDOW_AND % step == 0
    phases = list(range(0, window + step, step))
    plus_sizes = io.chrom_sizes(bw_plus)
    minus_sizes = io.chrom_sizes(bw_minus)
    chroms = [c for c, size in plus_sizes.items() if size > MIN_CHROM_SIZE]

    rows = []
    for chrom in tqdm(
        chroms, desc="scanning chromosomes", unit="chrom",
        disable=None if progress else True,
    ):
        chrom_size = plus_sizes[chrom]
        has_minus = chrom in minus_sizes
        centers = []

        # WINDOW_OR/WINDOW_AND/every phase are all multiples of `step` -- one
        # step-resolution fetch per strand replaces up to 27 separate
        # per-phase bw.values() calls per chromosome (see
        # _windowed_sums_from_fine's docstring for the exactness proof).
        fine_plus = io.windowed_sum(bw_plus, chrom, 0, step, chrom_size)
        fine_minus = io.windowed_sum(bw_minus, chrom, 0, step, chrom_size) if has_minus else None

        for phase in phases:
            plus_or = _windowed_sums_from_fine(fine_plus, phase, WINDOW_OR, step)
            if has_minus:
                # OR pass: combined depth (plus signed + minus abs) per tile.
                minus_or = np.abs(
                    _windowed_sums_from_fine(fine_minus, phase, WINDOW_OR, step)
                )
                idx = np.nonzero((plus_or + minus_or) > DEPTH_OR)[0]
            else:
                # bw_minus lacks this chromosome: falls back to plus-only
                # depth thresholding (matches data.two_bigwig.OR's fallback
                # to data.one_bigwig, which thresholds on abs(plus_sum)).
                idx = np.nonzero(np.abs(plus_or) > DEPTH_OR)[0]
            centers.append(idx * WINDOW_OR + phase + WINDOW_OR // 2)

            if has_minus:
                # AND pass: both strands independently nonzero per tile.
                # bw_minus lacking the chromosome contributes nothing here
                # (matches data.two_bigwig returning no candidates).
                plus_and = _windowed_sums_from_fine(fine_plus, phase, WINDOW_AND, step)
                minus_and = np.abs(
                    _windowed_sums_from_fine(fine_minus, phase, WINDOW_AND, step)
                )
                idx = np.nonzero((plus_and > DEPTH_AND) & (minus_and > DEPTH_AND))[0]
                centers.append(idx * WINDOW_AND + phase + WINDOW_AND // 2)

        all_centers = (
            np.unique(np.concatenate(centers))
            if centers
            else np.array([], dtype=np.int64)
        )
        rows.append(
            pd.DataFrame({"chrom": chrom, "start": all_centers, "end": all_centers + 1})
        )

    if not rows:
        return pd.DataFrame(columns=["chrom", "start", "end"])
    return pd.concat(rows, ignore_index=True)

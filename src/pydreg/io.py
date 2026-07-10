"""bigWig I/O via pybigtools, and BED/tabix/bigWig output writers.

All disk I/O and no algorithmic content -- pydreg.infp and pydreg.features
call the read helpers here but never open file handles themselves.

Windowed-sum and raw-fetch semantics were verified directly against the
installed pybigtools package (not just its docs): `values(bins=n, summary=
"sum", exact=True, uncovered=0.0, fillna=0.0)` gives exact per-tile literal
sums (matching R's step.bpQuery.bigWig), and `values(start, end, fillna=0.0,
oob=0.0)` accepts out-of-chromosome-bounds start/end directly and returns a
full-length array zero-padded at the out-of-bounds positions -- no manual
clipping/padding needed on the Python side.
"""

import numpy as np
import pybigtools


def open_bigwig(path):
    return pybigtools.open(path)


def chrom_sizes(bw):
    return dict(bw.chroms())


def windowed_sum(bw, chrom, phase, window, chrom_size):
    """Exact literal sum of bw's signal in non-overlapping `window`-bp tiles
    starting at `phase`, tiling `[phase, phase + n_bins*window)`. Any trailing
    partial tile (narrower than `window`) at the chromosome end is dropped,
    matching get_informative_positions.R's assumed behavior (see
    docs/PLANNING.md). Returns an empty array if no full tile fits."""
    n_bins = (chrom_size - phase) // window
    if n_bins <= 0:
        return np.zeros(0)
    end = phase + n_bins * window
    return bw.values(
        chrom,
        phase,
        end,
        bins=n_bins,
        summary="sum",
        exact=True,
        uncovered=0.0,
        fillna=0.0,
    )


def fetch_raw(bw, chrom, start, end):
    """Raw per-bp signal over [start, end), zero-filled at uncovered
    positions AND at any portion of the range outside the chromosome
    (start < 0 or end > chrom size) -- always returns an array of length
    end - start."""
    return bw.values(chrom, start, end, fillna=0.0, oob=0.0)


def write_bed_gz(df, path, columns=None):
    """Sorts `df` by (chrom, start) and writes it as a bgzipped, tabix-indexed
    BED file at `path` (which should end in .bed.gz). `columns` selects and
    orders the columns to write (defaults to all of df's columns); the first
    three must be chrom, start, end. Returns the .bed.gz path."""
    if columns is not None:
        df = df[columns]
    chrom_col, start_col, end_col = df.columns[0], df.columns[1], df.columns[2]
    df = df.copy()
    # Genomic coordinates must be integers; upstream arithmetic (e.g. in
    # pydreg.rfsplit) can leave them as whole-numbered floats, which looks
    # sloppy in text output and breaks write_bigwig's strict int contract
    # below -- enforce int here, once, for every writer.
    df[start_col] = df[start_col].astype(int)
    df[end_col] = df[end_col].astype(int)
    df = df.sort_values([chrom_col, start_col], kind="stable")

    assert path.endswith(".bed.gz")
    plain_path = path[: -len(".gz")]
    df.to_csv(plain_path, sep="\t", header=False, index=False)

    import pysam

    return pysam.tabix_index(plain_path, preset="bed", force=True)


def write_bigwig(path, sizes, df, value_col=None):
    """Writes `df` (columns chrom, start, end, [value]) as a bigWig track at
    `path`. `value_col` names the value column if df has more than 3 columns
    (defaults to the 4th column). Input is sorted by (chrom, start) first --
    bigWig requires non-overlapping, sorted intervals."""
    chrom_col, start_col, end_col = df.columns[0], df.columns[1], df.columns[2]
    if value_col is None:
        value_col = df.columns[3]
    df = df.sort_values([chrom_col, start_col], kind="stable")

    bw = pybigtools.open(path, "w")
    # pybigtools' Rust binding requires plain Python int for start/end (not
    # float, not numpy int64) -- see write_bed_gz's docstring note; the
    # same upstream float contamination applies here.
    intervals = zip(
        df[chrom_col],
        df[start_col].astype(int),
        df[end_col].astype(int),
        df[value_col].astype(float),
    )
    bw.write(sizes, intervals)

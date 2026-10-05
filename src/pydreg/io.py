"""bigWig read helpers (operating on an already-open figwig.BigWigReader)
plus BED/tabix/bigWig output writers. Opening a reader is a direct
`figwig.BigWigReader(path)` call at its one real call site (pipeline.run) --
not wrapped here, since a bare pass-through added no behavior over calling
figwig directly. pydreg.infp and pydreg.features never open file handles
themselves either way; they only ever receive already-open readers.
"""

import numpy as np
from figwig import BigWigWriter


_N_JOBS = 1


def set_reader_threads(n):
    """Sets the thread count for figwig's internal block decompression,
    used by windowed_sum and fetch_raw. Called once from pipeline.run
    so that figwig's I/O parallelism honors the same --cores value as
    every other parallel stage."""
    global _N_JOBS
    _N_JOBS = n


_WINDOWED_SUM_CHUNK_BP = 5_000_000


def windowed_sum(bw, chrom, phase, window, chrom_size):
    """Exact literal sum of bw's signal in non-overlapping `window`-bp tiles
    starting at `phase`, tiling `[phase, phase + n_bins*window)`. Any trailing
    partial tile (narrower than `window`) at the chromosome end is dropped,
    matching get_informative_positions.R's assumed behavior (see
    docs/PLANNING.md). Returns an empty array if no full tile fits.

    Reads in chunks to bound transient memory: figwig returns raw
    base-resolution values (no server-side binning like pybigtools had),
    so a full-chromosome read would transiently allocate ~12 bytes per bp
    (float32 raw + float64 cast). Chunking caps this at ~60 MB per call
    regardless of chromosome size, which matters when infp scans many
    chromosomes concurrently."""
    n_bins = (chrom_size - phase) // window
    if n_bins <= 0:
        return np.zeros(0)

    chunk_bins = _WINDOWED_SUM_CHUNK_BP // window
    if n_bins <= chunk_bins:
        width = n_bins * window
        raw = bw.read([chrom], [phase], width=width, missing=0.0, n_jobs=_N_JOBS)
        return raw[0].astype(np.float64).reshape(n_bins, window).sum(axis=1)

    result = np.empty(n_bins, dtype=np.float64)
    offset = phase
    pos = 0
    remaining = n_bins
    while remaining > 0:
        this_bins = min(remaining, chunk_bins)
        this_width = this_bins * window
        raw = bw.read([chrom], [offset], width=this_width, missing=0.0, n_jobs=_N_JOBS)
        result[pos:pos + this_bins] = (
            raw[0].astype(np.float64).reshape(this_bins, window).sum(axis=1)
        )
        pos += this_bins
        offset += this_width
        remaining -= this_bins
    return result


def fetch_raw(bw, chrom, start, end):
    """Raw per-bp signal over [start, end), zero-filled at uncovered
    positions AND at any portion of the range outside the chromosome
    (start < 0 or end > chrom size). If the bigWig lacks `chrom` entirely,
    that strand contributes all-zero signal over the requested span. Always
    returns an array of length end - start."""
    length = end - start
    if length <= 0:
        return np.zeros(0, dtype=np.float64)
    if chrom not in bw.chrom_sizes:
        return np.zeros(length, dtype=np.float64)
    read_start = max(0, start)
    read_width = end - read_start
    if read_width <= 0:
        return np.zeros(length, dtype=np.float64)
    raw = bw.read([chrom], [read_start], width=read_width, missing=0.0,
                  n_jobs=_N_JOBS)
    vals = raw[0].astype(np.float64)
    np.nan_to_num(vals, copy=False, nan=0.0)
    if start < 0:
        result = np.zeros(length, dtype=np.float64)
        result[-start:] = vals
        return result
    return vals


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

    with BigWigWriter(path, sizes, n_jobs=_N_JOBS) as bw:
        bw.write(
            df[chrom_col].values,
            df[start_col].astype(int).values,
            df[value_col].astype(float).values,
            ends=df[end_col].astype(int).values,
        )

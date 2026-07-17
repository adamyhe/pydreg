import numpy as np
import pandas as pd
import pybigtools

from pydreg import io


def test_windowed_sum_matches_manual_reshape(synthetic_bigwig_pair):
    plus_path, _ = synthetic_bigwig_pair
    bw = io.open_bigwig(plus_path)
    chrom_size = io.chrom_sizes(bw)["chr1"]

    phase, window = 7, 100
    ws = io.windowed_sum(bw, "chr1", phase, window, chrom_size)
    n_bins = ws.shape[0]
    raw = io.fetch_raw(bw, "chr1", phase, phase + n_bins * window)
    manual = raw.reshape(n_bins, window).sum(axis=1)
    np.testing.assert_allclose(ws, manual)


def test_fetch_raw_zero_pads_out_of_bounds(synthetic_bigwig_pair):
    plus_path, _ = synthetic_bigwig_pair
    bw = io.open_bigwig(plus_path)
    raw = io.fetch_raw(bw, "chr1", -50, 50)
    assert raw.shape[0] == 100
    assert np.all(raw[:50] == 0)


def test_fetch_raw_missing_chromosome_returns_zeroes(tmp_path):
    path = str(tmp_path / "one_chrom.bw")
    bw = pybigtools.open(path, "w")
    bw.write({"chr1": 1000}, [("chr1", 10, 20, 1.0)])

    raw = io.fetch_raw(io.open_bigwig(path), "chrMissing", -25, 75)

    assert raw.shape[0] == 100
    assert np.all(raw == 0)


def test_write_bed_gz_sorts_and_tabix_indexes(tmp_path):
    df = pd.DataFrame(
        {"chrom": ["chr1"] * 3, "start": [300, 100, 200], "end": [301, 101, 201], "score": [0.5, 0.9, 0.1]}
    )
    path = str(tmp_path / "out.bed.gz")
    result = io.write_bed_gz(df, path)
    assert result == path

    import pysam

    tbx = pysam.TabixFile(result)
    rows = list(tbx.fetch("chr1", 150, 250))
    assert rows == ["chr1\t200\t201\t0.1"]


def test_write_bigwig_roundtrips_int_coordinates(tmp_path):
    # Coordinates as whole-numbered floats (as rfsplit.py's arithmetic can
    # produce) must not break the strict int contract pybigtools requires.
    df = pd.DataFrame(
        {"chrom": ["chr1", "chr1"], "start": [0.0, 10.0], "end": [10.0, 20.0], "score": [1.5, 2.5]}
    )
    path = str(tmp_path / "out.bw")
    io.write_bigwig(path, {"chr1": 1000}, df)

    r = io.open_bigwig(path)
    vals = r.values("chr1", 0, 20, fillna=0.0)
    np.testing.assert_allclose(vals[:10], 1.5)
    np.testing.assert_allclose(vals[10:], 2.5)

import numpy as np
from figwig import BigWigReader, BigWigWriter

from pydreg import infp


def test_dedupe_centers_matches_unique_on_overlapping_candidates():
    # Deliberately overlapping/duplicated ranges, the way real OR/AND
    # candidates from different phases overlap -- np.unique(np.concatenate(
    # centers)) is the reference this helper replaced. int64, matching
    # get_informative_positions's actual candidate arrays (np.nonzero()'s
    # output is always integer, even from an empty/float-dtype input).
    centers = [
        np.array([50, 10, 30, 10], dtype=np.int64),
        np.array([30, 60, 5], dtype=np.int64),
        np.array([], dtype=np.int64),
        np.array([99, 5, 60], dtype=np.int64),
    ]
    chrom_size = 100

    result = infp._dedupe_centers(chrom_size, centers)
    reference = np.unique(np.concatenate(centers))

    np.testing.assert_array_equal(result, reference)
    assert np.all(np.diff(result) > 0)  # sorted, no duplicates


def test_dedupe_centers_handles_empty_list():
    result = infp._dedupe_centers(1000, [])
    assert result.shape == (0,)
    assert result.dtype == np.int64


def test_dedupe_centers_handles_all_empty_arrays():
    empty = np.array([], dtype=np.int64)
    result = infp._dedupe_centers(1000, [empty, empty])
    assert result.shape == (0,)


def test_get_informative_positions_finds_the_synthetic_peak(synthetic_bigwig_pair):
    plus_path, minus_path = synthetic_bigwig_pair
    bw_plus = BigWigReader(plus_path)
    bw_minus = BigWigReader(minus_path)

    result = infp.get_informative_positions(bw_plus, bw_minus)

    assert list(result.columns) == ["chrom", "start", "end"]
    assert (result["end"] == result["start"] + 1).all()
    assert (result["chrom"] == "chr1").all()
    # sorted and deduplicated within the chromosome
    starts = result["start"].to_numpy()
    assert np.all(np.diff(starts) > 0)
    # the fixture's one strong, well-separated signal peak sits around
    # position 50,000 -- at least one informative position should land
    # near it.
    assert ((starts > 49_500) & (starts < 50_500)).any()


def test_get_informative_positions_threaded_matches_serial(synthetic_bigwig_pair):
    plus_path, minus_path = synthetic_bigwig_pair
    bw_plus = BigWigReader(plus_path)
    bw_minus = BigWigReader(minus_path)

    serial = infp.get_informative_positions(bw_plus, bw_minus, cores=1)
    threaded = infp.get_informative_positions(bw_plus, bw_minus, cores=4)

    np.testing.assert_array_equal(serial["start"].values, threaded["start"].values)
    np.testing.assert_array_equal(serial["chrom"].values, threaded["chrom"].values)


def test_get_informative_positions_handles_chrom_only_in_plus(tmp_path):
    plus_path = str(tmp_path / "plus.bw")
    minus_path = str(tmp_path / "minus.bw")

    with BigWigWriter(plus_path, {"chrUn_gl000233": 5000}) as bw:
        bw.write(["chrUn_gl000233"], [100], np.array([5.0], dtype=np.float32), ends=[300])
    with BigWigWriter(minus_path, {"chr1": 5000}) as bw:
        bw.write(["chr1"], [10], np.array([-1.0], dtype=np.float32), ends=[11])

    bw_plus = BigWigReader(plus_path)
    bw_minus = BigWigReader(minus_path)
    result = infp.get_informative_positions(bw_plus, bw_minus)

    assert set(result["chrom"]) == {"chrUn_gl000233"}


def test_get_informative_positions_returns_empty_frame_for_no_chromosomes(tmp_path):
    path = str(tmp_path / "tiny.bw")
    with BigWigWriter(path, {"tiny": 100}) as bw:
        bw.write(["tiny"], [0], np.array([1.0], dtype=np.float32), ends=[10])

    bw_plus = BigWigReader(path)
    bw_minus = BigWigReader(path)
    result = infp.get_informative_positions(bw_plus, bw_minus)

    assert list(result.columns) == ["chrom", "start", "end"]
    assert len(result) == 0

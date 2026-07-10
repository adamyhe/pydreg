import numpy as np
import pandas as pd

from pydreg.peaks import _r_colon, find_gap_infp, merge_broad_peak


def test_r_colon_matches_r_semantics():
    # R's `:` never returns empty, even when counting down.
    np.testing.assert_array_equal(_r_colon(1, 0), [1, 0])
    np.testing.assert_array_equal(_r_colon(0, 1), [0, 1])
    np.testing.assert_array_equal(_r_colon(1, 3), [1, 2, 3])
    np.testing.assert_array_equal(_r_colon(3, 1), [3, 2, 1])


def test_merge_broad_peak_drops_trailing_group():
    # Verified directly against a real R session: merge_broad_peak's
    # separation-index sentinel is only ever prepended (never appended), so
    # the last contiguous group of rows per chromosome is always dropped.
    # Gaps here (after -50/+50 padding): 100, 100, 1400, 100 -- only the
    # 3rd exceeds the join threshold (500), so rows 1-3 merge into one peak
    # and rows 4-5 (after the only separation point) are silently lost.
    df = pd.DataFrame(
        {
            "chrom": ["chr1"] * 5,
            "start": [200, 400, 600, 2100, 2300],
            "end": [200, 400, 600, 2100, 2300],
            "score": [1, 1, 1, 1, 1],
        }
    )
    result = merge_broad_peak(df, threshold=0)
    assert len(result) == 1
    assert result.iloc[0]["start"] == 150
    assert result.iloc[0]["end"] == 650


def test_merge_broad_peak_drops_lone_chromosome_with_no_separation():
    # If no gap >= `join` exists at all, the whole chromosome's one group
    # is dropped too (same underlying bug -- separe_idx is just [0]).
    df = pd.DataFrame(
        {"chrom": ["chr1"] * 3, "start": [100, 200, 300], "end": [100, 200, 300], "score": [1, 1, 1]}
    )
    assert merge_broad_peak(df, threshold=0) is None


def test_find_gap_infp_fills_gaps_near_high_scoring_sides():
    df = pd.DataFrame(
        {
            "chrom": ["chr1"] * 4,
            "start": [1000, 1080, 5000, 5300],
            "end": [1001, 1081, 5001, 5301],
            "score": [0.3, 0.25, 0.1, 0.05],
        }
    )
    gaps = find_gap_infp(df, threshold=0.2)
    # Existing positions (1000, 1080) must never be re-emitted as "gaps".
    assert not set(gaps["start"]) & {1000, 1080}
    assert set(gaps["start"]) == {1030, 1050, 1130}


def test_find_gap_infp_no_gaps_below_threshold():
    df = pd.DataFrame(
        {"chrom": ["chr1"] * 2, "start": [1000, 5000], "end": [1001, 5001], "score": [0.05, 0.05]}
    )
    assert find_gap_infp(df, threshold=0.2) is None

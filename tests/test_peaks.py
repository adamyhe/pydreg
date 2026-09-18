import numpy as np
import pandas as pd

from pydreg import peaks, stats
from pydreg.peaks import _r_colon, call_peaks, find_gap_infp, get_dense_infp, merge_broad_peak


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


def test_get_dense_infp_handles_no_negative_or_zero_scores():
    df = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [100, 200, 300],
            "end": [101, 201, 301],
            "score": [0.1, 0.2, 0.3],
        }
    )

    try:
        get_dense_infp(df, lambda bed_df: np.full(len(bed_df), 0.1))
    except ValueError as e:
        assert "could not estimate finite min_score" in str(e)
        assert "negative=0, zero=0" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_call_peaks_parallel_matches_serial(monkeypatch, rf_model):
    x1 = np.arange(0, 2000, 10)
    x2 = np.arange(3000, 5000, 10)
    x = np.concatenate([x1, x2])
    y1 = np.zeros_like(x1, dtype=float)
    y1 += 0.8 * np.exp(-((x1 - 500) ** 2) / (2 * 80**2))
    y1 += 0.9 * np.exp(-((x1 - 1500) ** 2) / (2 * 80**2))
    y2 = np.zeros_like(x2, dtype=float)
    y2 += 0.7 * np.exp(-((x2 - 3500) ** 2) / (2 * 80**2))
    y2 += 0.85 * np.exp(-((x2 - 4500) ** 2) / (2 * 80**2))
    y = np.concatenate([y1, y2])

    dense_infp = pd.DataFrame(
        {"chrom": "chr1", "start": x, "end": x + 1, "score": y, "infp": 1}
    )
    peak_broad = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [0, 3000],
            "end": [1990, 4990],
            "min": [0.0, 0.0],
            "max": [float(y1.max()), float(y2.max())],
            "mean": [float(y1.mean()), float(y2.mean())],
            "sum": [float(y1.sum()), float(y2.sum())],
            "stdev": [float(y1.std(ddof=1)), float(y2.std(ddof=1))],
            "count": [len(y1), len(y2)],
        }
    )

    monkeypatch.setattr(peaks, "PEAK_CALLING_BLOCK_WIDTH", 1)
    monkeypatch.setattr(peaks.stats, "build_cormat", lambda starts, scores: np.eye(5) * 0.01)
    serial_raw, serial_bed = call_peaks(
        dense_infp, peak_broad, 0.05, rf_model,
        pv_adjust="none", pv_threshold=1.0, cores=1,
    )
    parallel_raw, parallel_bed = call_peaks(
        dense_infp, peak_broad, 0.05, rf_model,
        pv_adjust="none", pv_threshold=1.0, cores=2,
    )

    assert serial_raw is not None
    assert parallel_raw is not None
    pd.testing.assert_frame_equal(
        serial_raw.drop(columns=["prob"]).reset_index(drop=True),
        parallel_raw.drop(columns=["prob"]).reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        serial_bed.drop(columns=["prob"]).reset_index(drop=True),
        parallel_bed.drop(columns=["prob"]).reset_index(drop=True),
    )


def test_call_peaks_threads_pmv_laplace_tail_tol_to_workers(monkeypatch, rf_model):
    x = np.arange(0, 2000, 10)
    y = 0.8 * np.exp(-((x - 500) ** 2) / (2 * 80**2))
    y += 0.9 * np.exp(-((x - 1500) ** 2) / (2 * 80**2))

    dense_infp = pd.DataFrame(
        {"chrom": "chr1", "start": x, "end": x + 1, "score": y, "infp": 1}
    )
    peak_broad = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [0],
            "end": [1990],
            "min": [0.0],
            "max": [float(y.max())],
            "mean": [float(y.mean())],
            "sum": [float(y.sum())],
            "stdev": [float(y.std(ddof=1))],
            "count": [len(y)],
        }
    )

    monkeypatch.setattr(peaks, "PEAK_CALLING_BLOCK_WIDTH", 1)
    monkeypatch.setattr(peaks.stats, "build_cormat", lambda starts, scores: np.eye(5) * 0.01)
    try:
        call_peaks(
            dense_infp, peak_broad, 0.05, rf_model,
            pv_adjust="none", pv_threshold=1.0, cores=1,
            pmv_laplace_tail_tol=1e-6,
        )
        # call_peaks's own serial-path _init_peak_worker call runs in this
        # (main) process regardless of `cores`, so the global set there is
        # observable afterward -- the real signal that the parameter
        # actually reached stats.set_pmv_laplace_tail_tol() rather than
        # being silently dropped somewhere in call_peaks's plumbing.
        assert stats.get_pmv_laplace_tail_tol() == 1e-6
    finally:
        stats.set_pmv_laplace_tail_tol(0.0)


def test_attach_seed_keys_is_positional_not_dispatch_dependent():
    tasks = [("a",), ("b",), ("c",)]
    assert peaks._attach_seed_keys(tasks, None) == [
        ("a", None), ("b", None), ("c", None),
    ]
    assert peaks._attach_seed_keys(tasks, 7) == [
        ("a", (7, 0)), ("b", (7, 1)), ("c", (7, 2)),
    ]
    # The key must come from the block's index in the task list, so the
    # same block gets the same stream no matter how many workers run it or
    # in what order they finish. Re-keying a reordered list therefore gives
    # each block a different key -- which is exactly why the keys are
    # attached once, up front, before any dispatch happens.
    assert peaks._attach_seed_keys(tasks, 7) != peaks._attach_seed_keys(tasks[::-1], 7)


def test_call_peaks_seed_makes_probs_reproducible_across_cores(monkeypatch, rf_model):
    # test_call_peaks_parallel_matches_serial above has to drop the `prob`
    # column before comparing, because the QMC integration behind it is
    # unseeded. With a seed, prob has to match too -- across a different
    # `cores`, i.e. a different number of worker processes, a different
    # peak-to-worker assignment and a different completion order.
    x1 = np.arange(0, 2000, 10)
    x2 = np.arange(3000, 5000, 10)
    x = np.concatenate([x1, x2])
    y1 = np.zeros_like(x1, dtype=float)
    y1 += 0.8 * np.exp(-((x1 - 500) ** 2) / (2 * 80**2))
    y1 += 0.9 * np.exp(-((x1 - 1500) ** 2) / (2 * 80**2))
    y2 = np.zeros_like(x2, dtype=float)
    y2 += 0.7 * np.exp(-((x2 - 3500) ** 2) / (2 * 80**2))
    y2 += 0.85 * np.exp(-((x2 - 4500) ** 2) / (2 * 80**2))
    y = np.concatenate([y1, y2])

    dense_infp = pd.DataFrame(
        {"chrom": "chr1", "start": x, "end": x + 1, "score": y, "infp": 1}
    )
    peak_broad = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [0, 3000],
            "end": [1990, 4990],
            "min": [0.0, 0.0],
            "max": [float(y1.max()), float(y2.max())],
            "mean": [float(y1.mean()), float(y2.mean())],
            "sum": [float(y1.sum()), float(y2.sum())],
            "stdev": [float(y1.std(ddof=1)), float(y2.std(ddof=1))],
            "count": [len(y1), len(y2)],
        }
    )

    monkeypatch.setattr(peaks, "PEAK_CALLING_BLOCK_WIDTH", 1)
    # A cor_mat with enough variance that these peaks' p-values are not
    # saturated: at np.eye(5) * 0.01 (what the tests above use) every
    # summit sits ~9 sd out, every QMC batch returns 1.0 to machine
    # precision, and the integral has no run-to-run variance at all --
    # so the seed would be untestable, not because it isn't wired up but
    # because there is nothing left to randomize.
    monkeypatch.setattr(peaks.stats, "build_cormat", lambda starts, scores: np.eye(5) * 0.5 + 0.2)

    def run(cores, seed):
        raw, _ = call_peaks(
            dense_infp, peak_broad, 0.05, rf_model,
            pv_adjust="none", pv_threshold=1.0, cores=cores,
            peak_calling_block_width=1, seed=seed,
        )
        return raw

    serial = run(1, 20260917)
    parallel = run(2, 20260917)
    rerun = run(1, 20260917)
    other_seed = run(1, 12345)

    assert serial is not None and parallel is not None
    pd.testing.assert_frame_equal(serial, parallel)
    pd.testing.assert_frame_equal(serial, rerun)
    # A different seed must actually redraw -- otherwise the equality
    # assertions above could hold for the wrong reason (a seed that never
    # reaches pmv_laplace at all).
    assert not np.allclose(serial["prob"].to_numpy(), other_seed["prob"].to_numpy())

import threading

import numpy as np
import pandas as pd

from pydreg import pipeline


class _FakeModel:
    window_sizes = np.array([1])
    half_n_windows = np.array([1])


class _RecordingScorer:
    """predict(X) -> the sum of each row -- a trivial, checkable function
    of X, not just zeros, so a bug in how chunks/positions get sliced or
    reassembled would actually show up in the result."""

    backend = "fake"

    def predict(self, X):
        return X.sum(axis=1)


def _fake_extract_features_batch(bw_plus, bw_minus, chrom, centers, window_sizes, half_n_windows):
    return np.asarray(centers, dtype=float)[:, None]


def test_iter_score_chunks_flattens_across_chromosomes_and_chunk_boundaries():
    bed_df = pd.DataFrame(
        {"chrom": ["chr1", "chr1", "chr1", "chr2", "chr2"], "start": [10, 20, 30, 5, 15]}
    )
    chunks = list(pipeline._iter_score_chunks(bed_df, "chrom", "start", chunk=2))

    assert [c[0] for c in chunks] == ["chr1", "chr1", "chr2"]
    np.testing.assert_array_equal(chunks[0][2], [10, 20])
    np.testing.assert_array_equal(chunks[1][2], [30])
    np.testing.assert_array_equal(chunks[2][2], [5, 15])
    # positions are bed_df's original integer index, not recomputed offsets
    np.testing.assert_array_equal(chunks[0][1], [0, 1])
    np.testing.assert_array_equal(chunks[2][1], [3, 4])


def test_score_positions_matches_naive_sequential_result_across_chunks_and_chromosomes(
    monkeypatch,
):
    monkeypatch.setattr(pipeline.features, "extract_features_batch", _fake_extract_features_batch)
    bed_df = pd.DataFrame(
        {"chrom": ["chr1", "chr1", "chr1", "chr2", "chr2"], "start": [10, 20, 30, 5, 15]}
    )

    scores = pipeline._score_positions(None, None, _FakeModel(), _RecordingScorer(), bed_df, chunk=2)

    # _RecordingScorer.predict sums each row of X, and _fake_extract_features_batch
    # returns [[center]] per row -- so the expected score IS each position's own
    # start value, in the DataFrame's original row order.
    np.testing.assert_array_equal(scores, [10, 20, 30, 5, 15])


def test_score_positions_prefetches_next_chunk_while_scoring_current(monkeypatch):
    extract_started = []
    second_extract_started = threading.Event()
    predict_started = threading.Event()
    unblock_predict = threading.Event()

    def fake_extract(bw_plus, bw_minus, chrom, centers, window_sizes, half_n_windows):
        extract_started.append(tuple(int(c) for c in centers))
        if len(extract_started) == 2:
            second_extract_started.set()
        return np.zeros((len(centers), 1))

    monkeypatch.setattr(pipeline.features, "extract_features_batch", fake_extract)

    class BlockingScorer:
        backend = "fake"

        def predict(self, X):
            predict_started.set()
            unblock_predict.wait(timeout=5)
            return np.zeros(len(X))

    bed_df = pd.DataFrame({"chrom": ["chr1"] * 6, "start": list(range(6))})
    result = {}

    def run():
        result["scores"] = pipeline._score_positions(
            None, None, _FakeModel(), BlockingScorer(), bed_df, chunk=2
        )

    t = threading.Thread(target=run)
    t.start()
    try:
        assert predict_started.wait(timeout=5), "first chunk's predict() never started"
        # While the first chunk's predict() is still blocked, the second
        # chunk's extraction should already be running in the background --
        # the actual overlap guarantee, not just "the whole call eventually
        # finishes correctly".
        assert second_extract_started.wait(timeout=5), (
            "second chunk's extraction did not start while the first "
            "chunk's predict() was still blocked -- prefetch isn't overlapping"
        )
    finally:
        unblock_predict.set()
        t.join(timeout=5)

    assert not t.is_alive()
    np.testing.assert_array_equal(result["scores"], np.zeros(6))


def test_resolve_query_chunk_uses_cuml_specific_default():
    assert pipeline._resolve_query_chunk("cuml") == 2**20
    assert pipeline._resolve_query_chunk("cuml", cuml_query_chunk=400_000) == 400_000
    assert pipeline._resolve_query_chunk("cuml", cuml_query_chunk=None) == 2**20
    assert pipeline._resolve_query_chunk("numpy") == 4096
    assert pipeline._resolve_query_chunk("sklearn") == 50_000
    assert pipeline._resolve_query_chunk("cuml", query_chunk=123) == 123
    assert pipeline._resolve_query_chunk("numpy", query_chunk=123, cuml_query_chunk=2**20) == 123


def test_pipeline_runs_end_to_end_on_synthetic_signal(synthetic_bigwig_pair, tmp_path, dreg_model, rf_model):
    plus_path, minus_path = synthetic_bigwig_pair
    out_prefix = str(tmp_path / "out")

    result = pipeline.run(plus_path, minus_path, out_prefix, backend_name="numpy")

    assert len(result["dense_infp"]) > 0
    assert result["min_score"] > 0
    # The synthetic fixture has one strong, well-separated signal peak
    # around position 50,000 -- expect it to be called significant.
    assert result["peak_bed"] is not None
    assert len(result["peak_bed"]) >= 1
    hit = result["peak_bed"]
    assert ((hit["start"] < 50200) & (hit["end"] > 49800)).any()

    import os

    assert os.path.exists(f"{out_prefix}.dREG.infp.bed.gz")
    assert os.path.exists(f"{out_prefix}.dREG.infp.bw")
    assert os.path.exists(f"{out_prefix}.dREG.peak.full.bed.gz")

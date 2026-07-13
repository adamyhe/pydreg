from pydreg import pipeline


def test_resolve_query_chunk_uses_cuml_specific_default():
    assert pipeline._resolve_query_chunk("cuml") == 800_000
    assert pipeline._resolve_query_chunk("cuml", cuml_query_chunk=400_000) == 400_000
    assert pipeline._resolve_query_chunk("cuml", cuml_query_chunk=None) == 800_000
    assert pipeline._resolve_query_chunk("numpy") == 4096
    assert pipeline._resolve_query_chunk("sklearn") == 50_000
    assert pipeline._resolve_query_chunk("cuml", query_chunk=123) == 123
    assert pipeline._resolve_query_chunk("numpy", query_chunk=123, cuml_query_chunk=800_000) == 123


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

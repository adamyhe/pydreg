from pydreg import pipeline


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

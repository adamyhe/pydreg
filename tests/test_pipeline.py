from pydreg import pipeline


def _write_chr1_bigwig(path, values):
    import pybigtools

    bw = pybigtools.open(str(path), "w")
    intervals = []
    i = 0
    while i < len(values):
        if values[i] != 0:
            j = i
            while j < len(values) and values[j] == values[i]:
                j += 1
            intervals.append(("chr1", i, j, float(values[i])))
            i = j
        else:
            i += 1
    bw.write({"chr1": len(values)}, intervals)
    return str(path)


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


def test_minus_strand_sign_check_rejects_positive_signed_signal(tmp_path):
    import numpy as np
    import pandas as pd
    from pydreg import io

    values = np.zeros(2000)
    values[900:1100] = 3
    minus_path = _write_chr1_bigwig(tmp_path / "minus_positive.bw", values)
    bw_minus = io.open_bigwig(minus_path)
    infp_bed = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [950, 1000, 1050],
            "end": [951, 1001, 1051],
        }
    )

    try:
        pipeline._check_minus_strand_sign(bw_minus, infp_bed, flank=100)
    except ValueError as e:
        assert "positive-signed" in str(e)
        assert "negative-signed" in str(e)
    else:
        raise AssertionError("positive-signed minus signal should be rejected")


def test_minus_strand_sign_check_accepts_negative_signed_signal(tmp_path):
    import numpy as np
    import pandas as pd
    from pydreg import io

    values = np.zeros(2000)
    values[900:1100] = -3
    minus_path = _write_chr1_bigwig(tmp_path / "minus_negative.bw", values)
    bw_minus = io.open_bigwig(minus_path)
    infp_bed = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [950, 1000, 1050],
            "end": [951, 1001, 1051],
        }
    )

    pipeline._check_minus_strand_sign(bw_minus, infp_bed, flank=100)

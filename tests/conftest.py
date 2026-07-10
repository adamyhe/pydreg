"""Shared fixtures. Model fixtures download from the `adamyhe/dREG` HF repo
and are skipped (not failed) if that's unreachable, so the dependency-free
unit tests (smoothing/stats/peaks) always run even offline."""

import numpy as np
import pybigtools
import pytest


@pytest.fixture
def synthetic_bigwig_pair(tmp_path):
    """A small, self-contained bigWig pair with one clear signal peak
    around position 50,000 on a 100,000bp "chr1" -- no dependency on the
    (gitignored) _reference/ example data."""
    rng = np.random.default_rng(0)
    chrom_size = 100_000
    length = chrom_size

    plus = np.zeros(length)
    minus = np.zeros(length)
    x = np.arange(length)
    plus += 8 * np.exp(-((x - 50200) ** 2) / (2 * 150**2))
    minus -= 6 * np.exp(-((x - 49800) ** 2) / (2 * 150**2))
    plus += rng.poisson(0.01, size=length)
    minus -= rng.poisson(0.01, size=length)

    paths = {}
    for strand, vals in (("plus", plus), ("minus", minus)):
        path = str(tmp_path / f"{strand}.bw")
        bw = pybigtools.open(path, "w")
        intervals = []
        i = 0
        while i < length:
            if vals[i] != 0:
                j = i
                while j < length and vals[j] == vals[i]:
                    j += 1
                intervals.append(("chr1", i, j, float(vals[i])))
                i = j
            else:
                i += 1
        bw.write({"chr1": chrom_size}, intervals)
        paths[strand] = path

    return paths["plus"], paths["minus"]


@pytest.fixture
def dreg_model():
    from pydreg.models import DREGModel

    try:
        return DREGModel.from_pretrained()
    except Exception as e:
        pytest.skip(f"could not download pretrained SVR model: {e}")


@pytest.fixture
def rf_model():
    from pydreg.models import DREGPeakSplitForest

    try:
        return DREGPeakSplitForest.from_pretrained()
    except Exception as e:
        pytest.skip(f"could not download pretrained RF model: {e}")

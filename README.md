# pydreg

[![PyPI](https://img.shields.io/pypi/v/pydreg)](https://pypi.org/project/pydreg/)
[![Tests](https://github.com/adamyhe/pydreg/actions/workflows/tests.yml/badge.svg)](https://github.com/adamyhe/pydreg/actions/workflows/tests.yml)
[![Weights](https://img.shields.io/badge/%F0%9F%A4%97-Weights-yellow)](https://huggingface.co/adamyhe/pydreg)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/pydreg?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/pydreg)

An inference-only Python port of [dREG](https://github.com/Danko-Lab/dREG) (Danko Lab) — detects active transcriptional regulatory elements (promoters and enhancers) from PRO-seq/GRO-seq nascent-transcription data.

Given a pair of strand-specific bigWig files, `pydreg` scores every informative genomic position with a pretrained SVR model, then calls significant peaks with FDR control, mirroring the original R package's recommended `run_dREG.R` pipeline end to end.

## Installation

```bash
pip install pydreg[gpu]
pydreg --help
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add pydreg[gpu]
pydreg --help
```

`[gpu]` installs [cuML](https://docs.rapids.ai/api/cuml/stable/) and enables GPU-accelerated scoring on Linux with an NVIDIA GPU. It is not required, but CPU-only scoring is much slower.

Pretrained model weights (an RBF-kernel SVR scorer and a small random-forest peak-splitter) are downloaded automatically from [`adamyhe/dREG`](https://huggingface.co/adamyhe/dREG) on Hugging Face the first time they're needed, and cached locally by `huggingface_hub`.

## Usage

### CLI

```bash
pydreg plus.bw minus.bw out_prefix --verbose
```

- `plus.bw`/`minus.bw`: strand-specific bigWig files (3′-mapped, point-mode, unnormalized read counts — the same input format the original dREG expects). See [proseq2.0](https://github.com/Danko-Lab/proseq2.0/) for the Danko lab's pipeline. **`minus.bw` must use negative-signed values** (proseq2.0's default convention) — see Caveats below.
- `out_prefix`: prefix for all output files (see below).

Options:

| flag | default | meaning |
|---|---|---|
| `--backend {auto,cuml,sklearn,numpy}` | `auto` | Scoring backend. `auto` uses cuML when CuPy sees a CUDA device, otherwise pure NumPy. scikit-learn is selectable explicitly but is not auto-selected. An explicit choice raises if that backend isn't actually usable, rather than silently falling back. |
| `--smoothwidth N` | `4` | Smoothing window used during peak-splitting. |
| `--pv-adjust METHOD` | `fdr` | Multiple-testing correction method (any `statsmodels.stats.multitest.multipletests` method name). |
| `--pv-threshold P` | `0.05` | Significance threshold applied after correction. |
| `--query-chunk N` | backend-specific | Positions scored per batch; defaults to a size tuned per backend (`pydreg.backend.DEFAULT_QUERY_CHUNK`). |
| `--cuml-query-chunk N` | `800000` | Positions scored per batch for cuML when `--query-chunk` is not set; ignored by CPU backends. |
| `-v`, `--verbose` | off | Log progress at INFO level. |

### Python API

```python
from pydreg import pipeline

result = pipeline.run("plus.bw", "minus.bw", "out_prefix", backend_name=None)
# result: {"dense_infp": ..., "raw_peak": ..., "peak_bed": ..., "min_score": ...}
```

Pass `write_outputs=False` to get the result dict back without writing files, if you just want to work with the DataFrames directly.

### Output files

Given `out_prefix`, pydreg writes:

| file | contents |
|---|---|
| `{out_prefix}.dREG.infp.bed.gz` (+`.tbi`), `.bw` | Every informative position and its raw dREG score. |
| `{out_prefix}.dREG.raw.peak.bed.gz` (+`.tbi`) | All candidate peaks before FDR filtering. |
| `{out_prefix}.dREG.peak.full.bed.gz` (+`.tbi`) | Significant peaks: chrom, start, end, score, p-value, center. |
| `{out_prefix}.dREG.peak.score.bed.gz`/`.bw` (+`.tbi`) | Significant peaks' scores only. |
| `{out_prefix}.dREG.peak.prob.bed.gz`/`.bw` (+`.tbi`) | Significant peaks' `1 - p-value`. |

`.bed.gz` files are bgzipped and tabix-indexed; `.bw` files are standard bigWig tracks.

## How it works

1. **Informative-position scan** — tiles the genome looking for positions with any transcriptional signal on either strand, to avoid scoring silent regions.
2. **Feature extraction** — for each informative position, bins nearby read counts into multiple nested window sizes ("zoom levels") per strand, producing a fixed-length feature vector.
3. **Scoring** — an RBF-kernel SVR (605,187 support vectors, trained on the original dREG data) maps each feature vector to a dREG score in ~[0, 1].
4. **Peak calling** — merges scored positions into broad candidate regions, refines local maxima with a small random-forest model to decide where to split adjacent peaks, computes a per-peak p-value, and applies FDR control to select significant peaks.

See `docs/PLANNING.md` for the full algorithmic detail (this is a faithful port, including several upstream R quirks reproduced deliberately since the pretrained model's expected behavior depends on them) and `docs/PERF_LOG.md` for the performance-optimization history.

## Development

Install `pydreg` from source:

```bash
git clone https://github.com/adamyhe/pydreg.git
cd pydreg
uv sync --extra gpu --group dev
uv run pytest tests/ -q
```

The test suite (25 tests) covers each module in isolation plus a full synthetic end-to-end pipeline run; model-dependent tests are skipped (not failed) if the Hugging Face repo is unreachable.

## Caveats

- **`minus.bw` must use negative-signed values, not unsigned/positive counts.** Informative-position detection is sign-invariant (it takes `abs()` of the minus-strand depth), but feature extraction is not: each strand's multi-scale bins are logistic-scaled independently and deliberately *without* `abs()` (ported verbatim from the original C, `get_max_d1`), since that's the exact transform the pretrained SVR's support vectors were fit against. A positive-signed `minus.bw` won't raise an error — positions are still found correctly — but every score will be silently wrong, because the reverse-strand features no longer match what the model expects.
- The cuML GPU backend is implemented against cuML's documented `SVR.from_sklearn()` API but has not been exercised on real GPU hardware (none was available during development) — do a real smoke test on CUDA hardware before relying on it in production.
- A handful of upstream R bugs/quirks are faithfully replicated rather than fixed, because the pretrained model's expected behavior was produced by that exact code (e.g. a `mean()`-argument-binding bug in the p-value calculation, an off-by-one in broad-peak merging that drops the last group per chromosome, and others) — see `docs/PLANNING.md` for the full list and reasoning.
- Peak-calling p-values have small inherent run-to-run noise (`scipy.stats.multivariate_normal.cdf`'s underlying quasi-Monte-Carlo algorithm is unseeded, matching the original R implementation's `mvtnorm::pmvnorm`, which is also unseeded) — this doesn't affect which peaks are called significant in practice.

## License

GPL-3.0 (matching the original dREG R package, which is GPL-3-licensed).

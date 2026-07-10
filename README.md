# pydreg

An inference-only Python port of [dREG](https://github.com/Danko-Lab/dREG) (Danko Lab) — detects active transcriptional regulatory elements (promoters and enhancers) from PRO-seq/GRO-seq nascent-transcription data.

Given a pair of strand-specific bigWig files, `pydreg` scores every informative genomic position with a pretrained SVR model, then calls significant peaks with FDR control, mirroring the original R package's recommended `run_dREG.R` pipeline end to end. Training is out of scope — this project only ports inference.

## Installation

Requires Python ≥3.11. Dependency management is via [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

For GPU-accelerated scoring on Linux with an NVIDIA GPU (via [cuML](https://docs.rapids.ai/api/cuml/stable/)):

```bash
uv sync --extra gpu
```

This is a no-op (not an error) on macOS/Windows or GPU-less machines — the `gpu` extra is a platform-restricted marker, not a hard requirement.

Pretrained model weights (an RBF-kernel SVR scorer and a small random-forest peak-splitter) are downloaded automatically from [`adamyhe/dREG`](https://huggingface.co/adamyhe/dREG) on Hugging Face the first time they're needed, and cached locally by `huggingface_hub` — no manual download step.

## Usage

### CLI

```bash
uv run pydreg plus.bw minus.bw out_prefix --verbose
```

- `plus.bw`/`minus.bw`: strand-specific bigWig files (5′-mapped, point-mode, unnormalized read counts — the same input format the original dREG expects).
- `out_prefix`: prefix for all output files (see below).

Options:

| flag | default | meaning |
|---|---|---|
| `--backend {auto,cuml,sklearn,numpy}` | `auto` | Scoring backend. `auto` tries cuML, then scikit-learn, then pure NumPy. An explicit choice raises if that backend isn't actually usable, rather than silently falling back. |
| `--smoothwidth N` | `4` | Smoothing window used during peak-splitting. |
| `--pv-adjust METHOD` | `fdr` | Multiple-testing correction method (any `statsmodels.stats.multitest.multipletests` method name). |
| `--pv-threshold P` | `0.05` | Significance threshold applied after correction. |
| `--query-chunk N` | backend-specific | Positions scored per batch; defaults to a size tuned per backend (`pydreg.backend.DEFAULT_QUERY_CHUNK`). |
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
| `{prefix}.dREG.infp.bed.gz` (+`.tbi`), `.bw` | Every informative position and its raw dREG score. |
| `{prefix}.dREG.raw.peak.bed.gz` (+`.tbi`) | All candidate peaks before FDR filtering. |
| `{prefix}.dREG.peak.full.bed.gz` (+`.tbi`) | Significant peaks: chrom, start, end, score, p-value, center. |
| `{prefix}.dREG.peak.score.bed.gz`/`.bw` (+`.tbi`) | Significant peaks' scores only. |
| `{prefix}.dREG.peak.prob.bed.gz`/`.bw` (+`.tbi`) | Significant peaks' `1 - p-value`. |

`.bed.gz` files are bgzipped and tabix-indexed; `.bw` files are standard bigWig tracks.

## How it works

1. **Informative-position scan** — tiles the genome looking for positions with any transcriptional signal on either strand, to avoid scoring silent regions.
2. **Feature extraction** — for each informative position, bins nearby read counts into multiple nested window sizes ("zoom levels") per strand, producing a fixed-length feature vector.
3. **Scoring** — an RBF-kernel SVR (605,187 support vectors, trained on the original dREG data) maps each feature vector to a dREG score in ~[0, 1].
4. **Peak calling** — merges scored positions into broad candidate regions, refines local maxima with a small random-forest model to decide where to split adjacent peaks, computes a per-peak p-value, and applies FDR control to select significant peaks.

See `docs/PLANNING.md` for the full algorithmic detail (this is a faithful port, including several upstream R quirks reproduced deliberately since the pretrained model's expected behavior depends on them) and `docs/PERF_LOG.md` for the performance-optimization history.

## Development

```bash
uv sync
uv run pytest tests/ -q
```

The test suite (25 tests) covers each module in isolation plus a full synthetic end-to-end pipeline run; model-dependent tests are skipped (not failed) if the Hugging Face repo is unreachable.

## Caveats

- The cuML GPU backend is implemented against cuML's documented `SVR.from_sklearn()` API but has not been exercised on real GPU hardware (none was available during development) — do a real smoke test on CUDA hardware before relying on it in production.
- A handful of upstream R bugs/quirks are faithfully replicated rather than fixed, because the pretrained model's expected behavior was produced by that exact code (e.g. a `mean()`-argument-binding bug in the p-value calculation, an off-by-one in broad-peak merging that drops the last group per chromosome, and others) — see `docs/PLANNING.md` for the full list and reasoning.
- Peak-calling p-values have small inherent run-to-run noise (`scipy.stats.multivariate_normal.cdf`'s underlying quasi-Monte-Carlo algorithm is unseeded, matching the original R implementation's `mvtnorm::pmvnorm`, which is also unseeded) — this doesn't affect which peaks are called significant in practice.

## License

GPL-3.0 (matching the original dREG R package, which is GPL-3-licensed).

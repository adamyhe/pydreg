# pydreg

[![PyPI](https://img.shields.io/pypi/v/pydreg)](https://pypi.org/project/pydreg/)
[![Tests](https://github.com/adamyhe/pydreg/actions/workflows/ci.yml/badge.svg)](https://github.com/adamyhe/pydreg/actions/workflows/ci.yml)
[![Weights](https://img.shields.io/badge/%F0%9F%A4%97-Weights-yellow)](https://huggingface.co/adamyhe/pydreg)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/pydreg?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/pydreg)

An inference-only Python port of [dREG](https://github.com/Danko-Lab/dREG) (Danko Lab) — detects active transcriptional regulatory elements (promoters and enhancers) from PRO-seq/GRO-seq nascent-transcription data.

Given a pair of strand-specific bigWig files, `pydreg` scores every informative genomic position with a pretrained SVR model, then calls significant peaks with FDR control, mirroring the original R package's recommended `run_dREG.R` pipeline end to end.

## Installation

```bash
pip install pydreg[gpu]
pydreg --help
```

Or with [uv](https://docs.astral.sh/uv/) — `uv tool install` if you only need the `pydreg` CLI (isolated environment, nothing else to manage):

```bash
uv tool install pydreg[gpu]
pydreg --help
```

If you want the Python API (`from pydreg import pipeline`, see below) available in your own project instead, use `uv add pydreg[gpu]` there, or `uv pip install pydreg[gpu]` into an already-active environment.

`[gpu]` installs [cuML](https://docs.rapids.ai/api/cuml/stable/) and enables GPU-accelerated scoring on Linux with an NVIDIA GPU. It is not required, but CPU-only scoring is **much** slower and so is not generally recommended.

**GPU requirement:** cuML needs a GPU with compute capability ≥7.0 (NVIDIA Volta or newer — e.g. Turing, Ampere, Ada, Hopper). Older GPUs (Pascal and earlier, e.g. the GTX 10-series/TITAN X) are not just unsupported — confirmed on real hardware, they silently return *wrong* scores rather than an error. `pydreg` checks compute capability up front and falls back to the CPU (`numpy`) backend automatically in `--backend auto` mode, or raises a clear error for an explicit `--backend cuml` request; see `docs/OPTIMIZATION.md` for the full writeup.

Pretrained model weights (an RBF-kernel SVR scorer and a small random-forest peak-splitter) are downloaded automatically from [`adamyhe/dREG`](https://huggingface.co/adamyhe/dREG) on Hugging Face the first time they're needed, and cached locally by `huggingface_hub`.

## Usage

### CLI

```bash
pydreg plus.bw minus.bw out_prefix --verbose
```

- `plus.bw`/`minus.bw`: strand-specific bigWig files (3′-mapped, point-mode, unnormalized **read counts** — the same input format the original dREG expects). See [proseq2.0](https://github.com/Danko-Lab/proseq2.0/) for the Danko lab's pipeline. `minus.bw` may be positive- or negative-signed — `pydreg` takes the absolute value of both strands during feature extraction (matching the original C implementation), so sign convention doesn't affect scoring.
- `out_prefix`: prefix for all output files (see below).

Options:

| flag                                  | default          | meaning                                                                                                                                                                                                                                                                                        |
| ------------------------------------- | ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--backend {auto,cuml,sklearn,numpy}` | `auto`           | Scoring backend. `auto` uses cuML when CuPy sees a CUDA device with compute capability ≥7.0, otherwise pure NumPy. scikit-learn is selectable explicitly but is not auto-selected (see `docs/OPTIMIZATION.md` for why). An explicit choice raises if that backend isn't actually usable, rather than silently falling back. |
| `--smoothwidth N`                     | `4`              | Smoothing window used during peak-splitting.                                                                                                                                                                                                                                                   |
| `--pv-adjust METHOD`                  | `fdr`            | Multiple-testing correction method (any `statsmodels.stats.multitest.multipletests` method name).                                                                                                                                                                                              |
| `--pv-threshold P`                    | `0.05`           | Significance threshold applied after correction.                                                                                                                                                                                                                                               |
| `--query-chunk N`                     | backend-specific | Positions scored per batch; defaults to a size tuned per backend (`pydreg.backend.DEFAULT_QUERY_CHUNK`).                                                                                                                                                                                       |
| `-c`, `--cuml-query-chunk N`          | `1048576`        | Positions scored per batch for cuML when `--query-chunk` is not set; ignored by CPU backends.                                                                                                                                                                                                  |
| `--peak-calling-cores N`              | `1`              | Worker processes for the final peak-calling stage (embarrassingly parallel across broad candidate peaks).                                                                                                                                                                                      |
| `--peak-calling-block-width N`        | `100`            | Candidate broad peaks handed to each peak-calling worker per task; smaller blocks improve load balancing on uneven peak sizes.                                                                                                                                                                 |
| `--pmv-laplace-cdf-maxpts N`          | `25000`          | Max integration points for the per-summit p-value's quasi-Monte-Carlo integral; matches R's `mvtnorm::pmvnorm()`/`GenzBretz()` default. Only lower this if you want to trade fidelity with R for further speed.                                                                                |
| `--pmv-laplace-cdf-eps EPS`           | `0.001`          | Absolute/relative tolerance for the same integral; also matches R's default. Only lower this (i.e. tighten precision) if you specifically want to exceed R's own reference precision, at real speed cost.                                                                                      |
| `--no-progress`                       | off              | Disable tqdm progress bars (auto-hidden anyway when stdout isn't a terminal).                                                                                                                                                                                                                  |
| `-v`, `--verbose`                     | off              | Log progress at INFO level.                                                                                                                                                                                                                                                                    |

### Python API

```python
from pydreg import pipeline

result = pipeline.run("plus.bw", "minus.bw", "out_prefix", backend_name=None)
# result: {"dense_infp": ..., "raw_peak": ..., "peak_bed": ..., "min_score": ...}
```

Pass `write_outputs=False` to get the result dict back without writing files, if you just want to work with the DataFrames directly.

### Output files

Given `out_prefix`, pydreg writes:

| file                                                  | contents                                                      |
| ----------------------------------------------------- | ------------------------------------------------------------- |
| `{out_prefix}.dREG.infp.bed.gz` (+`.tbi`), `.bw`      | Every informative position and its raw dREG score.            |
| `{out_prefix}.dREG.raw.peak.bed.gz` (+`.tbi`)         | All candidate peaks before FDR filtering.                     |
| `{out_prefix}.dREG.peak.full.bed.gz` (+`.tbi`)        | Significant peaks: chrom, start, end, score, p-value, center. |
| `{out_prefix}.dREG.peak.score.bed.gz`/`.bw` (+`.tbi`) | Significant peaks' scores only.                               |
| `{out_prefix}.dREG.peak.prob.bed.gz`/`.bw` (+`.tbi`)  | Significant peaks' `1 - p-value`.                             |

`.bed.gz` files are bgzipped and tabix-indexed; `.bw` files are standard bigWig tracks.

## How it works

1. **Informative-position scan** — tiles the genome looking for positions with any transcriptional signal on either strand, to avoid scoring silent regions.
2. **Feature extraction** — for each informative position, bins nearby read counts into multiple nested window sizes ("zoom levels") per strand, producing a fixed-length feature vector.
3. **Scoring** — an RBF-kernel SVR (605,187 support vectors, trained on the original dREG data) maps each feature vector to a dREG score in ~[0, 1].
4. **Peak calling** — merges scored positions into broad candidate regions, refines local maxima with a small random-forest model to decide where to split adjacent peaks, computes a per-peak p-value, and applies FDR control to select significant peaks.

See `docs/METHODS.md` for a plain-language walkthrough of each stage, and `docs/OPTIMIZATION.md` for the performance design choices layered on top without changing any of the above. `docs/PLANNING.md`/`docs/PERF_LOG.md` are the underlying comprehensive design/research records (full algorithmic spec, every upstream R quirk and why it's kept, every benchmark) for anyone going deeper.

This is validated directly against the original: on real test data, pydreg's called peaks agree with real dREG's at a >0.999 Jaccard index.

## Caveats

- `minus.bw`'s sign doesn't matter: both informative-position detection and feature extraction take the absolute value of the minus-strand signal (the latter matching the original C's `bigwig_readi(..., abs=1, ...)` read call, which strips sign from both strands before any binning) — see `docs/PLANNING.md` for the sourced trace.
- A handful of upstream R bugs/quirks are faithfully replicated rather than fixed, because the pretrained model's expected behavior was produced by that exact code (e.g. a `mean()`-argument-binding bug in the p-value calculation, an off-by-one in broad-peak merging that drops the last group per chromosome, and others) — see `docs/PLANNING.md` for the full list and reasoning.
- Peak-calling p-values have small inherent run-to-run noise (the per-summit p-value's underlying quasi-Monte-Carlo integral is unseeded, matching the original R implementation's `mvtnorm::pmvnorm`, which is also unseeded) — this doesn't affect which peaks are called significant in practice, and is reflected in the 0.999728 (not exactly 1.0) Jaccard index above.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for development setup, running tests, and what to read before making algorithmic or performance changes.

## License

GPL-3.0 (matching the original dREG R package, which is GPL-3-licensed).

## Citation

If you use this package, please cite the original dREG papers:

> Danko, C. G., Hyland, S. L., Core, L. J., Martins, A. L., Waters, C. T., Lee, H. W., Baranello, L., Yang, Z., Wong, S. E., Setola, V., Lee, S. K., ... & Siepel, A. (2015). Identification of active transcriptional regulatory elements from GRO-seq data. *Nature Methods*, 12(5), 433-438. https://doi.org/10.1038/nmeth.3329

> Wang, Z., Chu, T., Choate, L. A., & Danko, C. G. (2018). Identification of regulatory elements from nascent transcription using dREG. *Genome Research*, 29, 293–303. https://doi.org/10.1101/gr.238279.118

Please also cite the version number of this port to improve reproducibility.

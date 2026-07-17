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

`[gpu]` installs [CuPy](https://cupy.dev/) and enables GPU-accelerated scoring on Linux with an NVIDIA GPU (auto-selected whenever one's detected). It is not required, but CPU-only scoring is **much** slower and so is not generally recommended.

**GPU requirement:** compute capability ≥3.0 (essentially any CUDA-capable NVIDIA GPU from the last decade-plus, including older Pascal-class cards). Scoring runs pydreg's own RBF kernel implementation directly on a CuPy device array (`pydreg.backend._build_cupy_predict_fn`) and enables broad CUDA compatibility and fp32 support. v0.1.x used cuML, which came with significant GPU restrictions (no Pascal support) and no fp32 support — see `docs/OPTIMIZATION.md` for the full writeup, including why that library was dropped (real hardware confirmed it silently returned wrong scores below compute capability 7.0).

Pretrained model weights (an RBF-kernel SVR scorer and a small random-forest peak-splitter) are downloaded automatically from [`adamyhe/pydreg`](https://huggingface.co/adamyhe/pydreg) on Hugging Face the first time they're needed, and cached locally by `huggingface_hub`.

## Usage

### CLI

```bash
pydreg plus.bw minus.bw out_prefix --verbose
```

- `plus.bw`/`minus.bw`: strand-specific bigWig files (3′-mapped, point-mode, unnormalized **read counts** — the same input format the original dREG expects). See [proseq2.0](https://github.com/Danko-Lab/proseq2.0/) for the Danko lab's pipeline. `minus.bw` may be positive- or negative-signed — `pydreg` takes the absolute value of both strands during feature extraction (matching the original C implementation), so sign convention doesn't affect scoring.
- `out_prefix`: prefix for all output files (see below).

Options:

**Arguments meant to be set per-run/per-system.** These change performance and logging only, not the results — safe to tune.

| flag                                  | default          | meaning                                                                                                                                                                                                                                                                                                                                       |
| ------------------------------------- | ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--backend {auto,cupy,sklearn,numpy}` | `auto`           | Scoring backend. `auto` uses `cupy` when a usable CUDA device is detected, otherwise pure NumPy. scikit-learn is selectable explicitly but is not auto-selected (see `docs/OPTIMIZATION.md` for why). An explicit choice raises if that backend isn't actually usable, rather than silently falling back.                                     |
| `-p`, `--peak-calling-cores N`        | `1`              | Worker processes for the final peak-calling stage (embarrassingly parallel across broad candidate peaks). **Set this to the max number of cores you can spare on your machine** — the default of `1` is just a safe, non-surprising starting point, not a recommendation.                                                                     |
| `--peak-calling-block-width N`        | `100`            | Candidate broad peaks handed to each peak-calling worker per task; smaller blocks improve load balancing on uneven peak sizes. Tune alongside `--peak-calling-cores`.                                                                                                                                                                         |
| `--query-chunk N`                     | backend-specific | Positions scored per batch; defaults to a size tuned per backend (`pydreg.backend.DEFAULT_QUERY_CHUNK`). Pure batching — does not change scores.                                                                                                                                                                                              |
| `--cupy-sv-chunk N`                   | `32768`          | Support vectors (of 605,187) evaluated per GPU kernel/GEMM call for the `cupy` backend specifically. The main lever for trading GPU memory for fewer, larger, better-amortized kernel launches — real headroom varies by card, so sweep a few values on your target GPU (see `docs/OPTIMIZATION.md`). Pure batching — does not change scores. |
| `--no-progress`                       | off              | Disable tqdm progress bars (auto-hidden anyway when stdout isn't a terminal).                                                                                                                                                                                                                                                                 |
| `-v`, `--verbose`                     | off              | Log progress at INFO level.                                                                                                                                                                                                                                                                                                                   |

**Arguments that will change results.** `--pv-adjust`/`--pv-threshold` are genuine statistical choices. The rest reproduce specific constants/tolerances from legacy dREG's own hardcoded behavior; pydreg's >0.999 Jaccard agreement with real dREG was measured at their defaults, so moving off those defaults is unexplored territory — nothing downstream validates the result against R at other values.

| flag                         | default | meaning                                                                                                                                                                                                         |
| ---------------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--pv-adjust METHOD`         | `fdr`   | Multiple-testing correction method (any `statsmodels.stats.multitest.multipletests` method name).                                                                                                               |
| `--pv-threshold P`           | `0.05`  | Significance threshold applied after correction.                                                                                                                                                                |
| `--smoothwidth N`            | `4`     | Smoothing window used during peak-splitting; matches legacy dREG's own hardcoded `smoothwidth=4` in `find_rf_peaks`.                                                                                            |
| `--pmv-laplace-cdf-maxpts N` | `25000` | Max integration points for the per-summit p-value's quasi-Monte-Carlo integral; matches R's `mvtnorm::pmvnorm()`/`GenzBretz()` default. Only lower this if you want to trade fidelity with R for further speed. |
| `--pmv-laplace-cdf-eps EPS`  | `0.001` | Absolute/relative tolerance for the same integral; also matches R's default. Only lower this (i.e. tighten precision) if you specifically want to exceed R's own reference precision, at real speed cost.       |

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
- Chromosome/contig selection follows the original plus-strand-driven scan: contigs present only in `plus.bw` can still be scored, with the missing `minus.bw` contig treated as all-zero signal during feature extraction; contigs present only in `minus.bw` are not discovered by the initial informative-position scan.
- A handful of upstream R bugs/quirks are faithfully replicated rather than fixed, because the pretrained model's expected behavior was produced by that exact code (e.g. a `mean()`-argument-binding bug in the p-value calculation, an off-by-one in broad-peak merging that drops the last group per chromosome, and others) — see `docs/PLANNING.md` for the full list and reasoning.
- Peak-calling p-values have small inherent run-to-run noise (the per-summit p-value's underlying quasi-Monte-Carlo integral is unseeded, matching the original R implementation's `mvtnorm::pmvnorm`, which is also unseeded) — this doesn't affect which peaks are called significant in practice, and is reflected in the >0.999 (not exactly 1.0) Jaccard index above.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for development setup, running tests, and what to read before making algorithmic or performance changes.

## License

GPL-3.0 (matching the original dREG R package, which is GPL-3-licensed).

## Citation

If you use this package, please cite the original dREG papers:

> Danko, C. G., Hyland, S. L., Core, L. J., Martins, A. L., Waters, C. T., Lee, H. W., Baranello, L., Yang, Z., Wong, S. E., Setola, V., Lee, S. K., ... & Siepel, A. (2015). Identification of active transcriptional regulatory elements from GRO-seq data. *Nature Methods*, 12(5), 433-438. https://doi.org/10.1038/nmeth.3329

> Wang, Z., Chu, T., Choate, L. A., & Danko, C. G. (2018). Identification of regulatory elements from nascent transcription using dREG. *Genome Research*, 29, 293–303. https://doi.org/10.1101/gr.238279.118

Please also cite the version number of this port to improve reproducibility.

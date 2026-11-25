# Command-line reference

```bash
pydreg plus.bw minus.bw out_prefix [options]
```

`pydreg --help` prints the same options in short form. This page explains
what each one does and when you'd change it.

## Inputs

- **`plus.bw` / `minus.bw`**: strand-specific bigWig files of raw **read
  counts**. They should be 3′-mapped (one point per read), not normalized,
  and not extended over whole reads. This is the same input the original
  dREG expects. The Danko lab's
  [proseq2.0](https://github.com/Danko-Lab/proseq2.0/) pipeline produces
  them.
- `minus.bw` can hold positive or negative values. pydreg takes the
  absolute value of both strands, as the original does, so the sign
  convention doesn't affect results.
- **`out_prefix`**: prefix for every output file (see
  [Output files](#output-files)).

Before doing anything else, pydreg samples both bigWigs and stops with an
error if they look wrong: non-integer (normalized) values, a mix of
positive and negative values in one file, or reads that cover whole
regions instead of single 3′ points. This check is ported from dREG's
`check_bigwig`. `--skip-input-check` turns it off.

## Main options

Most people only need these.

| flag | default | what it does |
|---|---|---|
| `-p`, `--cores N` | `1` | How many CPU cores pydreg may use. One number covers every parallel step (see [Parallelism](OPTIMIZATION.md#one---cores-setting-for-everything)). **Set this to as many cores as you can spare.** The default of 1 is only a safe starting point. |
| `--backend {auto,cupy,mlx,sklearn,numpy}` | `auto` | Where the scoring step runs. `auto` uses an NVIDIA GPU (`cupy`) or an Apple Silicon GPU (`mlx`) when it finds a usable one, and the CPU (`numpy`) otherwise. If you name a backend explicitly and it can't be used, pydreg stops with an error instead of quietly falling back. `sklearn` is never picked automatically because it is much slower ([why](OPTIMIZATION.md#why-numpy-and-not-scikit-learn-on-the-cpu)). |
| `--tf32` | off | Run the GPU's main matrix multiply at reduced (TF32) precision, on hardware that supports it: NVIDIA compute capability 8.0+ (A100, L40S, RTX 30-series and newer) and Apple M5. About 1.2-1.5x faster end to end, but peaks agree a little less closely with real dREG (Jaccard 0.996 instead of 0.9998), and results then differ slightly between GPU models. Has no effect on CPU backends. ([details](OPTIMIZATION.md#gpu-precision-float32-and-tf32)) |
| `--seed N` | unset | Makes peak calling reproducible. See [Randomness and `--seed`](#randomness-and---seed). |
| `-v`, `--verbose` | off | Print progress messages. |
| `--no-progress` | off | Hide progress bars. They're hidden automatically when output isn't a terminal. |
| `--svr-model PATH` | download | Use a local SVR scoring model (`.safetensors` or `.safetensors.zst`) instead of downloading the pretrained one. |
| `--rf-model PATH` | download | Use a local random-forest peak-splitting model instead of downloading the pretrained one. |
| `--skip-input-check` | off | Skip the bigWig sanity check described above. |

## Tuning options

These only change how work is divided into batches. They never change
results, and the defaults work well on every system tested. In particular,
bigger chunks don't make things faster.

| flag | default | what it does |
|---|---|---|
| `--peak-calling-block-width N` | `100` | How many candidate regions each peak-calling worker takes at a time. Smaller blocks spread uneven work more evenly across cores. |
| `--query-chunk N` | 4096 (50,000 for `sklearn`) | How many positions are scored per batch. |
| `--cupy-sv-chunk N` | `16384` | How many of the model's 605,187 support vectors each NVIDIA GPU call handles. Larger values use more GPU memory without being faster. Lower it if you run out of GPU memory. |
| `--mlx-sv-chunk N` | `16384` | The same setting for Apple Silicon. Here, values well above the default are actually slower, and very large values can crash because of a Metal limit on the size of a single buffer. |

## Options that change results

`--pv-adjust` and `--pv-threshold` are ordinary statistical choices. The
others reproduce constants hardcoded in the original dREG. pydreg's
agreement with dREG was measured only at the defaults, so changing them
takes you somewhere nobody has checked.

| flag | default | what it does |
|---|---|---|
| `--pv-adjust METHOD` | `fdr` | Multiple-testing correction. Accepts any method name that `statsmodels.stats.multitest.multipletests` accepts (`fdr` is Benjamini-Hochberg). |
| `--pv-threshold P` | `0.05` | Keep peaks whose corrected p-value is below this. |
| `--smoothwidth N` | `4` | Smoothing window used when splitting broad regions into peaks. Matches dREG's hardcoded value. |
| `--pmv-laplace-cdf-maxpts N` | `25000` | Maximum number of sample points for the per-peak p-value integral. Matches R's default. Lower values are faster but agree less with R. |
| `--pmv-laplace-cdf-eps EPS` | `0.001` | Error tolerance for the same integral. Matches R's default. Smaller values are more precise than R and slower. |
| `--pmv-laplace-tail-tol TOL` | `1e-6` | Stops each p-value calculation early once the part left over is guaranteed to change the answer by less than `TOL`. This can only make a peak look *less* significant, never more. At the default, a full real run called exactly the same peaks as the exact calculation. `0.0` gives the exact calculation. ([details](OPTIMIZATION.md#an-optional-shortcut-that-is-on-by-default)) |
| `--pmv-laplace-exact` | off | Same as `--pmv-laplace-tail-tol 0.0`. Ignored if `--pmv-laplace-tail-tol` is also given. |

## Randomness and `--seed`

One step of the pipeline is random: the p-value for each peak is computed
by randomized sampling, as it is in the original R code. As a result,
running pydreg twice on the same data can give very slightly different
p-values. This almost never changes which peaks are called.

`--seed N` makes runs repeatable, including across different `--cores`
values. It is off by default so that this small run-to-run noise stays
visible. Repeatability holds on the same machine with the same package
versions and the same `--pmv-laplace-*` settings, not across platforms or
SciPy releases.

A useful check: run twice with two different seeds and compare the peak
sets. The peaks that differ are the ones close enough to the significance
threshold to be affected by noise.

## Output files

Given `out_prefix`, pydreg writes:

| file | contents |
|---|---|
| `{out_prefix}.dREG.infp.bw` | Every informative position and its raw dREG score. |
| `{out_prefix}.dREG.raw.peak.bed.gz` (+`.tbi`) | All candidate peaks before significance filtering. |
| `{out_prefix}.dREG.peak.full.bed.gz` (+`.tbi`) | Significant peaks: chrom, start, end, score, p-value, center. |
| `{out_prefix}.dREG.peak.score.bed.gz` / `.bw` (+`.tbi`) | Significant peaks' scores only. |
| `{out_prefix}.dREG.peak.prob.bed.gz` / `.bw` (+`.tbi`) | Significant peaks' `1 - p-value`. |

`.bed.gz` files are bgzipped and tabix-indexed. `.bw` files are standard
bigWig tracks.

## Python API

The CLI is a thin wrapper around `pipeline.run`, which takes the same
options as keyword arguments:

```python
from pydreg import pipeline

result = pipeline.run(
    "plus.bw",
    "minus.bw",
    "out_prefix",
    cores=8,
    seed=0,            # optional
    backend_name=None, # None = auto
    tf32=False,
)
# result: {"dense_infp": ..., "raw_peak": ..., "peak_bed": ..., "min_score": ...}
```

Pass `write_outputs=False` to get the results back as DataFrames without
writing any files. `svr_model_path=` and `rf_model_path=` correspond to
`--svr-model` and `--rf-model`.

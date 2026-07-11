# pydreg: end-to-end peak-calling package

> **Status: implemented.** This plan was executed as written — the module
> layout, algorithm specs, backend tiering, and packaging decisions below
> match `src/pydreg/` as built. Treat this document as the design record
> (why things are structured this way), not a to-do list. A follow-up
> vectorization/JIT optimization pass (feature-extraction I/O batching, a
> numba-JIT'd RF forest, several vectorized hot loops) was done after initial
> implementation and is *not* reflected in the algorithm descriptions below,
> since they were performance-only changes with no behavioral difference —
> see `docs/PERF_LOG.md` for that history instead.

## Context

`pydreg` is a from-scratch Python port of dREG (Danko Lab), currently only an R/C package (reference at `_reference/dREG/`, gitignored). Prior work this session already validated and ported the two pretrained model artifacts (an RBF-kernel SVR scoring model and a small random-forest peak-splitter) into portable `.safetensors`/`.safetensors.zst` files, now hosted at **`adamyhe/dREG` on Hugging Face** (`svm.model.safetensors.zst`, `rf.model.safetensors.zst`), with working NumPy-based loaders already in `scripts/dreg_model.py` and `scripts/dreg_rf_model.py`.

What's missing — and what this plan builds — is everything *around* those two models: reading bigWig input, finding informative genomic positions, extracting the 360-dim feature vectors the SVR expects, running the full peak-calling/FDR pipeline, and writing output files, all packaged as an installable, uv-managed Python package with GPU acceleration where available. No training functionality is in scope — this is inference-only, mirroring the original's recommended `run_dREG.R` pipeline (not the legacy broad-peak/dREG-HD path, which the upstream README itself says is no longer needed).

Three rounds of research (reading the R/C source directly, not from memory/docs) plus an architecture review pinned down every algorithmic detail and packaging mechanic below with verified precision — implement from this document without re-deriving the math.

## Decisions locked in

- Package name `pydreg`, managed with `uv` (`pyproject.toml` + `uv.lock`, both git-tracked).
- bigWig I/O via `pybigtools` (Rust-backed), not pyBigWig.
- Models hosted on HF at `adamyhe/dREG`; download/cache via `huggingface_hub.hf_hub_download` (hard core dependency — nearly every invocation needs it; no bundled local model).
- Scoring backend: three real fallback tiers, cuML → `sklearn.svm.SVR` → dependency-free NumPy, in that order.
- Full output parity with `run_dREG.R`: `.bed.gz`+`.tbi` for infp/peak.full/peak.score/peak.prob/raw.peak, plus `.bw` tracks for infp/peak.score/peak.prob. Zero external CLI/subprocess calls (no bedops/htslib/UCSC-tools) — replace with pandas + `pysam` (tabix bgzip/index) + `pybigtools` (bigwig write).
- Single-process for v1 (vectorized NumPy/GPU batching only); architecture must not preclude adding multiprocessing later.

## Module layout

```
src/pydreg/    # src-layout: prevents `import pydreg` from silently resolving to an
               # uninstalled source dir on sys.path, and keeps repo root decluttered
               # from scripts/, docs/, _reference/, _models/.
  __init__.py
  io.py         # pybigtools wrapper: chrom sizes, raw per-bp fetch (edge zero-fill),
                # windowed-sum tiling, BED/tabix writers (pandas+pysam), bigwig writers
                # (pybigtools write mode). All disk I/O, zero algorithmic content.
  infp.py       # get_informative_positions port: vectorized per-chromosome OR/AND scan.
                # Calls io.py's windowed-sum helpers; never opens file handles itself.
  features.py   # read_genomic_data port: the multi-scale logistic-scaled feature
                # extraction — the heavy numpy tiling/reshaping module.
  models.py     # DREGModel + to_sklearn_svr (from scripts/dreg_model.py) and
                # DREGPeakSplitForest (from scripts/dreg_rf_model.py), moved essentially
                # unchanged; adds from_pretrained() classmethods wrapping hf_hub_download.
  _safetensors_io.py  # unchanged from scripts/_safetensors_io.py; private helper used by
                # models.py.
  backend.py    # cuML/sklearn/NumPy tiered dispatch: detect_backend(), build_scorer()/
                # Scorer, per-tier default query-chunk sizes.
  stats.py      # Pure statistical primitives, no bed/model coupling: get_laplace_sigma/
                # quantile, build_cormat, pmvLaplace (log-grid multivariate-Laplace tail
                # integration).
  smoothing.py  # peak_calling_ext.R port: SegmentedSmooth/deriv/boxcar passes. Pure
                # numpy, isolated for unit-testability against known vectors.
  rfsplit.py    # find_rf_peaks/split_peak: per-broad-peak local-maxima detection + RF
                # merge/split decision + per-summit p-value. Depends on models.py,
                # smoothing.py, stats.py.
  peaks.py      # BED-DataFrame-shaped operations only: get_dense_infp (+find_gap_infp/
                # pred_dense_infp), merge_broad_peak, get_broadpeak_summary,
                # select_sig_peak (BH-FDR over rfsplit.py's candidates). Depends on
                # stats.py + rfsplit.py, not the reverse.
  pipeline.py   # top-level orchestration mirroring run_dREG.R: io -> infp -> features ->
                # backend-scoring -> peaks(+stats+rfsplit+smoothing+models) -> output
                # writers. Owns the per-chromosome loop and query-batch sizing.
  cli.py        # thin CLI entry point (mirrors run_dREG.bsh's argument shape).
```

Dependency direction is one-way: `pipeline.py` → everything; `peaks.py` → `stats.py`, `rfsplit.py`; `rfsplit.py` → `models.py`, `smoothing.py`, `stats.py`; `infp.py`/`features.py` → `io.py`'s helpers only.

`scripts/pack_safetensors.py`, `export_dreg_model.R`, `export_dreg_rf_model.R` stay in `scripts/` as one-time model-conversion tooling — not part of the installable package.

## Algorithms to port (exact specs — implement from these, do not re-derive from R)

### Informative-position scan (`infp.py`) — from `get_informative_positions.R`

Params used in production (both `run_dREG.R` and `run_predict.R`): `depth=0, window=400, step=50, use_ANDOR=True` (`use_OR` is dead code — branch order makes it unreachable when `use_ANDOR=True`).

- Chromosomes scanned = plus-strand bigWig's chromosomes with `chromSize > 2500` (strict). **Known upstream bug, replicate faithfully, do not fix**: the minus-strand chromosome list is computed in R but never actually merged in (`unique(q_chroms, bw_minus$chroms)` — R treats the 2nd arg as `incomparables`, not a vector to union). The pretrained model's expected input distribution was produced by this exact scan; "fixing" it would change results.
- Per chromosome, for each phase `x` in `[0, 50, 100, ..., 400]` (9 phases):
  - **OR pass**: tile width 100bp at phase `x`; keep tile if `plus_signed_sum + minus_abs_sum > 2`.
  - **AND pass**: tile width 1000bp at phase `x`; keep tile if `plus_signed_sum > 0 AND minus_abs_sum > 0`.
  - Candidate center for 0-based tile index `idx`, window `W`, phase `x`: `center = idx*W + x + floor(W/2)`.
- Concatenate OR-pass + AND-pass candidates, `unique(sort(...))` **per chromosome**.
- Output: `chrom, chromStart, chromEnd` = 1bp intervals (`chromEnd = chromStart+1`).
- Assumption to validate empirically once real bigWigs are in hand: trailing partial tiles at chromosome ends are dropped (the underlying R bigWig library's source wasn't available to confirm directly).

### Multi-scale feature extraction (`features.py`) — from `read_genomic_data.R` + `src/read_genomic_data.c`

Two independent scaling stages:
- **Stage A (build here)**: raw window bin sums → per-zoom/per-strand logistic scaling.
- **Stage B (already done, in `models.py`'s `DREGModel`)**: the SVR's own z-score normalization on top of Stage A's output, then RBF kernel eval, then y un-scaling.

Stage A, precisely:
- Center per position = `floor((start+end)/2)` (for `infp.py` output rows this is just `start`).
- `max_dist = max_i(window_sizes[i] * half_nWindows[i])` — for the pretrained model (`window_sizes=[10,25,50,500,5000]`, `half_nWindows=[10,10,30,20,20]`): **max_dist = 100,000**. Fetch raw per-bp forward/reverse counts over `[center-100000, center+100000]`.
- Per zoom `i` (`W`, `H`): bins are non-overlapping, contiguous, exactly `W` bp wide, **no separate center bin** (the exact center bp is excluded from every zoom).
  - Left flank bin `j∈[0,H-1]`: covers `[C-W*H+j*W, C-W*H+(j+1)*W-1]` (bin 0 farthest, bin H-1 adjacent-left).
  - Right flank bin `j∈[H,2H-1]`: covers `[C+(j-H)*W+1, C+(j-H+1)*W]` (bin H adjacent-right, bin 2H-1 farthest).
  - Vectorizes as: reshape the fetched left/right-of-center arrays into `(H, W)` blocks per zoom, `.sum(axis=1)` — smaller zooms simply use the innermost slice of the full `max_dist`-wide fetch.
- **Edge handling**: any bp `<0` or beyond available chromosome data contributes 0 (bins zero-initialized; skip, never NA/error). **No sign flip on reverse strand** — raw bigWig values accumulate as-is.
- **Layout — NOT interleaved per zoom**: `[FWD block: zoom0(2H0) | zoom1(2H1) | ...] || [REV block: zoom0(2H0) | zoom1(2H1) | ...]` — all-forward-then-all-reverse, zooms in `window_sizes` order. For the pretrained model: 180 fwd + 180 rev = 360 total. This exact order must match the pretrained weight vector bit-for-bit.
- **Logistic scaling**, independently per zoom AND per strand (10 independent computations for the pretrained model's 5 zooms × 2 strands):
  ```
  true_max = max(raw bin values in this zoom/strand's 2H bins)   # plain max, NO abs()
  MAX = 1 if true_max == 0 else 0.05 * true_max
  alpha = ln(99) / MAX
  scaled_j = 1 / (1 + exp(-alpha * (raw_j - MAX)))                for each of the 2H bins
  ```
  **Must replicate the no-`abs()` gotcha exactly** — reverse-strand raw values are typically negative, so `true_max` there is often small/negative; this is what the pretrained weights expect, not a bug to fix. All-zero windows scale to the floor value `0.01`, not `0` (via the `true_max==0 → MAX=1` fallback).
- Linear-scaling mode exists in R but is **not used by the pretrained model** (confirmed via `asvm$scaled` being all-`TRUE`) — support logistic only.

### Peak-calling orchestration (`peaks.py`/`rfsplit.py`/`stats.py`) — from `peak_calling.R` + `peak_calling_rf.R`

Call order:
1. `infp.py` scan → `features.py` extraction → `models.py`'s `DREGModel.predict` scores every informative position.
2. `get_dense_infp` (in `peaks.py`):
   - `min_score` = 99.9th-percentile quantile of `Laplace(0,σ)` fit to the **negative**-score tail only (assumed pure noise); `σ` from `stats.py`'s `get_laplace_sigma`, closed-form quantile (`center - scale*sign(u-0.5)*ln(1-2|u-0.5|)`), no external package needed.
   - `find_gap_infp` — per chromosome, fills 50bp-spaced points into promising gaps between informative positions, re-scores them (`infp` flag 0=gap-filled, 1=original).
   - `get_broadpeak_summary(threshold=0.05)`: `merge_broad_peak` (filter score≥threshold, pad ±50bp, sort by (chr,start), merge across gaps <500bp — a numpy diff/cumsum-groupable interval merge) + per-peak summary stats (min/max/mean/sum/std/count of scores within each merged peak — the one place R shells out to `bedmap`; replace with vectorized pandas/numpy groupby via sorted non-overlapping intervals + `np.searchsorted`).
   - `pred_dense_infp` — inside candidate broad peaks (max ≥ min_score), re-densify at 10bp, re-score, keep `pred>0.05`, merge in.
3. `start_calling` (split across `peaks.py`/`rfsplit.py`/`stats.py`):
   - `build_cormat` (`stats.py`): **one genome-wide** 5×5 Toeplitz covariance matrix of neighboring-point score autocorrelation (empirical correlation at lags 1-4 on 20bp-apart pairs; `sigma2` = truncated variance of noise-region scores). Pure numpy, computed once.
   - Filter broad peaks to `max >= min_score`. R's block-splitting + snowfall + temp-`.rdata` serialization is pure R-cluster-IPC with zero algorithmic content — port as a plain per-broad-peak loop (v1, single-process).
   - Per broad peak, independently, `rfsplit.py`'s `find_rf_peaks(model, xp, yp, SlopeThreshold=0.01, AmpThreshold=min_score, smoothwidth=4, smoothtype=2, cor_mat=cor_mat)`: needs `smoothing.py`'s `SegmentedSmooth`/`deriv` (boxcar smoothing passes, pure numpy) plus `models.py`'s `DREGPeakSplitForest` plus `stats.py`'s `pmvLaplace`.
   - Per-summit p-value (pre-BH) = `1 - pmvLaplace(y[i.sample], cor_mat)`: tail probability of a 5-dim multivariate-Laplace null (`Laplace = sqrt(Z)·N(0,Σ)`, `Z~Exp(1)`), integrated over a log-spaced `Z` grid using the Gaussian CDF at each point (`scipy.stats.multivariate_normal.cdf` in place of R's `mvtnorm::pmvnorm` — same Genz-Bretz family algorithm; port the same log-grid integration loop, not a single CDF call).
   - `select_sig_peak` (`peaks.py`): drop `prob==-1` sentinel rows (peak <100bp, no p-value computed), BH-FDR-adjust (`statsmodels.stats.multitest.multipletests(method="fdr_bh")`) across **all** candidate peaks genome-wide, keep `adj_p <= 0.05`.
4. Output writing (`io.py`, pure glue): sort + `pysam.tabix_index(preset="bed")` for `.bed.gz`+`.tbi`; `pybigtools.open(path,"w")` write mode for `.bw` tracks. No subprocess calls.

Full output set: `infp.bed.gz` (chr,start,end,score,infp-flag), `peak.full.bed.gz` (chr,start,end,score,prob,center=original.mode), `peak.score.bed.gz` (cols 1-4 of full), `peak.prob.bed.gz` (chr,start,end,1-prob), `raw.peak.bed.gz` (pre-FDR, 8 cols), plus `.bw` versions of infp/peak.score/peak.prob. `.bw` outputs are never read back in anywhere — pure visualization artifacts.

## Backend dispatch (`backend.py`)

- `detect_backend()`, `@functools.lru_cache(maxsize=1)` — probes once per process, lazily (never at module import time — importing `cuml` alone can take seconds and drags in cupy/numba-cuda/rmm, a bad tax on every invocation including `--help`).
  - `try: import cuml.svm` → `ModuleNotFoundError` means not installed → tier 2. If the import *succeeds*, that only proves the library is present, not that a GPU is visible — RAPIDS wheels install fine on a GPU-less box. Confirm real GPU usability with a cheap real op (construct a tiny `cuml.svm.SVR()`, `.predict()` a 1-row dummy array) wrapped in a broad `except Exception` (CUDA init failures aren't one clean exception type).
  - Log **different** messages for "cuml not installed" vs "cuml installed but no GPU detected" even though both fall through to the same next tier.
  - Then `try: import sklearn.svm` → tier 2, else tier 3 (NumPy, always available, zero extra deps).
- `build_scorer(dreg_model, backend=None)` constructs the tier's model object exactly once per model instance and **caches it on the `DREGModel` instance itself** (e.g. `dreg_model._scorer_cache`) — both `to_sklearn_svr()`'s dummy-fit-then-overwrite and `cuml...from_sklearn()`'s ~1.7GB host→device copy are expensive enough to matter.
- `Scorer` is a uniform `.predict(X_chunk) -> np.ndarray` wrapper — `pipeline.py` never branches on backend, only calls `scorer.predict(batch)`.
- CLI flag `--backend {auto,cuml,sklearn,numpy}`, default `auto`. Explicit `--backend cuml` **raises** (doesn't silently fall back) if cuML isn't usable — a user who asked for GPU wants a loud failure, not a silent 50x slowdown on a job sized for GPU throughput.

## Batching

Chunking responsibility lives in **`pipeline.py`**, not inside `DREGModel.predict` (which keeps its existing `chunk` param — over support vectors — unchanged; adding an outer query loop there would be a real behavior change, and only the pipeline layer sees feature-extraction's own memory footprint, which the model class shouldn't need to know about). `pipeline.py`'s per-chromosome loop slices informative positions into `query_chunk`-sized batches, backend-specific default sizes, CLI-overridable:

- **NumPy CPU tier**: existing SV-chunking (`chunk=20_000`) stays as-is; bound the *outer* query-chunk so the three same-shaped `(query_chunk, sv_chunk)` float64 temporaries (`cross`, `sqdist`, `K`) stay ≤~2GB transient: `query_chunk ≈ 4096`.
- **sklearn CPU tier**: libsvm's C predict loop evaluates row-by-row internally (not memory-bound the way the NumPy tier is) — chunk mainly for streaming/checkpointing consistency: `query_chunk ≈ 50,000`.
- **cuML GPU tier**: pass the full 605,187×360 SV matrix to `from_sklearn()` once, don't re-chunk over SVs; chunk only over queries to bound host→device transfer size. The CLI exposes a cuML-only `--cuml-query-chunk` defaulting to 800,000; `--query-chunk` still overrides all backends when set. Tune this on real GPUs by watching VRAM and throughput.

This also sets up future multiprocessing cleanly (each worker owns one query batch) without touching `DREGModel` or `backend.py`.

## HF model loading (`models.py`)

`from_pretrained()` classmethods directly on `DREGModel`/`DREGPeakSplitForest` (mirrors the `transformers`/`diffusers` convention users already expect; only 3 lines, keeps "how do I get a model" co-located with "what is a model"):

```python
@classmethod
def from_pretrained(cls, repo_id="adamyhe/dREG", filename="svm.model.safetensors.zst", **hf_kwargs):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=repo_id, filename=filename, **hf_kwargs)
    return cls(path)
```

(analogous for `DREGPeakSplitForest` with `filename="rf.model.safetensors.zst"`). `**hf_kwargs` passes through `revision`/`cache_dir`/`token`; omitted, uses HF's own default cache (`HF_HOME`/`HUGGINGFACE_HUB_CACHE`, ETag revalidation) — no custom caching logic anywhere in `pydreg`. Plain-constructor local-path loading stays fully supported unchanged (air-gapped/custom-model use).

## Packaging (`pyproject.toml`, uv)

**Verified against PyPI's JSON API directly** (not docs alone): as of RAPIDS 25.10, cuML ships as real wheels on plain PyPI (`cuml-cu12`, confirmed present, `manylinux_2_24/2_28`, `x86_64`/`aarch64` only — no macOS/Windows, ever, by design) — **no NVIDIA custom index needed**. (Landmine: bare `cuml` on PyPI is an unrelated squatted ancient package — never depend on it, always the CUDA-suffixed name.)

```toml
[project.optional-dependencies]
gpu = [
  "cuml-cu12>=25.10; sys_platform == 'linux' and platform_machine in 'x86_64 aarch64'",
]
```

The `sys_platform`/`platform_machine` marker means `uv sync`/`uv sync --extra gpu` on macOS or unsupported architectures simply omits the package for that platform — no error, no `[[tool.uv.index]]`/`[tool.uv.sources]` routing needed for v1 (document as a fallback appendix only, for a future RAPIDS reversion or wanting nightly builds via `pypi.anaconda.org/rapidsai-wheels-nightly/simple`, not needed now).

Update: `cuml-cu12` requires Python `>=3.11`; rather than living with the version-mismatch failure mode described above, `pyproject.toml`'s own `requires-python` floor was raised to `>=3.11` to match, so `uv sync --extra gpu` never fails on a Python-version mismatch in the first place.

Core dependencies: `numpy`, `pybigtools`, `pysam`, `pandas`, `scipy` (multivariate_normal), `statsmodels` (BH-FDR) or a lighter equivalent, `huggingface_hub`, `safetensors`, `zstandard`. `sklearn` as a core dependency too (needed for the sklearn fallback tier, not just to feed cuML). `cuml-cu12` only via the `gpu` extra as above.

## Verification plan

- Unit-test `smoothing.py`/`stats.py` primitives against known input/output vectors (isolated, no bed/model coupling — designed for exactly this).
- Cross-check `features.py`'s output against the pretrained SVR's expected input distribution: run a handful of real informative positions through the full `infp → features → models` chain and confirm scores land in the expected dREG range (~[0,1]), matching the manual NumPy/sklearn cross-validation already done earlier this session for the model layer itself.
- Validate `infp.py`'s trailing-partial-tile assumption empirically against a real bigWig pair once available (flagged as unverified from source alone).
- End-to-end smoke test: run the full pipeline on the example data already present in `_reference/dREG/example/` (`K562.chr21.plus.bw`/`.minus.bw`) and sanity-check output peak count/positions look plausible (chr21-scale, not genome-wide, so single-process v1 should complete in reasonable time).
- Confirm `--backend numpy` and `--backend sklearn` both run to completion on this dev machine (no GPU available here); `--backend cuml` can only be verified by inspection/CI on real GPU hardware — flag this explicitly when implementation is done, do not claim it's been run.
- `uv sync` (no extras) must succeed on macOS; `uv sync --extra gpu` is expected to no-op the cuml package on macOS (not fail) per the marker design above.

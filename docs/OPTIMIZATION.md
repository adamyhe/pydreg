# Performance design choices

`pydreg`'s guiding rule for every performance change is that it must not
change the pipeline's output: same scores, same peaks, same
faithfully-replicated R quirks (see `docs/PLANNING.md`) — verified against
the test suite and, for peak-calling changes, directly against real dREG
output (~0.999 Jaccard index on test data; see `docs/METHODS.md`). This
document is the distilled "why it's built this way" summary, for anyone
using or extending `pydreg`. `docs/PERF_LOG.md` has the full chronological
research log — every benchmark, every dead end, every source citation
behind the claims below.

## End-to-end performance

On an NVIDIA Titan Xp using 16 cores, pydreg is consistently faster and uses
less peak memory than dREG across completed paired experiments:

<p>
  <img src="../figures/plots/walltime.svg" alt="dREG versus pydreg walltime" width="45%">
  <img src="../figures/plots/memory.svg" alt="dREG versus pydreg peak RSS" width="45%">
</p>

## Scoring backends

Evaluating the pretrained SVR (605,187 support vectors) against every
informative position is dominated by one computation: an RBF kernel matrix
between the query positions and every support vector. `pydreg` offers four
backends (`--backend {auto,cupy,mlx,sklearn,numpy}`), all computing
identical math against the same pretrained weights:

- **NumPy** (default with no usable GPU): one chunked matmul
  (`X_scaled @ sv_block.T`) plus a numba-fused elementwise/reduce step,
  dispatching to whatever BLAS NumPy is linked against.
- **scikit-learn**: wraps the pretrained weights into an `sklearn.svm.SVR`
  (`to_sklearn_svr()`) and predicts through libsvm. Available
  (`--backend sklearn`) but never auto-selected.
- **cupy** (`pydreg[gpu]`, Linux + NVIDIA, auto-selected when a usable CUDA
  device is present): the same chunked-matmul formula as the NumPy tier, on
  a CuPy device array.
- **mlx** (`pydreg[mlx]`, macOS + Apple Silicon, auto-selected when a usable
  Metal GPU is present and cupy isn't applicable): the same formula again,
  on an MLX device array.

### Why NumPy, not scikit-learn, is the CPU default

libsvm's prediction path evaluates the kernel one query-support-vector pair
at a time (with a heap allocation per pair); `DREGModel.predict`'s chunked
matmul computes the whole kernel matrix in one BLAS call plus one fused
numba kernel for the elementwise step. That's a genuinely different
computational shape, not a tuning gap — parallelizing libsvm's loop
wouldn't close it, and Intel's oneDAL-accelerated `scikit-learn-intelex`
fork ships no macOS/ARM wheels and was never `.fit()` through this model
anyway. Measured on an Apple M4 at a 4096-query batch
(`scripts/bench_backends.py`): sklearn takes 269.3s against the fused NumPy
tier's 6.8s (**~39.6x** slower), both agreeing to ~1e-10. See
`docs/PERF_LOG.md`'s 2026-07-09 and 2026-07-14 entries.

### Why `cupy`, not `cuML`, is the GPU tier

The GPU tier used to be `cuml.svm.SVR` (via `from_sklearn()`). It was
dropped after RAPIDS/cuML dropped Pascal (compute capability < 7.0) support
in 24.02, and a Pascal-incompatible cuML build doesn't error on such
hardware — it silently returns wrong results. Confirmed on real hardware:
cuml 26.06.00 diverged from the NumPy reference by ~0.05 on a TITAN X
(Pascal), while the identical bigWig input on an A100 (compute capability
8.0) matched dREG at Jaccard > 0.999. `from_sklearn` itself only shipped in
cuML 25.02, a year after the Pascal drop, so no cuML release ever supported
both.

`pydreg.backend._build_cupy_predict_fn` sidesteps this by not routing
through any third-party SVM library — it's a near-verbatim port of
`DREGModel.predict`'s chunked RBF dual-sum formula onto a CuPy device
array. Being *the same formula*, not an independent implementation, gives
two things: no cross-library conversion risk to audit, and no
compute-capability floor (CuPy's array primitives support ≥3.0; RAPIDS's
Pascal drop was a policy choice about its own compiled kernels, not a
CUDA-wide limit). `_wrap_sklearn_like`'s first-batch smoke test (comparing
against the NumPy reference) stays in place regardless, as the backstop
that caught both the cuml divergence above and a real cupy fusion bug
during this tier's own development (below). Getting the old cuml tier to
float32 was investigated and rejected as unsafe — `from_sklearn()` hardcoded
`dtype=float64` with no override, and cuML's C++ layer picks its template
by that flag, so a mismatch wouldn't downgrade precision, it would
misinterpret raw bytes. See `docs/PERF_LOG.md`'s 2026-07-15 entries.

### Installing the GPU extra needs `[ctk]`

As of CuPy 14.x, `cupy-cuda12x` no longer bundles its own CUDA toolkit — it
uses `cuda-pathfinder` to locate one, system-installed or via CuPy's
`[ctk]` extra. On a machine with only an NVIDIA driver and no separately
installed toolkit, this fails at JIT-compile time with a confusing "CUDA
versions below 12 are not supported" error, because `nvidia-smi`'s "CUDA
Version" field reports driver capability, not toolkit availability. This
regressed specifically because dropping `cuml-cu12` also removed the
pip-installed CUDA runtime it transitively pulled in, which cupy had been
silently using. **Fix**: `pyproject.toml`'s `gpu` extra requests
`cupy-cuda12x[ctk]`, confirmed via `uv lock` to pull the full CUDA 12.x
toolkit back in as real pip dependencies. See `docs/PERF_LOG.md`'s
2026-07-16 entry.

### Kernel fusion and batch size

Two levers, in the order they're worth pulling:

1. **Fuse the elementwise glue between the two GEMMs.** The formula between
   them (`exp(-gamma * (sq_x + sq_sv - 2*cross))`) was originally ~5
   separate elementwise kernel launches, each round-tripping a full
   `(query_chunk, sv_chunk)` array to GPU global memory on what's a
   memory-bandwidth-bound step. `cupy.fuse()` was tried first and produced
   a real ~3.5e-4 divergence on actual GPU hardware (caught by the smoke
   test); `cupy.ElementwiseKernel` — CuPy's older, explicit-dtype mechanism
   — replaced it and cuts memory traffic roughly 5x. See `docs/PERF_LOG.md`'s
   2026-07-15 entry.
2. **Batch size** (`--query-chunk`, `--cupy-sv-chunk`/`--mlx-sv-chunk`) —
   real hardware showed this matters differently per platform:
   - **cupy**: a full `sv_chunk` sweep on a real TITAN Xp (4,096 through
     65,536) and A100 (up to no-chunking at all) left `predict()`
     throughput flat within ~2% — this step is purely memory-bandwidth
     bound, so there's no per-launch overhead for a bigger batch to
     amortize. What batch size *does* change is memory: ~1844 MB is a fixed
     floor (SV matrix + CUDA context) on the TITAN Xp regardless of
     `sv_chunk`, so `_build_cupy_predict_fn`'s default was lowered from
     32,768 to **16,384** — the largest value still at that floor,
     cutting peak usage ~45% (3379 MB → ~1844 MB) for zero throughput cost.
   - **mlx**: does *not* transfer the same conclusion — Metal's unified
     memory has no discrete-VRAM equivalent. A real M4 sweep found
     throughput flat from 4,096 through 32,768 but genuinely degrading
     beyond that (up to ~2x slower at 524,288), and an unchunked call can
     crash outright: Metal enforces a hard ~9.53 GB single-buffer
     allocation ceiling independent of total unified memory, which an
     unchunked `(4096, 605187)` float32 buffer (~9.92 GB) exceeds.
     `_build_mlx_predict_fn`'s default was likewise lowered to **16,384**,
     sitting in the same flat/cheap region as the old 32,768 default while
     using ~19% less peak memory.

   See `docs/PERF_LOG.md`'s 2026-08-20 and 2026-08-22 entries for the full
   sweeps.

### float32 on GPU: why it's safe here (and not on CPU)

A further, since-implemented lever: float32 instead of float64 for the two
GEMMs and the fused kernel (`y_scaled` still accumulates in float64, cheap
insurance against cross-chunk summation error). This matters most on
consumer GPUs (e.g. TITAN X), which have much lower float64 than float32
throughput.

This is a smaller risk than "changes arithmetic precision" usually implies:
the pretrained models are trained via **Rgtsvm** (dREG's GPU-accelerated
SVM tool), and tracing its actual CUDA source shows it uses float32
internally throughout, unconditionally, with no double-precision build
path ever exercised — the alphas/support-vectors/rho pydreg exports as
float64 never had more than float32 accuracy to begin with. Running cupy
inference at float32 doesn't trade away precision the model actually has.
This doesn't extend to the CPU tiers: libsvm (what `e1071` and
`sklearn.svm.SVR` bind to) is genuinely double-precision throughout its
predict path, so downcasting the NumPy/sklearn tiers would be a real
fidelity regression for a speed motivation that's GPU-specific.

Confirmed on real hardware: the float32 switch tripped the smoke test as
expected (`max_abs_diff` ~2.3e-4 to ~5.4e-4 against the float64 reference,
a classic catastrophic-cancellation effect in the expanded squared-distance
formula, not a bug — sklearn/libsvm independently agrees with the same
reference to ~6e-11). `_wrap_sklearn_like` now takes an `atol`/`rtol`
override, and `build_scorer()` passes `CUPY_SMOKE_TEST_ATOL = 1e-3` for
`cupy` specifically (`sklearn` keeps the default `1e-4`). See
`docs/PERF_LOG.md`'s 2026-07-15 and 2026-07-17 entries.

### The `mlx` tier (Apple Silicon)

`pydreg.backend._build_mlx_predict_fn` is the Apple Silicon counterpart to
the cupy tier above — the same near-verbatim dual-sum formula, run on an
[MLX](https://github.com/ml-explore/mlx) device array targeting the
machine's Metal GPU, developed and validated on real Apple M4 hardware.

float32 isn't a choice here — Metal has no float64 support at all (a
float64 matmul on an MLX GPU array raises `ValueError`), so unlike cupy's
deliberate tradeoff, there was never a double-precision option to give up.
MLX also has nowhere on-device to hold a float64 accumulator, so each
chunk's small `(query_chunk,)` result is pulled back to a host NumPy
float64 array and accumulated there — cheap, since it's one short vector
per iteration, not the full kernel matrix.

The elementwise glue is wrapped in `mx.compile`, MLX's graph-fusion
mechanism, which — unlike cupy's `cp.fuse()` — handled this function's
actual call pattern (including the ragged last chunk, since
`n_sv=605,187` isn't divisible by `sv_chunk`) correctly on the first try,
so no fallback to a hand-written Metal kernel was needed.

Confirmed against the real pretrained model: max-abs-diff ~1.1e-4 to
~3.4e-4 against the float64 NumPy reference (comfortably inside
`MLX_SMOKE_TEST_ATOL = 1e-3`, the same float32 cancellation limitation as
cupy). Speed: **~6.1x** faster than the NumPy tier at a 4096-query batch on
an Apple M4 (1.16s vs 7.06s per call, warm/post-compile, re-measured after
the NumPy tier's own numba-fusion speedup). See `docs/PERF_LOG.md`'s
2026-08-10 entry.

## Batching

Each backend gets its own default query-chunk size
(`pydreg.backend.DEFAULT_QUERY_CHUNK`, overridable via `--query-chunk`):
NumPy is bounded so the transient `(query_chunk, sv_chunk)` intermediate
stays memory-manageable (this tier is memory-bandwidth-bound, not
compute-bound); scikit-learn's chunk size is mainly for
streaming/checkpointing, since libsvm isn't memory-bound the same way;
cupy and mlx both materialize the same `(query_chunk, sv_chunk)`
intermediate directly (unlike the old cuml tier, which tiled it internally
in C++) and so reuse NumPy's default.

## Overlapping feature extraction with scoring

`pipeline._score_positions` alternates CPU-bound feature extraction
(bigWig I/O + multi-scale binning) with the backend's `scorer.predict()`
call. Real-hardware profiling on a TITAN X showed GPU utilization cycling
0–90% between chunks once the cupy tier's float32 downcast cut its own
compute time by an order of magnitude — a cost that was always there but
had been hidden behind a slower GPU kernel (the same effect showed up on
an A100 running cuml, independent of float32 — this bottleneck is
backend-agnostic).

`_score_positions` now runs a one-chunk-ahead prefetch: a single
background thread extracts the *next* chunk's features while the current
chunk's `scorer.predict()` blocks on the GPU. This is safe because exactly
one thread ever touches the bigWig readers at a time
(`ThreadPoolExecutor(max_workers=1)`), and the overlap works because
`scorer.predict()` releases the GIL while blocked on the GPU (true of
CuPy's device-sync calls) — it can't hurt correctness on the CPU-only
backends, it just may not overlap as usefully.

Real runs on a TITAN Xp and an A100 confirmed the aggregate
predict:extract ratio drops from 3.05x to 1.46x between the two cards as
GPUs get faster — extraction time is fixed and CPU-bound, so it becomes a
growing fraction of total time as `predict()` shrinks. Parallelizing
extraction itself across multiple independently-opened bigWig readers was
investigated (pybigtools' `BBIReader` isn't safely shareable across
threads, but is safely reopenable per-thread) and initially shipped, then
reverted after real production data showed each reader accumulates an
unbounded per-chromosome index cache with no eviction inside `bigtools`
(`--cores 16` cost ~1.6GB extra RSS, a consistent ~20% regression) — see
"Feature-extraction clustering" below. Separately, a numba-jitted
`features._binned_sums_batch` (fused, no gather-index arrays, later
`prange`-parallelized across positions) made the single-threaded path
itself 2-4x faster, which is what actually fixed the worst case
(gap-filled positions) rather than adding more readers. See
`docs/PERF_LOG.md`'s 2026-07-15 and 2026-08-10 entries.

## Informative-position scanning

`infp.get_informative_positions` scans each chromosome for candidate
positions passing an OR/AND read-depth filter. Profiling a realistic
2-chromosome synthetic bigWig found two Python-side steps costing more
combined than the actual bigWig I/O: deduplicating candidate positions via
`np.unique(np.concatenate(...))` (28% of total time, replaced with a
chromosome-sized boolean mask — same sorted+deduplicated result without a
comparison sort, ~3x faster), and `reshape(...).sum(axis=1)` inside
`_windowed_sums_from_fine` (35%, replaced with a numba kernel — NumPy's
generic N-dimensional reduction carries fixed per-row overhead that
dominates when each reduction is short, and dREG's most common case sums
only 2 elements per bin; measured **11x faster** in numba there). Combined:
**305ms → 146ms (~2.1x)** on the same synthetic scan. See
`docs/PERF_LOG.md`'s 2026-08-10 entry.

## One `--cores` knob, not several

`pipeline.run`'s `cores` parameter (`--cores`/`-p`) is applied consistently
across every parallel stage, not just peak calling:
`numba.set_num_threads(cores)` governs the numba-parallelized kernels
(`features._binned_sums_batch_numba`, `infp._windowed_sums_numba`,
`models._rbf_accumulate`), the same value feeds
`peaks.call_peaks`'s `ProcessPoolExecutor(max_workers=cores)`, and
`threadpoolctl.threadpool_limits(limits=cores)` is called alongside so BLAS
doesn't independently default to auto-detecting the whole machine
(confirmed harmless where no threadpoolctl-visible BLAS exists —
`threadpool_limits()` is a documented no-op there). Deliberately one
number: a pipeline where peak calling was capped at 4 processes while
numba/BLAS defaulted to every detected core elsewhere would both undersell
and oversubscribe hardware depending on which stage you looked at.

## Feature-extraction clustering: density-aware clustering shipped, multi-threaded extraction shelved

Real production runs (TITAN Xp and P100, 16 cores) showed gap-filled-position
scoring extraction-bound by 6-7x (~1,600-2,000 pos/s vs. ~75,000 pos/s for
the bulk scan), with CPU usage never approaching 16 cores. Root cause:
`extract_features_batch`'s clustering only capped a shared-fetch cluster's
absolute span (5,000,000bp) — since gap-filled points are sparse by
construction, isolated points kept merging into multi-megabase clusters
that fetched far more genomic span than needed, and that fetch +
`np.cumsum` work is single-threaded, so extra cores didn't help.

**Density-aware clustering** (`features._build_clusters`) fixes this for
free: a cluster now only extends to the next point when doing so adds no
wasted span (i.e. the two points' `max_dist`-wide windows would already
overlap), with the absolute-span cap remaining only as a backstop. Dense
position sets are unaffected; only sparse ones change behavior. Ships
unconditionally, since it only changes how a single reader groups its own
fetches.

**Multi-threaded extraction** (extra, independently-opened reader pairs
processing clusters concurrently) was also built, measured at a real 2.2x
speedup on a synthetic sparse benchmark — then found on real production
data to multiply an unbounded memory cost, since each independently-opened
`pybigtools` reader accumulates its own per-chromosome index cache with no
eviction (confirmed against `bigtools`' Rust source): a single reader
grows ~120MB over a full genome sweep, `--cores 16` grows ~1.6GB, a
consistent ~20% whole-pipeline RSS regression. Every mitigation tried
(capping reader count, resetting per chromosome, restricting to the one
call site it helps) traded away most of the speedup without fixing the
memory cost — the real fix needs eviction added to `bigtools` itself.
**Not shipped**: removed from this release and preserved, unmodified, on
the `multithreaded-extraction-dev` branch; feature extraction here is
always single-threaded, one reader, regardless of `--cores`, but still
gets density-aware clustering. Real production re-validation (K562_groseq,
G2, `--cores 16`) confirmed this is a clean win: peak RSS within 0.2-1.1%
of the pre-clustering baseline (noise-level), wall-clock unaffected. See
`docs/PERF_LOG.md`'s 2026-08-10, 2026-08-14, and 2026-08-15 entries for the
full mitigation attempts.

## Peak calling: process parallelism and per-worker BLAS pinning

The peak-calling stage runs as one independent unit of work per broad
candidate peak, parallelizing across `--cores` worker processes (each
handling `--peak-calling-block-width` broad peaks at a time, for load
balancing across uneven peak sizes). Each worker pins itself to a single
BLAS thread on startup, since the per-peak p-value calculation only
involves tiny 5×5 matrices, far too small for BLAS's own multithreading to
help — leaving it unconstrained would oversubscribe real cores across many
worker processes for no benefit.

## The per-summit p-value: three exactness-preserving speedups

The per-summit p-value (a 5-dimensional multivariate-Laplace tail
probability, `stats.pmv_laplace`) was, before optimization, over 97% of
peak-calling time in one real production run. Three changes, each verified
to leave the statistical result unchanged (within R's own existing QMC
noise):

1. **Match R's actual CDF precision.** SciPy's `multivariate_normal.cdf`
   defaults to precision ~100-200x tighter than R's real
   `GenzBretz(maxpts=25000, abseps=1e-3)` (configurable via
   `--pmv-laplace-cdf-maxpts`/`--pmv-laplace-cdf-eps`, but these should
   only be loosened further than R's defaults if you explicitly want to
   trade fidelity for speed) — this alone was the single biggest win, since
   the extra precision was never needed.
2. **Stop recomputing identical setup work.** Each evaluation internally
   rebuilds a fixed QMC integration lattice hundreds of times per call with
   identical parameters; this is now cached.
3. **Adaptive sample count, like R, instead of a fixed floor.** SciPy's
   public API always starts its sampling budget at a floor sized for a
   "typical" hard case; R's algorithm starts small and grows only as
   needed. `pydreg` drives SciPy's own integration kernel with that same
   small-start, grow-as-needed schedule.

Combined: ~3s → ~17ms per call in isolated benchmarking. On a full real
production run (Jurkat PRO-seq, `--cores 16`, cupy backend), the "calling
peaks" step's wall time dropped 509.31s → 289.05s (**1.76x**), total
pipeline wall time 20m22.6s → 16m49.4s (**1.21x**). See `docs/PERF_LOG.md`'s
2026-07-14 and 2026-07-21 entries for the full precision/caching/adaptive-
sampling investigation.

### Fast approximate mode: `--pmv-laplace-tail-tol`

Everything above stays exact (identical, within R's own QMC noise).
`--pmv-laplace-tail-tol` (defaulted to `1e-6`, not `0.0`) deliberately
breaks that equivalence for more speed on top of an already-fast default;
pass `--pmv-laplace-exact` (or `--pmv-laplace-tail-tol 0.0`) for the old
exact-by-default behavior.

The remaining cost is a 290-point z-grid loop, each point needing its own
QMC box-probability evaluation. Uniformly subsampling that grid was tried
and **rejected**: a boundary-focused stress test (60 cases constructed so
R's exact result lands right at the FDR threshold, `prob = 0.05`) showed
even mild thinning collapsing `prob` toward 0 in every case — every
borderline call would flip from non-significant to significant. What
shipped instead is a **provably-bounded tail truncation**: each z-grid
point's contribution has a hard, data-independent upper bound
(`width × exp(-z)`, computable before ever looking at a specific peak), so
`--pmv-laplace-tail-tol` sets a real, honored worst-case error bound rather
than a heuristic — and because truncated terms can only be ≥ 0, dropping
them can only push `prob` toward *less* significant, never toward a false
positive. The boundary stress test showed no measurable effect at the
default `1e-6`.

Confirmed on a full real production run (Jurkat PRO-seq, `--cores 16`,
cupy backend): exact vs. fast-default gave identical peak counts (33350)
and near-identical dREG agreement (0.999317 vs. 0.999344 Jaccard), while
`pmv_laplace`'s own block-CPU time dropped 7735.58s → 4240.99s (**1.82x**)
and total pipeline wall time improved **1.21x** — this validation is why
the fast tolerance became the default. See `docs/PERF_LOG.md`'s 2026-08-19
and 2026-08-21 entries for the rejected grid-thinning attempt and the full
numbers.

## The random-forest peak splitter: numba, not scikit-learn

The small random-forest model deciding whether adjacent local maxima
should merge or split (~500 trees) is evaluated via a hand-written
numba-compiled tree traversal, not
`sklearn.ensemble.RandomForestRegressor`. `pydreg`'s actual usage pattern
is many *tiny* predict calls (often 1-20 rows, since the decision is made
incrementally as regions merge), and scikit-learn's forest dispatches one
parallel task per tree plus full estimator-validation machinery on every
call — ~10-25ms fixed overhead regardless of work done, versus numba's
microseconds for the same input. (At batch sizes in the thousands, that
overhead amortizes and scikit-learn's own tree-parallelism wins — just not
at the sizes this pipeline uses.)

## The CPU ("numpy") backend: fusing the RBF kernel's elementwise step

`DREGModel.predict()` — the "numpy" backend, and the reference every other
backend's smoke test validates against — computes, per SV chunk: a GEMM, an
elementwise squared-distance-and-`exp` step, then a second GEMM/GEMV.
Measured directly that the elementwise step, not either GEMM, was the
bottleneck: ~70% of wall time on a real 605,187 SV model, running at
effectively one core since plain NumPy ufuncs don't parallelize.
`models._rbf_accumulate`, a `numba.njit(parallel=True)` kernel `prange`'d
over queries, fuses that step with the final reduction — the CPU-tier
counterpart to the cupy/mlx fusion above. Verified bit-close to the old
separate-NumPy-calls formula (~1.65e-13 max abs diff, the same
numba-vs-NumPy `exp` ULP difference already accepted elsewhere).

Net effect on a 10-core Apple Silicon machine: `predict()` dropped 21.1s →
7.0s (~3x) — undersold by Accelerate BLAS's own parallelism capping out at
~1.6x there regardless of thread count. Confirmed on real 32-core x86/Linux
(OpenBLAS) that this ceiling is Accelerate-specific, not fundamental: the
same GEMM scales 7.4x there and the fused kernel 17.6x, with no BLAS/numba
contention at any thread-count combination tested. This surfaced a real
gap — `pipeline.run()` never told BLAS about `--cores` — fixed via
`threadpoolctl.threadpool_limits(limits=cores)` (see "One `--cores` knob"
above). See `docs/PERF_LOG.md`'s 2026-08-13 entries for the full profiling
breakdown.

## Loading the pretrained SVR: in-memory, not through a temp file

`DREGModel.from_pretrained()` used to take 7-9s per call even on a warm,
no-network Hugging Face cache. Profiling found decompression and the temp
file write were both fast (well under 2s combined); the actual cost was
`safe_open(tmp_path).get_tensor("support_vectors")` at **3.65s** for one
605,187×360 float64 tensor, against `safetensors.numpy.load()`
materializing *all four* model tensors from the same in-memory bytes in
**0.52s total** — `safe_open`'s mmap-based per-tensor path is dramatically
slower than a plain `load()` once a tensor gets large, and pydreg never
needed `safe_open`'s one advantage (lazy, selective access) since every
call site reads every tensor anyway.

`pydreg._safetensors_io.open_safetensors` now decompresses straight to
bytes (no temp file) and calls `safetensors.numpy.load(bytes)`, parsing the
file's `__metadata__` header by hand (a couple of stdlib calls against
safetensors' documented, stable 8-byte-length-prefix format) since that API
doesn't expose it. Net effect: **~7-9s → ~2-2.7s** per SVR load, verified
bit-identical against the old path. The RF peak-splitter model is
unaffected in practice — its tensors are tiny.

## Writing outputs in parallel

`pipeline._write_outputs` writes up to 7 files (informative-position
bigWig, raw/full/score/prob peak bed.gz, score/prob peak bigWig), which
used to run fully sequentially despite having no dependencies on each
other. The informative-position `.bed.gz` was dropped entirely — it
duplicated the `.bw` purely for debugging, was never read back anywhere,
and was by far the largest, slowest file this step wrote.

The remaining writes are dispatched across two pools, because the two
writers behave oppositely under threading. `pysam.tabix_index`'s bgzip
compression releases the GIL (~5.6x speedup threading 8 concurrent calls on
a 10-core machine), so `.bed.gz` writes go on a
`ThreadPoolExecutor(max_workers=min(cores, n_bedgz_files))`. `pybigtools`'
bigWig writer does **not** — threading 4 concurrent `write_bigwig` calls
measured 4x *slower* than serial (20.7s vs 5.2s), real Rust-binding lock
contention. BigWig writes instead go on a
`ProcessPoolExecutor(max_workers=min(cores, n_bigwig_files))`, which
correctly parallelizes (2.5s for the same 4 files) — pickling cost across
the process boundary is cheap here since `pydreg.io` only imports
numpy/pybigtools (pool startup ~0.2s) and bigWig outputs are all small now
that the large infp `.bed.gz` is gone.

## Reproducing these results

`scripts/bench_backends.py` benchmarks the SVR backends against each other
on your own hardware. `scripts/bench_numpy_backend_threading.py` diagnoses
the CPU backend's GEMM-vs-numba-kernel threading behavior specifically
(BLAS thread sweeps, numba thread sweeps, and every combination against
full `predict()`) — see "The CPU ('numpy') backend" above for why that
combination, not just each piece alone, is the real question on any given
machine. `docs/PERF_LOG.md` has the full history for every change
summarized above, including exact numbers, dead ends, and source-level
evidence behind each root cause.

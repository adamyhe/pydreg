# Performance design choices

`pydreg`'s guiding rule for every performance change is that it must not
change the pipeline's output: same scores, same peaks, same
faithfully-replicated R quirks (see `docs/PLANNING.md`) — verified against
the existing test suite and, for the peak-calling changes below, directly
against real dREG output (0.999728 Jaccard index on test data; see
`docs/METHODS.md`). This document explains the resulting design choices at a
level meant for anyone using or extending `pydreg`, not just the people who
made them. The full chronological research log — every benchmark, every
dead end, every number — lives in `docs/PERF_LOG.md`; this document is the
distilled "why it's built this way" version.

## Scoring: three backends, why NumPy (not scikit-learn) is the CPU default, and why the GPU tier is `cupy` (not `cuML`)

Evaluating the pretrained SVR (605,187 support vectors) against every
informative position is dominated by one computation: an RBF kernel matrix
between the query positions and every support vector. `pydreg` offers three
backends for this (`--backend {auto,cupy,sklearn,numpy}`):

- **NumPy** (default on any machine without a usable GPU): computes the
  kernel matrix as one chunked matrix multiplication
  (`X_scaled @ sv_block.T`), dispatching to whatever BLAS library NumPy is
  linked against.
- **scikit-learn**: wraps the same pretrained weights into an
  `sklearn.svm.SVR` (via `to_sklearn_svr()`) and predicts through libsvm.
- **cupy** (`pydreg[gpu]`, Linux + NVIDIA only, auto-selected whenever a
  usable CUDA device is present): the exact same chunked-matmul formula as
  the NumPy tier, run directly on a CuPy device array.

scikit-learn is available (`--backend sklearn`) but is **never
auto-selected on CPU** — it's measured at ~14-15x slower than the NumPy tier
for this workload, despite computing identical math (both agree to ~1e-10).
This isn't a threading gap (forcing single-threaded BLAS doesn't change the
NumPy tier's wall-clock time at all) — it's that libsvm's prediction path
evaluates the kernel one query-support-vector pair at a time (with a
heap allocation per pair), while `DREGModel.predict`'s chunked matmul
computes the entire kernel matrix in one BLAS call. Different computational
shape, not a tuning difference — so it isn't fixable by parallelizing
libsvm's loop, and there's no reason to expect Intel's oneDAL-accelerated
scikit-learn fork (`scikit-learn-intelex`) to help either, beyond the fact
that it doesn't ship any macOS/ARM wheels at all and would need real
engineering to even engage on a model that was never `.fit()` through it (see
`docs/PERF_LOG.md` for the full investigation of both).

### Why `cupy`, not `cuML`

The GPU tier used to be `cuml.svm.SVR` (built via `from_sklearn()`), not
`cupy`. It was dropped after real-hardware testing found a serious, confirmed
problem: RAPIDS/cuML dropped support for Pascal GPUs (compute capability
< 7.0) in its 24.02 release, and running a Pascal-incompatible cuML build on
such a GPU doesn't raise an error — per RAPIDS's own deprecation notice, it
"will either fail or return invalid results." Confirmed end-to-end on real
production data: cuml 26.06.00's `SVR.from_sklearn()`-built model diverged
from the NumPy reference by ~0.05 on an NVIDIA TITAN X (Pascal, compute
capability 6.1), while the *exact same bigWig inputs*, run on an A100
(compute capability 8.0), produced no divergence (Jaccard > 0.999 vs. real
dREG). `from_sklearn` itself only shipped in cuML 25.02, a full year after
Pascal support was dropped in 24.02 — there was no cuML release that
supported both, so pinning an older cuML was never a workaround. Pascal's
own crippled double-precision throughput (this SVR is inherently float64,
and cuML offered no float32 override — see below) made GPU acceleration a
poor fit there even setting compatibility aside.

`pydreg.backend._build_cupy_predict_fn` sidesteps all of this by not routing
through `cuml.svm` (or any third-party SVM library) at all — it's a
near-verbatim port of `DREGModel.predict`'s chunked RBF dual-sum formula
(same expanded squared-distance trick, same chunking over support vectors),
just evaluated on a CuPy device array instead of a NumPy host array. Two
things follow directly from that being *the same formula* rather than a
separate implementation:

- **No cross-library conversion risk.** The old cuml tier (and the
  still-present sklearn tier) depend on `to_sklearn_svr()`'s
  private-attribute round trip and then on an independent kernel-evaluation
  codebase (cuML's or libsvm's own C++) agreeing with `DREGModel.predict` —
  exactly the class of bug the Pascal investigation chased down. The cupy
  tier has nothing to independently agree with; it *is* the reference
  formula, just relocated to the GPU.
- **It isn't limited to compute capability ≥7.0.** CuPy's own array
  primitives (elementwise ops, matmul via cuBLAS) support compute
  capability ≥3.0 — RAPIDS/cuML's Pascal drop was a policy decision about
  its own compiled kernels, not a CUDA-wide one. Confirmed correct on the
  exact TITAN X that broke the cuml tier, and now faster than cuml ever was
  there (and on an A100) after this session's fusion/batching/float32 work
  (see below).

Getting the old cuml tier to float32 (to close the speed gap a different
way) was investigated and found infeasible to do safely: `cuml.svm.SVR.
from_sklearn()` hardcoded `dtype=float64` with no override, and a
workaround would have meant bypassing it to manually replicate its private
attribute-setting logic — risky, version-fragile, and unverifiable without
real GPU hardware. `cupy` doesn't have this problem since it's pydreg's own
code with full control over every array's dtype.

`pydreg.backend.detect_backend()` now picks `cupy` whenever `_cuda_runtime_
available()` finds a usable CUDA device — no compute-capability gate is
needed, since `cupy` has no floor to speak of. `_wrap_sklearn_like`'s
first-batch smoke test (comparing the first batch's predictions against the
NumPy reference) remains in place regardless, as the last line of defense
against *any* backend conversion issue, hardware-related or not — it's what
caught the real cuml divergence above, and a real bug in `cupy`'s own
fusion code during this tier's development (see below).

### Speeding it up: kernel fusion, then batch size

Two independent levers, in the order they're worth pulling:

1. **Fuse the elementwise glue between the two GEMMs.** The two matmuls
   (`X @ SV.T` and `K @ coefs`) are already cuBLAS calls — near-optimal
   without touching precision. The formula between them
   (`exp(-gamma * (sq_x + sq_sv - 2*cross))`) was originally ~5 separate
   elementwise kernel launches, each reading/writing a full
   `(query_chunk, sv_chunk)` array to GPU global memory — pure
   memory-bandwidth overhead on what's fundamentally a memory-bound step
   (same reason the NumPy tier itself is memory-bandwidth-bound, not
   compute-bound). Fusing that whole chain into one kernel that reads its
   inputs once and writes `K` once cuts that traffic roughly 5x — same
   formula, same precision, just far less memory round-tripping.
   `cupy.fuse()` was tried first and produced a real, confirmed ~3.5e-4
   divergence on actual GPU hardware (caught by `_wrap_sklearn_like`'s own
   smoke test — a bug of this tier's own making, not cuML's, but caught by
   the exact same mechanism). Switched to `cupy.ElementwiseKernel` instead
   — CuPy's older, more battle-tested mechanism for this pattern, with
   every argument's dtype declared explicitly and no shape-based
   tracing/caching to get wrong; see `docs/PERF_LOG.md`'s 2026-07-15 entry
   for the full root-cause investigation. It also drops one live
   `(query_chunk, sv_chunk)`-shaped buffer entirely (the old separate
   `sqdist` intermediate no longer exists), which is why `sv_chunk`'s
   default could grow without exceeding the pre-fusion tier's peak memory.
2. **Grow the batch size** (`--query-chunk` for the outer per-call size,
   `--cupy-sv-chunk`/`build_scorer`'s `cupy_sv_chunk` for the inner
   per-support-vector-chunk size inside `_build_cupy_predict_fn`). Unlike
   the old cuml tier (which tiled the kernel matrix internally in C++
   without ever materializing the whole thing), this tier's own Python
   code materializes
   the `(query_chunk, sv_chunk)` intermediate directly — so for a *fixed*
   GPU memory budget `B`, the total number of kernel-launch iterations is
   `total_queries * n_sv * 8 bytes / B`, independent of how `B` is split
   between the two chunk sizes. Growing `B` (either knob) is what reduces
   iteration count and better amortizes per-launch overhead; rebalancing
   the same `B` between the two dimensions doesn't. Real per-GPU memory
   headroom wasn't known while writing this, which is why both are exposed
   as tunables rather than hardcoded — worth sweeping a few `--cupy-sv-chunk`
   values on the actual target GPU and picking the fastest that doesn't OOM.

A further lever, since implemented: **float32 instead of float64** for the
two GEMMs and the fused RBF kernel (`y_scaled` still accumulates in
float64, cheap insurance against cross-chunk summation error). This
changes actual arithmetic precision rather than just scheduling, so it
needed its own justification, not just "the smoke test tolerance is
generous" — see below. It matters most on exactly the hardware that
motivated this whole tier: consumer Pascal GPUs (e.g. the TITAN X) have
crippled float64 throughput (~1:32 vs float32), so this is a large win
there specifically, separate from and additive to the fusion/batching
levers above.

That said, the risk here is smaller than "changes arithmetic precision"
usually implies, once you know where these model weights actually came
from. The current pretrained dREG models are trained via **Rgtsvm**
(dREG's GPU-accelerated SVM tool; `e1071` now exists purely as an
S3-compatibility layer around it, not as the thing that actually fits the
model). Traced Rgtsvm/GTSVM's actual C++/CUDA source
(`github.com/Danko-Lab/Rgtsvm`) to check for exactly this: does it use
float32 internally despite R passing `double`s across the API boundary?
It does, unconditionally, with no build-time opt-out ever exercised:

- `gtsvmpredict_epsregression_C` (`Rgtsvm.cpp:398-401`) narrows
  `gamma`/`coef0`/`degree`/`cost` straight from `double*` to a local
  `float`, with no double-precision code path at all for these.
- The support-vector matrix itself is stored internally as
  `SparseVector = std::vector<std::pair<unsigned int, float>>`
  (`svm.hpp:280`) — `InitializeDense`/`InitializeSparse` convert the
  incoming `GTSVM_TYPE_DOUBLE`-tagged R data down into this float-based
  representation on the way in; the `DOUBLE` tag just describes the input
  buffer's element type for reading purposes.
- The SVM optimizer's own internal type, `CUDA_FLOAT_DOUBLE`
  (`cuda_helpers.hpp:40-44`), is `float` unless the `CUDA_USE_DOUBLE`
  macro is defined at compile time — checked `configure.ac` end-to-end
  and that macro is never defined anywhere in the actual build.

So the alphas/support-vectors/rho this project exports to safetensors as
float64 were themselves *produced* by a training process with no
double-precision arithmetic anywhere internally — their real accuracy
ceiling was already float32, before pydreg's float64 storage ever enters
the picture. Running cupy-tier **inference** at float32 wouldn't trade away
precision the model actually has; there isn't more precision there to
trade away. If anything it would move pydreg's GPU behavior *closer* to
how real GPU-accelerated dREG behaved historically, not further from it.

This doesn't extend to the CPU tiers, though, and shouldn't be read as "so
just make everything float32." Checked libsvm's actual source too
(`cjlin1/libsvm`, what both `e1071` and `sklearn.svm.SVR` bind to for CPU
prediction): `svm_node.value` is `double`, `Kernel::k_function` and
`svm_predict_values` operate in `double` throughout. `Qfloat` (`typedef
float`) exists in libsvm but only for the *training*-time kernel cache
(`Cache`/`SVC_Q`/`SVR_Q`) — never in the predict path. So dREG's CPU
inference mode (`e1071`) has always been genuinely double-precision, and
pydreg's own NumPy/scikit-learn tiers already match that exactly. Down
casting those to float32 would be a real fidelity regression relative to
the actual historical CPU reference, for a speed motivation (crippled FP64
throughput) that's GPU-specific and doesn't apply to CPUs at all — not
recommended.

**Confirmed on real hardware, and the smoke test's tolerance adjusted
accordingly.** The float32 switch tripped `_wrap_sklearn_like`'s own
smoke test: a real `max_abs_diff` of ~2.3e-4 against the float64 NumPy
reference, with sklearn (CPU libsvm) independently agreeing with that same
reference to ~5.5e-11 on the same sample — the same cross-check pattern
from the original Pascal investigation, this time confirming the
divergence is cupy's own (expected) float32 arithmetic, not a conversion
bug. Root cause: the expanded-form squared-distance formula
(`sq_x + sq_sv - 2*cross`) is a classic catastrophic-cancellation setup for
nearby feature vectors, and while that's negligible in float64 (~15-16
significant digits to lose a few of), it consumes a much larger fraction
of float32's ~7 significant digits. The error is already baked into
`cross`'s value by the time it comes back from the float32 GEMM — doing
the subsequent subtract/exp step at higher precision doesn't recover it,
so this isn't cheaply fixable without a fundamentally different
mixed-precision GEMM technique. `_wrap_sklearn_like` now takes an
`atol`/`rtol` override, and `build_scorer()` passes a `CUPY_SMOKE_TEST_ATOL
= 5e-4` for the `cupy` tier specifically (`sklearn` keeps the default
`1e-4`, since it's genuinely float64) — loosened deliberately, with a real
measured number plus margin behind it, not a blanket weakening of the
safety net.

**Why this wasn't done for the old cuml tier.** `cuml.svm.SVR.from_sklearn()`
took no dtype parameter (`cuml/internals/base.pyx`'s `from_sklearn(cls,
model)` signature had none, and `SVMBase.__init__` didn't expose `dtype`
as a constructor option either — it was only ever set internally during
`.fit()`/`cpu_to_gpu()`, hardcoded to `np.float64` when converting from a
CPU model). Getting a genuinely float32 cuML SVR would have meant bypassing
`from_sklearn()` and manually setting `dtype`/`support_vectors_`/
`dual_coef_`/etc. directly, replicating cuML's own private
`_attrs_from_cpu` logic. That's a specific, real risk, not just "more
private-API surface": `_get_svm_model()` picked its C++ template
(`SvmModel<float>` vs `SvmModel<double>`) from the `dtype` flag, then
raw-pointer-reinterpreted the underlying array's memory accordingly. If
that flag and the array's actual dtype ever disagreed, it wouldn't
silently lose precision — it would read the wrong bytes entirely (garbage
or a crash), with no GPU available here to catch it. This risk (on top of
the Pascal incompatibility) is a big part of why `cuml` was dropped
entirely rather than kept around as a float32-patched second GPU tier.

## Batching

Each backend gets its own default query-chunk size
(`pydreg.backend.DEFAULT_QUERY_CHUNK`, overridable via `--query-chunk`),
sized for that backend's actual bottleneck:

- **NumPy**: bounded so the transient `(query_chunk, sv_chunk)`-shaped
  intermediate arrays stay a manageable size in memory — this tier is
  memory-bandwidth-bound, not compute-bound.
- **scikit-learn**: libsvm's predict loop isn't memory-bound the same way,
  so its chunk size is mainly for streaming/checkpointing, not correctness.
- **cupy**: this tier's own Python code materializes the
  `(query_chunk, sv_chunk)` kernel-matrix intermediate directly, the same
  as the NumPy tier does on the CPU (unlike the old cuml tier, which tiled
  the kernel matrix internally in C++ without ever materializing the whole
  thing) — so it reuses NumPy's conservative default. Unvalidated on
  real GPU memory; likely worth tuning up once tested on real hardware.

## Overlapping feature extraction with scoring

`pipeline._score_positions` (used by every backend, not just GPU ones)
alternates two very different kinds of work per chunk: CPU-bound feature
extraction (bigWig I/O + multi-scale binning, see `pydreg.features`) and
the backend's own `scorer.predict()` call. These used to run strictly
sequentially — extract, then predict, then extract the next chunk, and so
on — which is invisible when the GPU kernel itself is the bottleneck, but
becomes a real cost once it isn't: real-hardware testing on a TITAN X
showed GPU utilization cycling 0–90% between chunks once the `cupy` tier's
float32 downcast (see below) cut its own compute time by an order of
magnitude — the GPU sitting idle while the CPU extracts the next chunk's
features, a cost that was always there but had been hidden behind a much
slower GPU kernel until that point (the same effect showed up on an A100
running `cuml`, independent of the float32 work — this bottleneck is
backend-agnostic).

`_score_positions` now runs a one-chunk-ahead prefetch: while the current
chunk's `scorer.predict()` blocks on the GPU, a single background thread
extracts the *next* chunk's features concurrently. This overlaps the two
steps instead of eliminating either one — same calls, same inputs, same
order, purely a scheduling change. It's safe specifically because exactly
one thread ever touches the bigWig readers at a time: the main thread
never reads a bigWig while a background extraction is in flight, and a
`ThreadPoolExecutor(max_workers=1)` guarantees only one extraction call is
ever in progress regardless of how far ahead a chunk gets submitted. The
overlap itself depends on `scorer.predict()` releasing the GIL while
blocked on the GPU (true of CuPy's device-sync calls) — on the
NumPy/scikit-learn CPU backends this can't hurt correctness, it just may
not overlap as usefully since there's no GPU wait to hide behind.

### Real measurements, and why extraction parallelism isn't implemented (yet)

`_score_positions` logs accumulated `extract_seconds`/`predict_seconds`
once per call (see its docstring), and real runs on both a TITAN Xp and an
A100 confirmed the prefetch is working, with a clear pattern across the
three call sites:

| step | TITAN Xp (extract / predict) | A100 (extract / predict) |
|---|---|---|
| informative positions (bulk scan) | 196.90s / 733.57s | 193.81s / 338.67s |
| gap-filled positions | 69.33s / 14.46s (reversed) | 58.15s / 6.65s (reversed) |
| 10bp-densified positions | 154.40s / 534.75s | 153.29s / 245.90s |
| **aggregate ratio (predict:extract)** | **3.05x** | **1.46x** |

Two of the three steps are predict-dominated (extraction mostly hides
behind the GPU wait, as intended) on both cards. The gap-filled-positions
step is a real, isolated exception — extraction dominates there, plausibly
because gap-filled points are scattered into sparse gaps by construction
(`peaks.find_gap_infp`), which likely defeats `_extract_features_cluster`'s
shared-fetch batching (clustering pays off for nearby points, not isolated
ones). It's a small absolute contributor either way (~5-10% of total
scoring time on these runs).

The more interesting signal is the *aggregate ratio dropping from 3.05x to
1.46x* between the two cards — extraction time is essentially unchanged
(CPU-bound, GPU-independent), while predict time shrank substantially on
the faster card, so the same fixed CPU cost is a growing fraction of the
total. Extrapolate that trend (a faster GPU still, or `cupy`+float32
becoming the default path on every card) and extraction could eventually
stop being fully hideable. Estimated full-fix upside on the numbers
above: only ~6-10% more wall time saved, since two of three steps are
already well-overlapped — not enough on its own to justify the change
today, but the trend is worth tracking.

**Investigated whether extraction could be safely parallelized across
multiple background workers, in case that trend continues.** Read
pybigtools' actual Rust source (`jackh726/bigtools`,
`pybigtools/src/reader.rs`): its `BBIReader.values()`/`.intervals()`/
`.zoom_intervals()` all take `&mut self`, and since `BBIReader` isn't
marked `unsendable` in its `#[pyclass]` attribute, PyO3 wraps it with a
runtime borrow-check cell enforcing Rust's aliasing rules independent of
the GIL — concurrently calling a read method from two threads on the
*same* `BBIReader` object raises `PyBorrowMutError` (a safe, loud failure,
not silent corruption, but still not usable concurrently). The underlying
reader types are generic over `CachedBBIFileRead<ReopenableFile>`,
though — built specifically around a `Reopen` trait meant for independent
handles onto the same file. So the safe pattern, if this is ever
implemented, is **one independently-opened `BBIReader` per worker
thread** (`pydreg.io.open_bigwig()` is a thin wrapper around
`pybigtools.open()`, trivial to call once per worker) rather than sharing
`pipeline.run()`'s single `bw_plus`/`bw_minus` pair across threads. Not
implemented — the current numbers don't justify the added complexity —
but documented here specifically so this doesn't need re-deriving if the
ratio ever tips further.

## Peak calling: parallelism and per-worker BLAS pinning

The final peak-calling stage runs as one independent unit of work per broad
candidate peak, so it parallelizes trivially across `--peak-calling-cores`
worker processes (each handling `--peak-calling-block-width` broad peaks at
a time, tuned for load balancing across uneven peak sizes). Each worker is
pinned to a single BLAS thread on startup — the linear algebra inside the
per-peak p-value calculation (below) involves only tiny (5×5) matrices, far
too small to benefit from BLAS's own multithreading, so leaving it
unconstrained would oversubscribe real cores across many worker processes
for no benefit.

## The per-summit p-value: from the dominant cost to a minor one

The per-summit p-value (a 5-dimensional multivariate-Laplace tail
probability, `stats.pmv_laplace`) was, before this round of optimization,
the overwhelming majority of peak-calling time — over 97% of it in one real
production run. Three changes, each independently verified to leave the
statistical result unchanged (within the ordinary run-to-run noise this
calculation already has, inherited from R):

1. **Match R's actual precision settings.** The underlying integral is
   evaluated via SciPy's `multivariate_normal.cdf`, which defaults to a
   precision ~100-200x tighter than what R's own reference implementation
   (`mvtnorm::pmvnorm`) actually uses. Matching R's real
   `GenzBretz(maxpts=25000, abseps=1e-3)` defaults (configurable via
   `--pmv-laplace-cdf-maxpts`/`--pmv-laplace-cdf-eps`, but these should
   only ever be *loosened* from R's defaults if you explicitly want to
   trade fidelity for speed) was both more faithful to R and, since it was
   needless extra precision, the single biggest win available.
2. **Stop recomputing identical setup work.** Each p-value evaluation
   internally repeats a fixed setup step (constructing a quasi-Monte-Carlo
   integration lattice) hundreds of times per call with the exact same
   parameters — this is cached transparently.
3. **Use an adaptive sample count, like R does, instead of a fixed floor.**
   SciPy's public API for this integral always starts its sampling budget
   at a fixed floor sized for a "typical" hard case, regardless of how easy
   the actual box being integrated is. R's own algorithm has no such
   floor — it starts small and grows only as needed, stopping the moment
   its precision target is met. `pydreg` now drives SciPy's own (otherwise
   unmodified) integration kernel with that same small-start, grow-as-needed
   schedule, which is both a large speedup and, if anything, a closer match
   to R's actual behavior than the fixed-floor approach it replaced.

Combined, these took a representative case from ~3 seconds to ~17
milliseconds per call in isolated benchmarking — real production hardware
should be judged by its own before/after numbers (the gains observed in
production, while still substantial, are smaller than the gains measured on
faster/uncontended dev hardware, since a lot of this is raw per-core
throughput sensitive work).

## The random-forest peak splitter: numba, not scikit-learn

The small random-forest model used to decide whether adjacent local maxima
should be merged or split (~500 trees) is evaluated via a hand-written
numba-compiled tree traversal, not `sklearn.ensemble.RandomForestRegressor`.
This isn't a close call: `pydreg`'s actual usage pattern is many *tiny*
predict calls (often just 1-20 rows at a time, since the peak-splitting
decision is made incrementally as adjacent regions get merged), and
scikit-learn's random forest dispatches one parallel task per tree plus its
full estimator-validation machinery on every single call — fixed overhead
of ~10-25 milliseconds *regardless of how much work is actually being done*,
which numba's directly-compiled traversal does in **microseconds** for the
same tiny inputs. (At much larger batch sizes, in the thousands of rows,
that fixed overhead amortizes away and scikit-learn's own parallelism
across trees actually wins — just not at the batch sizes this pipeline
ever actually uses.)

## Reproducing these results

`scripts/bench_backends.py` benchmarks the SVR backends against each other
directly on your own hardware. `docs/PERF_LOG.md` has the full history for
every change summarized above, including the exact numbers, the dead ends
that didn't pan out, and the source-level evidence behind each root cause.

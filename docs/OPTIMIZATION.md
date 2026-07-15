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

## Scoring: three backends, and why NumPy (not scikit-learn) is the CPU default

Evaluating the pretrained SVR (605,187 support vectors) against every
informative position is dominated by one computation: an RBF kernel matrix
between the query positions and every support vector. `pydreg` offers three
backends for this (`--backend {auto,cuml,sklearn,numpy}`):

- **NumPy** (default on any machine without a usable GPU): computes the
  kernel matrix as one chunked matrix multiplication
  (`X_scaled @ sv_block.T`), dispatching to whatever BLAS library NumPy is
  linked against.
- **scikit-learn**: wraps the same pretrained weights into an
  `sklearn.svm.SVR` (via `to_sklearn_svr()`) and predicts through libsvm.
- **cuML** (`pydreg[gpu]`, Linux + NVIDIA only): the same weights loaded
  into `cuml.svm.SVR.from_sklearn()`, running the kernel matrix on GPU.

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

## cuML's real hardware requirement: compute capability ≥7.0

The cuML tier is now validated on real GPU hardware, and that validation
caught a real, confirmed constraint: RAPIDS/cuML dropped support for Pascal
GPUs (compute capability < 7.0) in its 24.02 release. Running a
Pascal-incompatible cuML build on such a GPU doesn't raise an error — per
RAPIDS's own deprecation notice, it "will either fail or return invalid
results." Confirmed end-to-end on real production data: cuml 26.06.00's
`SVR.from_sklearn()`-built model diverged from the NumPy reference by ~0.05
on an NVIDIA TITAN X (Pascal, compute capability 6.1), while the *exact same
bigWig inputs*, run on an A100 (compute capability 8.0), produced no
divergence (Jaccard > 0.999 vs. real dREG). Also worth noting: `from_sklearn`
itself only shipped in cuML 25.02, a full year after Pascal support was
dropped in 24.02 — there is no cuML release that supports both, so pinning
an older cuML isn't a workaround for Pascal-class hardware. Pascal's own
crippled double-precision throughput (this SVR is inherently float64) makes
GPU acceleration a poor fit there even setting compatibility aside.

`pydreg.backend.detect_backend()` and `build_scorer()`'s explicit
`--backend cuml` path both check the GPU's compute capability up front
(`MIN_CUDA_COMPUTE_CAPABILITY = 70`) and refuse cuML below that threshold —
auto mode silently falls back to NumPy, an explicit request raises
`BackendUnavailable` with the specific reason, rather than surfacing as a
confusing mid-pipeline smoke-test failure. That smoke test
(`_wrap_sklearn_like`, comparing the first batch's predictions against the
NumPy reference) remains in place regardless, as the last line of defense
against *any* backend conversion issue, hardware-related or not.

## Batching

Each backend gets its own default query-chunk size
(`pydreg.backend.DEFAULT_QUERY_CHUNK`, overridable via `--query-chunk`/
`--cuml-query-chunk`), sized for that backend's actual bottleneck:

- **NumPy**: bounded so the transient `(query_chunk, sv_chunk)`-shaped
  intermediate arrays stay a manageable size in memory — this tier is
  memory-bandwidth-bound, not compute-bound.
- **scikit-learn**: libsvm's predict loop isn't memory-bound the same way,
  so its chunk size is mainly for streaming/checkpointing, not correctness.
- **cuML**: chunked only to bound host→GPU transfer size; the full support
  vector matrix is uploaded once, not chunked.

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

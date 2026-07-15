# pydreg performance log

Experiment/implementation log for performance work on the `pydreg` pipeline,
kept so findings and results survive across sessions (unlike the ephemeral
per-session plan file). Append a new dated entry per change; do not edit
past entries except to fix factual errors.

**Ground rule for every entry below**: performance-only changes. Every fix
must produce bit-identical output to the pre-change implementation (same
scores, same peaks, same faithfully-replicated R quirks documented in
`docs/PLANNING.md`) — verified via the existing test suite plus a full CLI
diff, not just "looks right."

This is a comprehensive, chronological research log (every benchmark, every
dead end, every number), not user-facing documentation. For a plain-language
summary of the resulting design choices, see `docs/OPTIMIZATION.md` instead.

## 2026-07-09 — initial vectorization/JIT audit

Full audit of `src/pydreg/` (`io.py`, `infp.py`, `features.py`, `models.py`,
`backend.py`, `pipeline.py`, `rfsplit.py`, `stats.py`, `smoothing.py`,
`peaks.py`) for vectorization/batching/JIT opportunities, via three parallel
Explore agents plus direct reads. Six concrete opportunities found, ranked
by expected impact (informative positions run into the millions genome-wide;
broad peaks/summits run into the thousands; everything else runs once per
invocation):

**Already optimal, no action taken:**
- `DREGModel.predict` (the SVR): fully vectorized BLAS matmul, chunked only
  over support vectors for memory.
- `backend.py`/`pipeline.py`: no `.apply()`/`.iterrows()`/per-row loops found;
  already array-level.
- `smoothing.py`: per-segment/`ends==1` loops degenerate to 0-1 iterations at
  every real call site.

**Opportunities identified (implementation status tracked per-item below as
each is completed):**

1. **`features.py`/`io.py`** — `extract_features_batch` loops calling
   `extract_features` once per center; each call independently re-fetches a
   `2*max_dist+1`-wide (~200,001bp) bigWig window per strand even though
   nearby informative positions (as close as 10-50bp apart) overlap that
   fetch by >99.9%. Highest impact: runs once per informative position,
   potentially millions of times genome-wide. Fix: shared per-batch raw
   fetch + vectorized zoom-binning across positions.
2. **`infp.py`** — the 9-phase loop issues up to 27 separate
   `bw.values()` calls per chromosome per strand-pass; all window/step sizes
   are multiples of 50bp, so one fine-grid fetch per strand can derive every
   phase's windowed sums via NumPy striding, with zero extra bigWig calls.
3. **`models.py`** — `DREGPeakSplitForest.predict` loops 500 trees in
   Python, called repeatedly (once per `_split_peak` while-iteration) across
   thousands of broad peaks, each call on tiny `X.shape[0]` — Python/NumPy
   dispatch overhead dominates. Fix: JIT-compile the traversal with numba
   (`@njit(cache=True)`), mirroring `src/regTree.c`'s `predictRegTree()`
   directly, rather than NumPy-vectorizing across the tree axis.
4. **`stats.py`** — `pmv_laplace` rebuilds a frozen `multivariate_normal`
   object and a ~290-point z-grid from scratch on every call; both are
   invariant across an entire `call_peaks` run (fixed `cor_mat`, fixed
   `d=5`). Hoist/cache once per run instead of once per peak summit.
5. **`peaks.py`** — `find_gap_infp` ends with a row-by-row
   `gap_bed.apply(lambda r: ..., axis=1)` existence check; replace with a
   vectorized `isin`.
6. **`peaks.py`** — `get_broadpeak_summary`/`_pred_dense_infp` loop
   `iterrows()` per broad peak for searchsorted+reduction; lower priority
   (peak counts are thousands, not millions) but batchable across all peaks
   on a chromosome at once.

Implementation order: 4, 5 (lowest risk) → 2 → 3 → 6 → 1 (highest risk/impact,
done last with the most validation). See entries below for what actually
shipped and measured timing.

## 2026-07-09 — items 4, 5, 2, 3 shipped

- **Item 4** (`stats.py`, `pmv_laplace`): hoisted the frozen
  `multivariate_normal` object into a single-slot cache keyed on `cor_mat`'s
  identity, and the ~290-point z-grid into a lazily-computed module-level
  constant (it never depended on `x`/`cor_mat` in the first place — pure
  grid definition). Both now built once per process/run instead of once per
  peak summit. 21/21 tests pass.
- **Item 5** (`peaks.py`, `find_gap_infp`): replaced the row-by-row
  `gap_bed.apply(lambda r: (r["chrom"], r["start"]) in existing, axis=1)`
  with a vectorized `pd.MultiIndex.isin` check. 21/21 tests pass.
- **Item 2** (`infp.py`): replaced the 9-phase loop's up to 27 per-phase
  `bw.values()` calls per chromosome with one step-resolution (50bp) fetch
  per strand, deriving every phase's 100bp (OR) / 1000bp (AND) windowed sum
  via NumPy reshape+sum slicing (`_windowed_sums_from_fine`). Verified the
  bin-count formula matches `io.windowed_sum`'s exactly across a stress test
  of chrom sizes/phases, and verified the derived sums are **bit-for-bit
  identical** to direct per-phase `bw.values()` calls on real chr21 data
  (`_reference/dREG/example/K562.chr21.plus.bw`). Also confirmed
  `get_informative_positions`'s full output (182,605 rows on real chr21) is
  bit-identical before/after (compared inline against the pre-change logic,
  since `src/` isn't yet git-tracked so `git stash` couldn't provide a clean
  baseline — see note below for future changes). 21/21 tests pass.
- **Item 3** (`models.py`, `DREGPeakSplitForest.predict`): replaced the
  500-tree Python loop with a `numba.njit(cache=True)`-compiled function
  (`_forest_predict`) that loops trees x queries x node-depth directly in
  compiled code, mirroring `src/regTree.c`'s `predictRegTree()` literally.
  Added `numba` as a core dependency (`uv add numba` — installed cleanly on
  macOS/Python 3.12 venv, pulled in `llvmlite`). Verified bit-identical
  output against the old per-tree-loop implementation across 20 random
  trials of varying batch size. 21/21 tests pass.
  - **Investigated `numba.prange` and rejected it**: benchmarked
    `parallel=True` with `prange` over the tree axis and over the query axis
    against the plain serial `njit`, at call sizes matching the real
    workload (`predict()` is called from `rfsplit._split_peak`'s while-loop
    with `newdata` sized to the number of `ST==0` regions in one broad peak
    — realistically 1-5 rows). Result: `prange` over trees was 3-5x
    *slower* at every tested size (1/3/10 queries); `prange` over queries
    was 2-3x slower at 1-3 queries and only became faster (2x) at 10
    queries. Thread-launch/parallel-region overhead dominates because each
    tree traversal is tiny (~10-20 node visits) and per-call batches are
    small — this workload's real parallelism is at the "many small calls
    across thousands of broad peaks" level, not within one call. Kept the
    plain serial `njit`; no code change from the investigation.

Remaining: items 6 and 1.

**Process note**: `src/`, `docs/`, `tests/`, `pyproject.toml` etc. are not
yet git-tracked in this repo (still untracked/new). `git stash` silently
no-ops on untracked files, so it can't be used to get a clean "before" copy
for diffing during this optimization pass — use an inline copy of the
pre-change function (as done for item 2 above) or `git show HEAD:path` once
these files are actually committed.

## 2026-07-09 — item 1 shipped (feature-extraction batching)

Rewrote `extract_features_batch` (`features.py`) to fetch one shared raw
buffer per strand per cluster of nearby centers instead of one
`io.fetch_raw` call per position, and to bin/scale across the whole
cluster in one vectorized pass instead of looping `extract_features` per
center:

- Centers are sorted internally (input order need not be pre-sorted;
  restored on return) and grouped into clusters capped at
  `_MAX_SHARED_FETCH_WIDTH` (5,000,000bp) genomic span, falling back to a
  second shared fetch if a batch's positions are spread wider than that.
- Within a cluster, each zoom's W-bp bin sums are computed via a cumulative
  sum over the shared buffer (`csum[end] - csum[start]`) instead of a
  per-position `reshape(H, W).sum(axis=1)`. This is the key trick that
  makes vectorizing across positions tractable for wide zooms without
  memory blowup: a naive per-position gather-then-reshape across a batch
  would materialize an `(n_centers, W*H)` array per zoom (e.g. 100,000
  columns for the pretrained model's widest zoom), which for large batches
  (up to 200,000 queries on the cuml tier) would be tens to hundreds of GB.
  The cumsum-difference approach needs only `(n_centers, H+1)` per zoom
  (H ≤ 30), independent of window width W.
- `_logistic_scale_batch` vectorizes the per-position logistic scaling
  across rows via broadcasting (per-row max, no cross-row reduction) —
  exactly the same elementwise formula as the original per-position
  function, not an approximation.

**Validation**: bit-for-bit (`np.array_equal`) against the old
per-position-loop implementation on real chr21 data
(`_reference/dREG/example/K562.chr21.{plus,minus}.bw`) across 7 scenarios:
a dense sorted 500-position cluster, a 2000-position sample scattered
across the full ~46Mb chromosome (forces the multi-cluster fallback path),
the same sample unsorted, near-chromosome-start positions, near-
chromosome-end positions, a single-position batch, and all 182,605 real
chr21 informative positions at once. All exact.

**Precision caveat found during unit testing (not a real-world issue)**:
initial unit tests reused the existing `synthetic_bigwig_pair` fixture
(continuous Gaussian-curve signal, non-integer values) and one test
(forcing small clusters via a monkeypatched `_MAX_SHARED_FETCH_WIDTH`)
failed exact-equality by ~1e-10 to 1e-14 magnitude differences. Root cause:
cumsum-then-subtract and reshape-then-sum are only *bit-identical* when
summing exact integers in float64 (no rounding error regardless of
summation order) — for arbitrary non-integer floats, the two approaches
sum in a different order and can differ in the last few bits (still
numerically equivalent, just not bit-for-bit). This only matters because
the test fixture used unrealistic continuous-valued signal; dREG's actual
input contract is unnormalized point-mode read counts (see CLAUDE.md),
always integers, which is exactly what was validated bit-for-bit on real
chr21 data above. Fixed by adding a dedicated `integer_bigwig_pair` fixture
in `tests/test_features.py` (Poisson-count-based, matching the real input
contract) instead of reusing the continuous-signal fixture, and documented
the caveat in `_binned_sums_batch`'s docstring. 25/25 tests pass (added 4
new tests in `tests/test_features.py`).

## 2026-07-09 — final validation: full pipeline diff + timing

Ran the full CLI (`pydreg.cli`) end-to-end twice on the same 300kb chr21
slice used in prior sessions (chr21:~9,570,000-9,850,000, 1994 informative
positions, `--backend numpy`) — once against a reconstructed copy of the
pre-optimization code (all 6 items reverted, staged in a scratch directory
and loaded via `sys.path` injection since `src/` isn't git-tracked yet, so
`git stash`/`git show` can't produce a clean baseline), once against the
current optimized code — and diffed every output file.

**Result: fully equivalent except for one expected, pre-existing source of
noise.** `infp.bed.gz`, `infp.bw`, `peak.score.bed.gz`, `peak.score.bw`, and
the chrom/start/end/score/smooth.mode/original.mode/centroid columns of
`peak.full.bed.gz`/`raw.peak.bed.gz` are all bit-identical between old and
new. The exact same 16 raw candidate peaks and 7 significant peaks (same
loci) were called in both runs.

The one column that differs is `prob` (the pre-BH-FDR p-value from
`stats.pmv_laplace`, and its bigWig track `peak.prob.bw`) — differences of
~1e-4 to ~1e-8 absolute. **Root-caused as pre-existing, not a regression**:
`scipy.stats.multivariate_normal.cdf` (the Genz algorithm, matching R's
`mvtnorm::pmvnorm`) is an inherently stochastic quasi-Monte-Carlo method
with no fixed seed in this codebase. Calling the *unmodified original*
`pmv_laplace` five times in a row on identical `(x, cor_mat)` input gives
five different results (spread ~2e-7 on a test value of ~0.5835); calling
the *new* cached version five times gives the same spread. The old-vs-new
difference falls squarely inside that same run-to-run noise band — this
was already true of the pre-optimization code and is unrelated to the
`pmv_laplace` caching change (item 4) or anything else in this pass. All
observed `prob` values in the test run are well under the 0.05 significance
threshold (max ~0.03), so this noise never flips a significance decision in
practice, consistent with the identical significant-peak set observed.

**Timing** (log-timestamp deltas between pipeline phases, single run each,
this machine, `--backend numpy`, network/model-download time excluded):

| phase | old | new | delta |
|---|---|---|---|
| scoring informative positions (infp scan + feature extraction + SVR) | 13.05s | 9.46s | -28% |
| densify + broad-peak merge (gap-fill, 10bp redensify, broadpeak summary) | 3.68s | 3.11s | -15% |
| call_peaks (RF forest split + pmv_laplace + BH-FDR) | 9.45s | ~9.3-10.1s (2 runs) | ~neutral |
| **total** | 32.6s | 26.9s | **-17%** |

The scoring phase (items 1+2) shows the clearest, most representative win.
`call_peaks` (items 3+4+6) doesn't show a clear win *at this test's scale*
(only 10 broad peaks / 16 candidate summits) — those optimizations target
per-call Python/dispatch overhead that's amortized over thousands of broad
peaks on a real genome-wide run, not a few dozen; a 300kb slice is too small
a sample to see it, and isn't worth re-deriving a larger benchmark for here.
`numba`'s disk cache (`cache=True`) was confirmed present after the first
run (`__pycache__/models._forest_predict-79.py312.*.nbc`), so the
`call_peaks` timing above is not JIT-warmup cost, just genuinely small
absolute savings at this scale.

All 25 tests pass (`python3 -m pytest tests/ -q`). This closes out the
vectorization/JIT audit from 2026-07-09 — all 6 identified items shipped
and validated.

## 2026-07-09 — sklearn tier is ~15x slower than numpy on CPU; fixed default backend order

Prompted by a question about whether `scikit-learn-intelex` (Intel's oneDAL
patch for scikit-learn) is worth exploring to speed up the "sklearn" scoring
tier. Before chasing that, benchmarked the "numpy" and "sklearn" tiers
against each other directly (`scripts/bench_backends.py`, added this entry —
run it to reproduce on other hardware): both call the same RBF-SVR math
(`to_sklearn_svr()` reproduces `DREGModel.predict()`'s weights exactly,
verified to agree to ~1e-10), but `sklearn.svm.SVR.predict()` (libsvm) computes
one query-SV kernel value at a time in a C loop, while `DREGModel.predict`'s
chunked `X_scaled @ sv_block.T` / `K @ coefs` computes all of them at once via
BLAS.

**Correction (2026-07-14)**: the line above originally attributed this to
"single-threaded C loop vs. multithreaded BLAS" -- that's not the real
reason (see the 2026-07-14 entry below for the actual root cause, found by
reading libsvm's C++ source directly: forcing single-threaded BLAS via
`VECLIB_MAXIMUM_THREADS=1` doesn't change `DREGModel.predict`'s wall-clock
time at all, ruling out threading as the explanation).

Measured on this machine (605,187 SVs x 360 features, local safetensors,
1 rep):

| n | numpy | sklearn | ratio |
|---|---|---|---|
| 256 | 1.11s (231 pos/s) | 16.6s (15 pos/s) | 15.0x |
| 1024 | 4.74s (216 pos/s) | 66.3s (15 pos/s) | 14.0x |

**Conclusion on the original question**: not worth exploring intelex —
even a large intelex speedup on libsvm's predict would need to close a 14-15x
gap just to match the *already-CPU* numpy tier, and there's a real risk it
wouldn't engage at all: intelex's SVM acceleration hooks in via oneDAL state
built during `.fit()`, and `to_sklearn_svr()` deliberately skips `.fit()`
(writing directly to libsvm's private `support_vectors_`/`_dual_coef_`/etc.,
since this is a pretrained R model with no training data to refit from) —
so the patched estimator may have no oneDAL-backed state to dispatch
`.predict()` to.

**More importantly, this surfaced an actual bug**: `backend.detect_backend()`
preferred `"sklearn"` over `"numpy"` whenever `scikit-learn` was importable
— but `scikit-learn` is a hard dependency (`pyproject.toml`), so every
CPU-only user (i.e. anyone without cuML/a CUDA GPU, the common case per
README's "Caveats") was being silently routed to the ~15x-slower tier by
default. **Fixed**: `detect_backend()` no longer probes for/returns
`"sklearn"` at all — CPU auto-detection now goes straight to `"numpy"` (cuML
is still preferred first when a real GPU is usable). `"sklearn"` remains
selectable via `--backend sklearn` for anyone who wants it, and
`to_sklearn_svr()` is unchanged and still required as the input to
`cuml.svm.SVR.from_sklearn()` for the cuML tier.

No test relied on the old auto-detection order (`tests/test_pipeline.py`
pins `backend_name="numpy"` explicitly); all 25 tests still pass.

## 2026-07-14 — `pmv_laplace`'s default CDF precision was ~100-200x tighter than R's own reference, dominating peak-calling runtime

A real production run (`--peak-calling-cores 16`) reported: 40/807 blocks
done, 9817.23s of summed block CPU, with **9539.59s (97.2%) inside
`stats.pmv_laplace`** across 6859 calls / 1,245,449
`scipy.stats.multivariate_normal.cdf` evaluations (~7.7ms/eval). Extrapolated
to the full 807 blocks, this was many hours of wall-clock even parallelized
across 16 cores — parallelism (`peaks.py`'s `ProcessPoolExecutor`-based block
splitting, confirmed embarrassingly-parallel with no serialization
bottleneck) was already maxed out as a lever; the fix had to reduce the
actual per-CDF-eval cost.

**Root cause, not a tradeoff**: `pmv_laplace` calls `_box_cdf` up to 291
times per invocation (1 initial check + a fixed 290-point z-integration
grid — both counts confirmed exactly against the observed
1,245,449 = 291×4271 + 1×2588 split). The original R
(`_reference/dREG/dREG/R/peak_calling.R`'s `pmvLaplace`, calling
`pmvnorm(...)` with no `algorithm=` override) runs on R's `mvtnorm`
package's own default: `GenzBretz(maxpts=25000, abseps=1e-3, releps=0)`
— confirmed directly from `mvtnorm`'s R source (`R/mvt.R`) and its
`algorithms.Rd` doc, cross-checked via two independent sources. SciPy's own
defaults when `maxpts`/`abseps` are left unset (pydreg's previous behavior)
are `maxpts=1,000,000*dim=5,000,000`, `abseps=releps=1e-5` — **200x more
points and 100x tighter error tolerance than R's own reference
implementation ever used**. (`releps` is confirmed to have zero effect in
SciPy's actual d≥3 dispatch code either way — only `abseps`/`maxpts`
matter — so R's `releps=0` is moot.) So pydreg's previous "default" behavior
wasn't faithful to R at all; it was needlessly over-precise, and that
over-precision was the actual bottleneck.

**Fix**: `stats.py`'s `_PMV_CDF_MAXPTS`/`_PMV_CDF_EPS` module defaults (and
`set_pmv_laplace_cdf_options`'s parameter defaults) changed from
`None`/`1e-5` (SciPy's unset defaults) to `25000`/`1e-3` (R's real
`GenzBretz` defaults), propagated through every duplicate default (`cli.py`'s
two flags, `pipeline.py:run()`, `peaks.py`'s `_init_peak_worker`/`call_peaks`
— 6 signatures total, confirmed via grep). Also deleted `_box_cdf`'s
unfrozen-fallback branch entirely: confirmed against SciPy 1.18.0's actual
source that the frozen `multivariate_normal` constructor fully supports
custom `maxpts`/`abseps`/`releps` (the old comment claiming otherwise was
wrong), so there's now one code path, always frozen/cached, ~9% faster on
its own. The cache is now keyed on a version counter incremented by
`set_pmv_laplace_cdf_options()` rather than comparing option values for
equality, avoiding a float-equality-as-cache-key smell.

**Measured** (representative order-5 Toeplitz `cor_mat`, a "hard"
mid-probability `x`, 5 reps each):

| | mean time/call | std(p_max) | cdf_evals/call |
|---|---|---|---|
| old (SciPy unset defaults) | 3.03s | 8.4e-7 | 291 |
| new (R's real defaults) | 0.235s | 2.6e-5 | 291 |

**~12.9x speedup**, matching the ~13x predicted from isolated per-eval
benchmarking during investigation. The absolute difference between the old
and new mean `p_max` (1.0e-6) is far smaller than the pipeline's own
pre-existing, already-documented QMC run-to-run noise band (~1e-4 to 1e-8,
this file's 2026-07-09 entry) — i.e. the new, faster default is not less
accurate in any way that matters; it's simply no longer computing precision
R's own reference implementation never asked for.

Excluded from this fix (real but smaller/riskier wins, not pursued):
batching the 290 z-grid evals into one 2D SciPy call (~16% extra, measured
during investigation, but needs its own validation that SciPy's 2D batching
doesn't correlate QMC randomness across grid points); monkeypatching
SciPy's private `_qmvnt._cbc_lattice` to cache lattice construction across
calls (~1.5-2x extra at low `maxpts`, but couples to unstable private SciPy
internals). Worth revisiting if `pmv_laplace` is still the dominant cost
after this fix at real genome-wide scale.

All 38 tests pass.

## 2026-07-14 — real-world speedup was only ~2.9x, not ~12.9x: BLAS thread oversubscription across peak_calling_cores workers

A follow-up production run (same `--peak-calling-cores 16` machine, after
the above fix) reported: 2790.07s block CPU / 33 blocks = 84.6s/block and
2748.05s / 1,050,041 CDF evals = 2.62ms/eval — a real ~2.9x improvement over
the pre-fix numbers (245.4s/block, 7.66ms/eval), but far short of the
~12.9x measured in isolation on a single synthetic benchmark case.

**Hypothesis 1, tested and disproven: real boxes are "harder" (hit the
`maxpts=25000` ceiling more often) than the one synthetic case benchmarked
previously.** Monkeypatched `scipy.stats._qmvnt._qauto` to record
`n_samples`/`limit` for every CDF evaluation, then swept 5 correlation
structures (weak/mid/strong correlation × small/large variance) × 5 x-scales
(25 combinations total) through `pmv_laplace` at production settings
(`maxpts=25000, abseps=1e-3`). Result: **0/291 evals hit the ceiling in
every single combination** (average ~7010 of the 25000-point budget used),
and measured cost (0.64-0.95ms/eval) was *lower* than the original single-
case benchmark (0.808ms/eval), not higher. This directly rules out
ceiling-hitting as the explanation — real boxes converge comfortably within
budget across a wide difficulty range.

**Hypothesis 2, well-supported, not directly measurable from this sandbox
but consistent with all evidence: BLAS thread oversubscription across the
16 concurrent worker processes.** `_box_cdf`'s Genz-Bretz integration does
a handful of order-5 (5x5) Cholesky decompositions per call -- far too
small a matrix to benefit from BLAS multithreading at all, but on Linux
pip/conda installs (the likely production environment, vs. this sandbox's
Accelerate-backed macOS build) OpenBLAS/MKL default their thread pool size
to the *total visible core count*, independently, per process. With 16
worker processes on a reported 48-core remote machine, each defaulting to
~48 threads, that's up to 16x768 threads contending for 48 real cores --
pure scheduling/context-switch overhead with zero computational benefit,
repeated on the order of a million times. This matches the evidence
exactly: `peaks.py`'s profiling measures wall-clock (`time.perf_counter()`)
inside each worker, which inflates under contention even though the
underlying computation is unchanged -- explaining why a clean, uncontended
single-process sweep measured *lower* per-eval cost than production despite
deliberately testing harder cases.

**Fix**: pin every peak-calling worker to a single BLAS thread via
`threadpoolctl.threadpool_limits(limits=1)`, called once at the top of
`peaks.py`'s `_init_peak_worker` (the `ProcessPoolExecutor` `initializer=`,
also called directly in the main process for the `peak_calling_cores<=1`
serial fallback path). Per `threadpoolctl`'s own docs, calling this
(without using it as a context manager) applies process-wide and persists
for that process's lifetime -- exactly "once per worker at startup" is the
right idiom. `threadpoolctl` was already a transitive dependency (via
scikit-learn); added explicitly to `pyproject.toml` since `peaks.py` now
imports it directly. 16 single-threaded workers on a 48-core machine uses
16/48 cores with zero contention, and leaves headroom to raise
`--peak-calling-cores` further on that hardware.

Not verified end-to-end on the actual production machine (this sandbox has
no way to reproduce 16-way contention meaningfully) -- the diagnostic above
(disproving hypothesis 1) plus the well-established nature of this exact
BLAS-oversubscription failure mode in numpy/scipy multiprocessing workflows
is the basis for this fix; worth confirming with a real before/after
progress-log comparison on that machine.

All 38 tests pass.

**Correction, same day**: hypothesis 2 (BLAS oversubscription) is also
disproven by direct evidence from the production machine, not just
absence of confirmation. Two facts reported after this fix shipped: (a)
`htop`/`top` on that machine showed no sign of thread oversubscription
(no processes pegged well above 100% CPU, no runaway thread counts); (b)
reducing `--peak-calling-cores` from 16 to 1 reduced total throughput by a
*proportional* amount -- i.e. roughly linear scaling with core count. That
second fact is decisive: oversubscription would show up as *sub*-linear
scaling (16 workers achieving meaningfully less than 16x a single worker's
throughput, since contention overhead eats into the gain); clean linear
scaling means there was no meaningful contention penalty to begin with.
Also confirmed the remote machine runs the identical scipy version
(1.18.0) as this sandbox, ruling out a scipy-version/build difference too.

Net: the remaining gap between the ~12.9x measured in this sandbox and the
~2.9x observed in production is most likely simple per-core hardware
throughput difference (this sandbox is Apple Silicon; the production
machine is a remote x86 server) for this specific Cython-heavy numeric
workload -- not a bug, and not something this codebase can fix. The
`threadpoolctl` pin from this entry is kept regardless (pinning BLAS
threads for a 5x5 Cholesky decomposition is a no-regret safeguard on any
hardware -- it cannot be a regression, since that matrix is too small to
benefit from multithreading anywhere), but should not be credited with
explaining the observed real-world speedup, which is apparently just what
that hardware honestly delivers for this algorithm.

If more speed is still needed, the next levers are the ones already
identified and excluded above (batching the 290 z-grid evals into one 2D
call, ~16%, reduces per-call Python/SciPy-API overhead rather than QMC
sample cost -- plausibly more relevant now if overhead is a bigger
fraction of cost on this hardware) or, at higher effort/risk, a custom
compiled (e.g. numba) Genz-Bretz implementation tailored to this exact
fixed-`cor_mat`/scalar-scaled-box structure, to cut SciPy's generic
per-call overhead rather than the QMC math itself.

## 2026-07-14 — Phase A: cache SciPy's redundant per-eval QMC-lattice construction (~16.5% shipped); Cholesky-caching attempted and dropped

Following up on the user's request to pursue a custom Genz-Bretz
reimplementation for further speedup: before committing to that (multi-day,
real correctness risk -- a numerical bug here silently corrupts p-values,
not just slows things down), did a research pass first (3 agents) to
quantify exactly where `pmv_laplace`'s remaining cost lives and whether a
safe, low-risk win was available first.

**Cost breakdown** (direct `perf_counter` instrumentation, not cProfile --
an initial cProfile trace looked like it showed ~60% "avoidable Python
overhead" inside SciPy's `_qmvn`, but that's a profiling artifact from
cProfile not tracking the compiled Cython kernel call as a separate frame;
corrected via direct instrumentation of the actual internal functions,
cross-checked two independent ways): the compiled QMC kernel (`_qmvn_inner`)
is **~76-80%** of per-eval cost and is not reducible without loosening
precision (off the table -- that's exactly the R-fidelity setting fixed in
the prior entry). Of the remaining ~20-24%:
- `_cbc_lattice` (QMC lattice construction, ~11%) is **100% reusable**:
  `_qauto`'s adaptive point-growth loop converges in exactly 1 iteration
  every single time for this codebase's fixed order-5 `cor_mat`/
  `maxpts=25000` usage (verified across 5 test cases × 291 evals each,
  1455 evaluations total, zero exceptions), and since its starting point
  budget depends only on the fixed dimension and `maxpts` -- never on the
  box or `cor_mat` values -- the exact same lattice is legitimately reused
  for every eval, in every call, for the whole process's lifetime.
- `_permuted_cholesky` (box reordering + Cholesky factorization, ~5-8%)
  has highly repetitive *output* (only 2-4 distinct results across a whole
  291-point z-grid, confirmed empirically) but its *inputs* (the box bounds,
  continuously rescaled by `1/sqrt(z0)` per z-grid point) are essentially
  always unique. **Attempted caching keyed on the literal input values;
  measured result: 290/291 calls still recomputed** (see below) -- the
  redundancy lives in the output/permutation-decision space, not the
  continuous input space, so a naive input-keyed cache can't capture it.

**Shipped**: `src/pydreg/stats.py` now wraps `scipy.stats._qmvnt._cbc_lattice`
with a small dict cache keyed on `(n_dim, n_qmc_samples)` (pure/deterministic
given these), monkeypatched once at module-import time. Zero numerical
risk: it only reuses exactly the values SciPy would otherwise have
recomputed, never touches the randomized QMC shift/kernel evaluation
itself (drawn fresh per eval from SciPy's own RNG state, entirely
independent of this cached deterministic setup step), so `pmv_laplace`'s
existing run-to-run QMC noise is completely unaffected. Returns a `.copy()`
of the cached array on every access so no caller can corrupt the cache via
in-place mutation.

**Not shipped**: the equivalent cache for `_permuted_cholesky` was
implemented, measured, and **reverted** -- it added dict-lookup/array-copy
overhead for effectively zero benefit (290/291 cache misses in the
representative test case), since the actual redundancy is in which
*permutation* the algorithm chooses (a discrete decision depending
nonlinearly on the box bounds via `Phi(hi)-Phi(lo)` comparisons), not in
the literal bound values passed in. Safely exploiting that would mean
cheaply detecting which permutation class a box falls into *without*
running the expensive pivot-selection algorithm -- i.e. partially
reimplementing SciPy's private algorithm, not just wrapping it in a cache.
That's real reimplementation risk (the ~7-9% minority of evals using a
different permutation are not numerically negligible -- an approximate/
quantized cache key could silently misclassify some of them), and is
better scoped to a from-scratch implementation (Phase B, where this logic
gets reimplemented deliberately and validated end-to-end) than bolted on
here as a "safe wrapper."

**Measured** (representative order-5 `cor_mat`, hard box, 5 reps, this
machine):

| | mean time/call |
|---|---|
| before (no lattice cache) | 0.2048s |
| after (lattice cache only) | 0.1710s |

**~16.5% speedup** (1.20x) -- real, safe, and unconditionally shipped
regardless of what Phase B (the custom-kernel investigation) concludes.

**Also found, load-bearing for Phase B**: SciPy's CDF algorithm depends on
SciPy version, and this project's pinned SciPy (1.17.1/1.18.0) is on the
far side of a real backend swap (confirmed via SciPy PRs #22298/#22611 and
direct source reading). SciPy < 1.16 used a Fortran extension (`mvndst.f`,
Alan Genz's own MVNDST routine: Cholesky reorder + **tabulated Korobov
lattice + Cranley-Patterson randomization**) -- the *same lineage* as R's
`mvtnorm::pmvnorm()` (`mvt.f`'s `MVTDST`: `MVSORT` + `MVKBRV`, same recipe,
same author). SciPy >= 1.16 (what this codebase actually runs) replaced
that with a pure-Python/Cython reimplementation (`_qmvnt.py`) using a
**Fast Component-by-Component (CBC) lattice construction** (Nuyens & Cools
2004/2006 -- a different, later, unrelated-to-Genz publication) and its
own independently-written adaptive stopping rule, not a port of `MVKBRV`'s
logic. SciPy's own PR discussion acknowledges this is a real algorithm
change, not a value-preserving refactor. **This means the prior entry's
`maxpts=25000, abseps=1e-3` fix corrected the tolerance *settings* to match
R, but there remains a deeper, currently-unquantified algorithmic gap
between what this codebase computes (CBC lattice) and what R actually
computes (Korobov lattice)** -- independent of, and in addition to, the
ordinary QMC run-to-run noise both methods already share. Not yet measured
whether this gap is within the existing tolerated noise band or materially
larger; queued as the first step of the Phase B investigation (see the
session's plan file / next entry) before any custom-kernel implementation
effort, since it directly bears on what a from-scratch reimplementation
should actually target (R's Korobov-lattice method, not SciPy's current
CBC-lattice method).

All 39 tests pass (added one new test verifying `_cbc_lattice` is called
far fewer times than the number of CDF evaluations in a real `pmv_laplace`
call).

## 2026-07-14 — Phase B0: the CBC-vs-Korobov algorithm difference is NOT a real fidelity gap

Directly tested the load-bearing question from the entry above: does SciPy's
current CBC-lattice algorithm (`_qmvnt.py`, what this codebase actually
runs) produce meaningfully different results than the Korobov-lattice
algorithm R's `mvtnorm::pmvnorm()` actually uses (the same lineage as
SciPy's own pre-1.16 Fortran backend)?

Installed `scipy==1.15.2` (the last version with the Fortran `mvndst.f`
backend, confirmed via reading its `_cdf` source: it calls `_mvn.mvnun`,
not `_qauto`/`_qmvn`) into an isolated throwaway venv, and ran a
standalone replica of `pmv_laplace`'s exact math (z-grid + box-CDF loop,
`maxpts=25000, abseps=releps=1e-3`) under both that old Fortran/Korobov
backend and the currently-pinned `scipy==1.18.0` CBC-lattice backend, on
4 representative `(cor_mat, x)` cases, 15 reps each (both are stochastic):

| case | old (Fortran/Korobov) mean | new (CBC, current) mean | old std | new std |
|---|---|---|---|---|
| hard_mid_prob | 0.423285 | 0.423285 | 3.58e-7 | 1.03e-8 |
| small_x | 0.046348 | 0.046348 | 8.46e-7 | 3.84e-8 |
| wide_x | 0.917619 | 0.917619 | 2.57e-6 | 1.66e-7 |
| asymmetric | 0.470491 | 0.470491 | 4.51e-8 | 1.37e-9 |

**The two algorithms' means agree to within (often well within) each
algorithm's own run-to-run noise, on every case tested.** There is no
detectable systematic divergence between SciPy's current CBC-lattice
method and the Korobov-lattice method R actually uses -- both converge to
the same true integral value, as expected of two different-but-valid
randomized QMC estimators of the same quantity. Interestingly, the current
(CBC) backend's own noise is consistently *smaller* than the old
Fortran/Korobov backend's, at the same nominal `maxpts`/`abseps`.

**Conclusion: the algorithm-lineage difference flagged in the entry above,
while real (confirmed from source: different lattice-construction method,
different adaptive stopping-rule implementation), is not a practically
meaningful fidelity gap.** It does not move the needle on "how close is
this codebase's `pmv_laplace` to what R actually computes" beyond the
ordinary QMC noise both methods already share. This removes the fidelity
argument for a custom from-scratch reimplementation (Phase B1, the numba
kernel prototype) -- that step would now be justified on performance
grounds alone, which is a materially weaker case given the uncertainty
already flagged (scipy's kernel is already compiled; numba beating it is
unproven) and the multi-day effort/correctness-risk involved. Per the
session's plan, B1 was conditioned on B0 finding a real gap; since it
didn't, checking with the user before spending further effort on B1.

## 2026-07-14 — batch the 290 z-grid CDF evaluations into one 2D SciPy call (~3% at real settings, not the ~16% estimated earlier)

Given B0 ruled out the fidelity motivation for a from-scratch kernel (B1),
and B1's performance case alone was judged too uncertain against already-
compiled Cython for the multi-day effort/risk, took the lower-risk, already-
identified batching option instead: SciPy's `multivariate_normal.cdf()`
accepts a 2D `x`/`lower_limit` directly (its `_cdf` dispatches via
`np.apply_along_axis`), so `pmv_laplace`'s z-grid loop -- previously 290
separate Python-level `.cdf()` calls -- is now a single call with `x` shaped
`(290, 5)`. Verified this produces the same values (within ordinary QMC
noise, confirmed via direct comparison, max abs diff 3.8e-7 on a
representative case) before shipping, since each row is still evaluated
independently with its own random draws -- batching only combines the outer
SciPy-level dispatch, not the randomization.

**Measured, isolating batching's contribution alone (identical lattice-
cache state both sides, same representative case)**:

| | mean time |
|---|---|
| looped (290 separate `.cdf()` calls) | 0.1702s |
| batched (one `.cdf()` call, `x` shape (290,5)) | 0.1651s |

**~3.0% reduction (1.03x)** -- real, but far smaller than the ~16% estimated
during the original investigation. That earlier estimate was measured at
`maxpts=1000, abseps=1e-2` (a much lower-precision regime where per-call
Python/SciPy dispatch overhead was a much larger fraction of a much smaller
per-call cost); at this codebase's actual `maxpts=25000, abseps=1e-3`
settings, the QMC kernel so thoroughly dominates (~76-80% of cost, per the
Phase A entry above) that there's proportionally little dispatch overhead
left for batching to amortize. Shipped anyway since it's free and safe (zero
numerical risk, same reasoning as the lattice cache), and stacks with it.

**Combined effect of this session's three `pmv_laplace` changes** (R-
fidelity precision fix from the prior session + lattice caching + batching),
measured end-to-end on the same representative hard-box case used
throughout: **~3.03s -> ~0.165s per call, ~18.4x total**, on this
(uncontended, Apple Silicon) machine -- real production hardware should be
judged by its own before/after progress-log numbers, not this figure
directly, per the earlier finding that raw per-core throughput differs
substantially by hardware for this workload.

All 39 tests pass.

## 2026-07-14 — Phase B1: numba kernel prototype does not beat SciPy's compiled Cython kernel

Read `_qmvnt_cy.pyx` directly from SciPy's source (v1.18.0 tag) to get the
*exact* per-eval algorithm: Cholesky-transformed box, tent-periodized CBC
lattice point, Cranley-Patterson random shift, `ndtri`/`ndtr` (`Phi^-1`/
`Phi`), online-updated mean and error variance across 10 batches. Ported
this line-for-line to a `numba.njit` kernel (`ndtri` via a hand-rolled
Acklam rational approximation, since `scipy.special.ndtri` isn't
numba-typeable -- same approach validated in the B0/research phase, ~1e-9
error, far inside the 1e-3 `abseps` target), reusing SciPy's own
`_permuted_cholesky`/`_cbc_lattice` outputs (both already cached from Phase
A) so the comparison isolates exactly one variable: **does a numba-compiled
version of this inner loop beat SciPy's compiled Cython kernel**, given
byte-identical lattice points and random shifts.

**Correctness**: on 4 representative cases, the numba port reproduces
SciPy's probability estimate to 1e-12 to 1e-13 -- confirms it's a faithful,
same-algorithm port (the tiny residual is float op-ordering/`erf`
implementation noise), not a coincidence.

**Speed, head-to-head on identical captured inputs (`n_qmc_samples=701,
n_batches=10`, 200 reps each, JIT warm-up excluded)**:

| case | scipy kernel | numba kernel (`fastmath=False`) | speedup | numba kernel (`fastmath=True`) | speedup |
|---|---|---|---|---|---|
| hard_mid_prob | 751.2us | 841.5us | 0.89x | 860.4us | 1.13x |
| small_x | 479.9us | 434.0us | 1.11x | 463.3us | 1.08x |
| wide_x | 875.6us | 1060.0us | 0.83x | 757.5us | 1.40x |
| asymmetric | 782.7us | 912.0us | 0.86x | 709.2us | 1.07x |

**Verdict: no meaningful win.** `fastmath=False` (the fair, IEEE-safe
comparison) is a wash-to-slight-loss (0.83x-1.11x); `fastmath=True` nudges
it to 1.07x-1.40x, still well under the plan's 1.5-2x bar for "genuine,
meaningful," and it relaxes IEEE float semantics (reordering,
no-NaN/no-Inf assumptions) in exchange -- a real precision trade this
codebase's "maintain exactness with R" priority argues against taking for
an unproven, sub-1.5x gain. SciPy's kernel is already a tight, compiled,
purpose-built loop over scalar `erf`/`log`/`sqrt` calls; numba's general
JIT doesn't have obvious room to beat that for this shape of computation.

Did not go on to build the "stronger direction" from the plan (batch all
290 z-grid evaluations inside one compiled call, sharing one Cholesky
decomposition across all of them) -- its ceiling is capped by this same
per-eval-kernel finding (the ~76-80% dominant kernel cost isn't faster in
numba), and the outer-dispatch overhead it would additionally amortize is
the same ~1-3% already captured by the batching change above; the
remaining upside isn't worth the correctness risk of forcing one box's
Cholesky permutation onto all 290 z-grid points (the ~7-9% minority that
scipy deliberately reorders differently, per the Phase A cost-breakdown
entry).

**Conclusion: per the plan's explicit stopping rule ("if B1 shows no
meaningful win: stop here"), this closes out the `pmv_laplace` custom-kernel
investigation.** Phase A's caching + the batching change above are the
realistic performance ceiling without a fundamentally different algorithm
or hardware. Prototype code is scratch-only (not part of the package);
nothing shipped from this entry beyond the writeup.

## 2026-07-14 — found and fixed the real "much slower than R" cause: SciPy's adaptive driver has a hardcoded ~5000-sample floor per box; R's doesn't

The B1 conclusion above ("Phase A + batching is the realistic ceiling") was
wrong -- it only ever asked "is a from-scratch kernel faster than SciPy's
compiled one at the SAME sample count SciPy chooses." It never questioned
whether that sample count itself was the actual problem. Asked to audit the
peak-calling logic against R directly (not just re-benchmark), read
`peak_calling.R`/`peak_calling_rf.R`/`rfsplit.py` end-to-end first --
`pmv_laplace` is called exactly once per final peak region in both R and
Python (confirmed line-by-line, no extra-call bug there) -- then read
mvtnorm's actual Fortran source (`mvt.f`'s `MVKBRV`, fetched from
`cran/mvtnorm`) to see how R's `pmvnorm(GenzBretz(maxpts=25000))` actually
decides how many QMC samples to draw per box.

**The real gap**: SciPy's public adaptive driver (`_qmvnt._qauto`, used
inside `multivariate_normal.cdf`) always starts its search at
`min(maxpts, n_dim*1000)` -- for this codebase's fixed 5-dim `cor_mat`, a
hardcoded floor of ~5000 raw samples (`n_qmc_samples~=701` per batch x 10
batches) *per box*, no matter how easy that box is. R's `MVKBRV` has no such
floor: it walks a fine table of lattice sizes (`P(1)=31, P(2)=47, P(3)=73,
P(4)=113, P(5)=173, P(6)=263, ...`), starting at `P(MIN(NDIM,10))` --
`P(4)=113` for this problem's dimensionality -- and stops as soon as the
error estimate meets `abseps`. Most of dREG's boxes are nowhere near the
50/50 decision boundary and converge in the very first, tiny round; SciPy's
API has no way to express that and always pays for ~5000+ samples anyway.
This -- not raw per-sample kernel speed, which the B1 prototype already
showed is roughly at parity between SciPy's Cython and a from-scratch numba
port -- is the actual reason peak calling ran much slower than R.

**Fix**: `_qmvn_adaptive` in `stats.py` bypasses `multivariate_normal.cdf`/
`_qauto` entirely and drives SciPy's own private, trusted kernel
(`_qmvnt._qmvn` -- same lattice construction, same Cholesky reordering, same
randomized QMC evaluation, completely unchanged) with a custom loop that
starts at 150 samples instead of ~5000, growing by the same `sqrt(2)` factor
SciPy's own driver uses, checking the same stopping condition
(`est_error <= abseps`) on every round. This does not reimplement or copy
any of R's actual lattice-generator tables (`C`/`P` arrays in `mvt.f`,
GPL-2) -- it only changes how many samples SciPy's own machinery is asked
for before the first convergence check, matching the *shape* of R's
adaptive behavior (start small, grow only as needed), not its literals. The
final-precision guarantee is unchanged: the loop cannot return early with
worse error than before, since the stopping condition is identical to
SciPy's own.

**Validation** (25 random order-5 `cor_mat`/`x` cases, AR(1)-decay
correlation structure matching `build_cormat`'s real output, starting points
50/100/150/250 all tested):

| start | mean speedup | diff vs. SciPy-floor mean | diff max | mean `_qmvn` calls/pmv_laplace call |
|---|---|---|---|---|
| 50 | 4.58x | 1.23e-05 | 4.95e-05 | 293.0 |
| 100 | 4.44x | 4.30e-06 | 2.19e-05 | 291.2 |
| 150 | 4.32x | 2.85e-06 | 1.30e-05 | 291.0 |
| 250 | 4.05x | 1.78e-06 | 8.69e-06 | 291.0 |

All differences are far inside the ordinary run-to-run QMC noise both
methods already carry (~1e-6 to ~1e-7 per the B0 entry above). Picked
`start=150`: essentially always converges in exactly one round (mean/max
calls both 291.0, i.e. one `_qmvn` call per box, matching the un-adapted
count exactly) while keeping the smallest max-diff among the options tested.

**Trade-off**: this drops the "batch 290 z-grid evaluations into one 2D
SciPy `.cdf()` call" optimization from the prior entry -- `_qmvnt._qmvn`
only accepts one box at a time, so the z-grid loop is back to a Python
`for` loop. That optimization was only ever worth ~3%, dwarfed by this
change, so nothing of consequence is lost; the `_cbc_lattice` cache (Phase
A) is untouched and still applies transparently (same monkeypatch, same
near-constant `n_qmc_samples` since almost every box now converges at the
same `start=150`-derived sample count).

**Measured end-to-end** (real `pydreg.stats.pmv_laplace`, caches warm, 30
reps per case):

| case | mean time/call | mean value | std |
|---|---|---|---|
| hard_mid_prob | 32.1ms | 0.423285 | 2.7e-6 |
| small_x | 33.7ms | 0.046347 | 9.0e-6 |
| wide_x | 31.8ms | 0.917616 | 1.8e-5 |
| asymmetric | 33.4ms | 0.470491 | 9.8e-7 |

Compare to the prior entry's ~165ms/call (Phase A + batching) and the
session's original ~3.03s/call baseline: **~5x additional speedup on top of
everything shipped so far, ~94x total this session**, on this
(uncontended, Apple Silicon) machine. `hard_mid_prob`'s mean (0.423285)
matches the B0 fidelity entry's reference value exactly. `cdf_evals` in
`get_pmv_laplace_profile()` now counts actual underlying `_qmvn` kernel
invocations (usually 291 per call, one per box) rather than the previous
fixed 291-per-batched-call constant -- a more direct measure of real work,
worth knowing when reading old vs. new progress-log numbers side by side.

All 39 tests pass.

## 2026-07-14 — numba port of `_permuted_cholesky`: bit-exact, 28x faster in isolation, ~1.85x on real `pmv_laplace`

Asked to look at whether `smoothing.py`/`rfsplit.py` were the next bottleneck
(a real production log showed non-pmv at 24.4% of block CPU, up from 2.8%/
1.5% in earlier logs, now that `pmv_laplace` itself is so much faster).
`smoothing.py`'s boxcar smoothing is already cumsum-vectorized -- profiling
confirmed it's not a bottleneck. But profiling `pmv_laplace` itself
(cProfile over 1000 calls) turned up something bigger: `_permuted_cholesky`
-- the box-reordering/Cholesky-factorization setup step SciPy's `_qmvn`
runs before every QMC evaluation -- was **15.991s tottime / 21.859s cumtime
out of 51.659s total (~42%)**. This step was deliberately left uncached in
an earlier entry above, when it was only ~5-8% of a much larger total; the
small-start adaptive fix shrank the kernel's own cost so much that this
previously-minor setup step is now proportionally dominant.

Unlike `_qmvn_inner` (the QMC kernel, already compiled Cython -- a numba
port of that was tried and found no win, see the B1 entry above),
`_permuted_cholesky` is **pure Python** (confirmed by reading the installed
source directly). A numba port has real room to win here specifically
because the original was never compiled at all -- this is a different
situation from B1's numba-vs-Cython comparison, not a re-litigation of it.

**Implementation**: `_permuted_cholesky_numba` in `stats.py` is a
line-for-line `@numba.njit` translation of SciPy's own `_permuted_cholesky`
(read directly from the installed `scipy.stats._qmvnt` source), using a
hand-rolled `_ndtr` (`0.5*(1+erf(x/sqrt(2)))`) matching SciPy's `phi`.
Monkeypatched onto `_qmvnt._permuted_cholesky` at import time, same pattern
as the existing `_cbc_lattice` cache (original saved as
`_orig_permuted_cholesky`, so a future SciPy rename/removal fails loudly
rather than silently falling back to uncached-but-correct behavior).

**Correctness**: this is a deterministic pivoted-Cholesky algorithm with no
randomization involved at all -- so unlike every other change in this log,
there's no "within QMC noise" caveat: the numba port must match SciPy's own
output to near machine precision on every input. Validated across 200
random `(cor_mat, low, high)` cases (AR(1)-decay correlation structure,
box scales spanning `z0` from `1e-3` to `1e3`): **max diff = 0.0** (bit-exact).
A permanent regression test (`test_permuted_cholesky_numba_matches_scipy_exactly`)
checks this on 100 cases with a `1e-12` tolerance (loosened slightly from
literal bit-exactness only to avoid being a flaky float-ULP-sensitive test).

**Speed**: 53.9us/call (SciPy's pure Python) -> 1.9us/call (numba) in
isolation, a **28x** speedup.

**Measured end-to-end effect on real `pmv_laplace`** (4 representative
cases, caches/JIT warm, 40 reps each):

| case | before (scipy Cholesky) | after (numba Cholesky) |
|---|---|---|
| hard_mid_prob | 32.1ms | 17.5ms |
| small_x | 33.7ms | 17.1ms |
| wide_x | 31.8ms | 17.7ms |
| asymmetric | 33.4ms | 17.6ms |

**~1.85x additional speedup**, landing consistently across all 4 cases
(an earlier ad-hoc monkeypatch test during investigation showed uneven
per-case results, apparently a measurement artifact -- the properly wired
version above is consistent). Combined with everything else this session
(R-fidelity precision fix + lattice caching + small-start adaptive sample
schedule): **~3.03s -> ~17.5ms per call, ~173x total**, on this
(uncontended, Apple Silicon) machine -- as always, judge real production
hardware by its own before/after progress-log numbers.

All 40 tests pass.

## 2026-07-14 — 3-way block-CPU profiling breakdown (diagnostic only, no behavior change)

Synthetic profiling of `rfsplit.find_rf_peaks` (see the entry above) found a
real inefficiency in `_split_peak`'s sequential merge loop (repeated tiny
`model.predict()` calls, each rebuilding its input array from a list of
Python dicts), but even an extreme synthetic case (4000 points, 160 local
maxima) only produced ~13ms/call of non-pmv cost -- about 3x less than the
~42.7ms/peak implied by a real production log's 24.4%-non-pmv share. Rather
than guess which of several candidates (that RF-split inefficiency, or
something in `_call_peak_block`'s per-peak `searchsorted`/DataFrame handling
outside `find_rf_peaks` entirely) actually dominates in production, added a
3-way timing breakdown so the next real run answers this directly instead.

**Change**: `_call_peak_block` (`peaks.py`) now also accumulates
`find_rf_peaks_seconds` -- a timer bracketing only the
`rfsplit.find_rf_peaks(...)` call itself, separate from the existing
whole-block timer (`seconds`) and the existing `pmv_seconds` (from
`stats.get_pmv_laplace_profile()` deltas, unaffected). `call_peaks()`'s
progress-log line and final summary now report:
- `pmv_seconds` (unchanged) -- time inside `pmv_laplace`.
- `find_rf_peaks_non_pmv = find_rf_peaks_seconds - pmv_seconds` -- smoothing +
  `_split_peak` (incl. the `model.predict` merge-loop overhead) + result
  building, *inside* `find_rf_peaks`.
- `other_block = seconds - find_rf_peaks_seconds` -- `np.searchsorted`/
  slicing per peak, the `xp.shape[0] <= 3` skip check, and the per-peak
  `result.copy()`/`.insert()`/`raw_rows.append` DataFrame handling, all
  *outside* `find_rf_peaks` within `_call_peak_block`'s loop.

Purely additive (new timers only, no behavior change). **Verified the
3-way split is exact by construction** (`pmv_seconds + find_rf_peaks_non_pmv
+ other_block` telescopes algebraically back to `seconds`, since
`find_rf_peaks_non_pmv` and `other_block` are defined as differences of
nested supersets) and confirmed numerically on a small local `call_peaks`
run: `seconds=0.000937s`, reconstructed sum `=0.000937s`, diff `0.00e+00`;
breakdown was `pmv=0.33ms`, `find_rf_peaks_non_pmv=0.45ms`,
`other_block=0.16ms` for that run.

**Next step**: run a real production peak-calling job and read the new
3-way split in its progress log -- whichever of `find_rf_peaks_non_pmv` or
`other_block` dominates tells us precisely where to target a follow-up fix
(the `_split_peak`/`model.predict` call-overhead issue if the former; the
`_call_peak_block` DataFrame-assembly path if the latter), instead of
guessing further from synthetic data that couldn't reproduce the observed
production magnitude.

All 40 tests pass.

## 2026-07-14 — why is sklearn so much slower than the numpy/numba tiers? (SVM correction + new RF finding)

Asked to explore *why*, not just confirm *that*, sklearn is slower than
pydreg's own numpy (SVM) and numba (RF) implementations, for both models --
this corrects the 2026-07-09 entry's SVM explanation and adds a new RF
finding neither entry covered before.

**SVM: not actually about threading.** Tested the 2026-07-09 entry's
"single-threaded C loop vs. multithreaded BLAS" claim directly:
`VECLIB_MAXIMUM_THREADS=1` (forcing single-threaded Accelerate BLAS) gives
the *same* wall-clock time as the default (1.10s vs. 1.15s on a 256-query
batch), and the default run's own `cpu_time/wall_time` ratio is only ~1.2x
-- nowhere near saturating this machine's 10 cores. So multithreading isn't
the differentiator at all. Reading libsvm's actual C++ source
(`sklearn/svm/src/libsvm/svm.cpp`, `svm_predict_values` -> `Kernel::k_function`)
shows the real shape of the computation: for *each* query point, it loops
over all 605,187 support vectors and calls `k_function` once per pair, and
each call does `malloc(sizeof(double)*360)` + a single BLAS **level-1**
`dot()` (360 elements) + `free()` -- i.e. 605,187 heap-alloc/free round trips
and 605,187 tiny vector-vector BLAS calls, per query. `DREGModel.predict`
instead issues one BLAS **level-3** `X_scaled @ sv_block.T` GEMM call that
computes every query x SV dot product at once. Level-3 GEMM is cache-blocked
and SIMD-vectorized across the whole computation; repeated tiny level-1
calls (plus the allocator overhead) can't get anywhere near that throughput
regardless of thread count -- a genuinely different computational shape, not
a parallelism gap.

**RF: pure per-call Python/joblib orchestration overhead, confirmed via a
same-scale sklearn model built for direct comparison.** Fit a
`RandomForestRegressor(n_estimators=500, max_depth=8)` (matching pydreg's
real forest's 500 trees) on synthetic data and benchmarked `.predict()`
against `DREGPeakSplitForest.predict()` (numba) at matched row counts:

| n rows | sklearn (n_jobs=1) | sklearn (n_jobs=-1) | numba | ratio (1job/numba) |
|---|---|---|---|---|
| 1 | 9.9ms | 26.5ms | 0.016ms | **624x** |
| 20 | 10.2ms | 25.8ms | 0.26ms | 40x |
| 1000 | 28.8ms | 38.3ms | 13.0ms | 2.2x |
| 10000 | 176.7ms | 52.1ms | 131.9ms | 0.4x (parallel sklearn is *faster*) |

A flat ~10-26ms regardless of `n` until `n` reaches the thousands, then it
scales with real work -- the fixed-overhead signature. `cProfile` on the
`n=1` case (200 reps) pins it down exactly: `RandomForestRegressor.predict()`
dispatches **one joblib `Parallel`/`delayed()` task per tree** (500 tasks for
one top-level `.predict()` call), and each tree's own
`DecisionTreeRegressor.predict()` independently re-runs sklearn's full
estimator-API machinery from scratch: `check_is_fitted` (500 calls),
`get_tags()`/`__sklearn_tags__()` introspection (500-700+ calls),
`functools.update_wrapper` decorator rebuilding (500 calls), even
`warnings.filterwarnings()` and `re.compile()` (2500 calls *each*). None of
that is tree-traversal math -- it's the validated/introspectable general-
purpose estimator API, paid 500 times per prediction call. `_forest_predict`
does the whole 500-tree traversal inside one compiled numba function with
zero per-tree Python overhead, which is why it wins by 2-3 orders of
magnitude at pydreg's actual call shape (1-20 rows per call, confirmed in
the 2026-07-14 RF-split entry above) and only loses once `n` reaches the
thousands and multi-core parallelism has enough real work to amortize its
own dispatch cost.

**Common thread, not a coincidence**: sklearn's estimators are built and
tuned for a few large-batch calls (typical training/serving shape); pydreg's
actual workload -- confirmed independently for both models in this and
earlier sessions -- is many tiny calls. For SVM the mismatch is algorithmic
(libsvm's predict path has no batched-kernel-matrix option at all); for RF
it's pure API/orchestration overhead that happens to fully amortize past
~1000 rows. Neither is a bug in sklearn -- both are the wrong tool for this
specific call shape, which is exactly why pydreg carries its own numpy/numba
implementations instead of depending on sklearn's for the hot path.

Research only, no code changed; `backend.py`'s `detect_backend()` docstring
and the correction above already reflect this.

## 2026-07-14 — end-to-end validation: 0.999728 Jaccard index vs. real dREG on test data

The user ran pydreg's full peak-calling pipeline against real dREG (the
original R package) on test data and compared the resulting peak sets:
**Jaccard index (|pydreg ∩ dREG| / |pydreg ∪ dREG|) = 0.999728.**

This is the real-world fidelity check the "Ground rule" above asks for --
not literal bit-identical output (not expected, given `pmv_laplace`'s
inherent QMC randomization and the already-documented CBC-vs-Korobov
lattice-algorithm difference between SciPy and R's `mvtnorm`, see the
2026-07-14 B0 entry above), but near-total agreement on the actual called
peak set after this session's full run of changes: the R-fidelity
`maxpts`/`abseps` precision fix, `_cbc_lattice` caching, the small-start
adaptive QMC sample schedule (the biggest behavioral change of the session,
since it alters exactly how many samples `pmv_laplace` draws per box), and
the `_permuted_cholesky` numba port (bit-exact vs. SciPy's own
implementation, so not expected to move this number at all). None of that
work traded away correctness for speed -- this is the empirical confirmation.

## 2026-07-14 — correction: the `_permuted_cholesky` numba port is not *always* bit-exact, and that's fine

`test_permuted_cholesky_numba_matches_scipy_exactly` (from the entry above)
failed intermittently in CI/other runs despite passing repeatedly here --
`assert max_diff < 1e-12` failing with `max_diff` in the single-to-low-
hundreds range, not a tiny precision miss. Root-caused by searching 200,000
random cases directly on this machine (not a cross-platform artifact):
`_permuted_cholesky`'s greedy pivot search picks the dimension with the
smallest remaining marginal probability range (`de = phi(hi_i) - phi(lo_i)`)
at each step. For a near-saturated box -- every dimension's marginal
probability already ~1, exactly what happens at the small-z0 tail of
`pmv_laplace`'s own z-grid -- every candidate's `de` is a near-tie
simultaneously, not just two candidates against each other. At that point,
which pivot wins is decided by sub-ULP floating-point differences between
NumPy's `@` (used inside SciPy's own pure-Python implementation) and this
port's explicit summation loop -- found a concrete case where this flips
the chosen pivot order, producing intermediate `cho`/`lo`/`hi` arrays that
differ by up to ~196 (roughly 1-in-40000 random draws in the search).

**This is not a correctness bug.** Verified directly on the worst case
found: feeding *either* pivot choice's resulting `(cho, lo, hi)` into
SciPy's own unmodified `_qmvn_inner` kernel (same lattice, same random
shift, isolating the decomposition as the only variable) produces the same
final probability (0.9999999999999998 vs. 1.0), matching a high-precision
`scipy.stats.multivariate_normal.cdf` reference computed independently
(0.9999999999999999). The pivot order is a variance-reduction heuristic for
the subsequent QMC integration, not something the final integral's
correctness depends on -- any valid pivot choice, correctly decomposed,
gives a statistically valid estimate. Bit-identical *intermediates* was
simply the wrong invariant to assert: even SciPy's own algorithm isn't
guaranteed to reproduce this exact pivot choice across different BLAS
backends at this kind of degenerate input, so holding a from-scratch port
to a stricter bar than SciPy holds itself to was the actual mistake in the
2026-07-14 entry above, not the port's math.

**Fix**: split the one overreaching test into two in `tests/test_stats.py`:
`test_permuted_cholesky_numba_matches_scipy_exactly_for_typical_boxes`
(unchanged bit-exactness assertion, restricted to a moderate z0 range where
the pivot search has a real signal to discriminate on -- still catches a
genuine implementation bug in the common case) and
`test_permuted_cholesky_numba_agrees_with_scipy_on_final_probability_at_saturated_boxes`
(the full z0 range including saturating tails, asserting agreement on the
*downstream probability* via a shared-lattice/shared-shift `_qmvn_inner`
call, which is the invariant that actually matters -- includes the exact
worst case found as a permanent, non-random regression case, since random
draws essentially never reproduce a ~1-in-40000 divergence on their own).

No production code changed -- `_permuted_cholesky_numba` itself is
unmodified; this was purely a test-correctness fix. All 41 tests pass,
stable across repeated runs.

## 2026-07-15 — cuML backend validated on real GPU hardware: confirmed Pascal (compute capability < 7.0) silently returns wrong predictions

First real-GPU run of the cuML tier (previously unvalidated per this
module's own long-standing caveat) hit `_wrap_sklearn_like`'s first-batch
smoke test: `cuml backend predictions do not match the NumPy reference ...
(max_abs_diff=0.0498932)` on an NVIDIA TITAN X (Pascal).

Investigation, in order:
1. Read cuML's actual `SVMBase._attrs_from_cpu`/`_get_svm_model`/`predict`
   source (`svm_base.pyx`, fetched directly from the `rapidsai/cuml`
   GitHub repo at the installed 26.06.00 tag) looking for a dtype/attribute
   conversion bug in `from_sklearn`. Ruled out: `from_sklearn` explicitly
   forces `dtype=np.float64`; `n_support_`/`_gamma`/`intercept_`/
   `dual_coef_` all copy through correctly; all of pydreg's own exported
   model weights are float64 end-to-end (`scripts/pack_safetensors.py`
   loads `<f8`), so this wasn't a float32-precision issue either.
2. Enhanced `_wrap_sklearn_like`'s smoke-test failure to also run
   scikit-learn's CPU libsvm path (`to_sklearn_svr(dreg_model)`, already
   independently verified to agree with the NumPy reference to ~1e-9) on
   the exact same failing sample, and report whether it agrees or also
   diverges -- turning the smoke test into a triage tool rather than a bare
   pass/fail (`_sklearn_cross_check_detail` in `backend.py`, tests added in
   `tests/test_backend.py`). Result on the real failure: **sklearn agreed
   with NumPy to 5.5e-11** -- pinpointing the divergence specifically to
   cuML's own GPU path, not pydreg's conversion code or `DREGModel.predict`'s
   own math (ruling out the initial hypothesis that the expanded-form
   squared-distance formula was the culprit).
3. User re-ran the identical bigWig inputs on an A100 (compute capability
   8.0): no divergence, Jaccard > 0.999 vs. real dREG across multiple loci.
   Same cuml version (26.06.00) both times -- pinned the variable to GPU
   architecture, not data or software version.
4. Fetched RAPIDS's actual install docs and support-notice pages: **Pascal
   GPU support (compute capability 6.x) was removed in RAPIDS 24.02**, and
   the deprecation notice states outright: "Effective this release, use of
   a Pascal GPU will either fail or return invalid results." The TITAN X
   is compute capability 6.1 -- below the 7.0 minimum every cuML release
   since 24.02 requires. Also checked: `from_sklearn`/`as_sklearn` (PR
   #6102) only shipped in cuML 25.02 -- a full year *after* Pascal support
   was dropped, so no cuML version supports both; there's no older-cuML
   workaround for Pascal-class hardware here. (Pascal's own crippled FP64
   throughput would have made GPU accel a poor fit for this
   inherently-float64 SVR regardless.)
5. Confirmed directly: the user transferred the exact same bigWig files
   that failed on the TITAN X over to the A100 machine, and the same job
   ran with no deviation warning -- isolating the variable to the GPU
   itself, not anything data-dependent about that particular run.

**Fix**: `backend.py` now probes GPU compute capability directly
(`_cuda_compute_capability()`, via `cupy.cuda.Device().compute_capability`)
against a new `MIN_CUDA_COMPUTE_CAPABILITY = 70` constant. `detect_backend()`
falls back to NumPy (with a log message) on unsupported hardware in auto
mode; `build_scorer()`'s explicit `--backend cuml` path raises
`BackendUnavailable` with the specific compute-capability reason instead of
building a GPU model that would fail the smoke test downstream. The smoke
test itself is unchanged and remains the last line of defense for any other
backend-conversion issue. Documented in `docs/OPTIMIZATION.md`. Tests added
in `tests/test_backend.py` (13 backend tests, 47 total, all passing).

This is exactly the scenario `_wrap_sklearn_like`'s smoke test was built
for -- it caught a real, silent-wrong-answers hardware incompatibility
before it reached any output file.

## 2026-07-15 — experimental `cupy` backend tier (own branch, unvalidated)

Follow-up to the Pascal finding above: the actual incompatibility is
RAPIDS/cuML's own compiled SVM kernel dropping Pascal support in 24.02, not
CUDA/GPU compute in general. Researched two alternatives to routing scoring
through `cuml.svm`:

1. **Constructing a cuML SVM more directly from raw parameters** (skipping
   the `to_sklearn_svr()` round trip). Read cuML's `from_sklearn`/
   `_attrs_from_cpu` source again: there's no more-direct public path, and
   a private-API version of the same idea wouldn't fix anything -- the
   Pascal bug lives in cuML's compiled CUDA kernel itself, not in how the
   Python object gets built. Not pursued.
2. **Reimplementing the RBF SVR directly on GPU arrays, bypassing
   `cuml.svm` entirely.** `DREGModel.predict` is already just chunked array
   ops (`@`, elementwise square/sub/div, `exp`, `sum`) -- CuPy mirrors
   NumPy's API closely enough that this ports with almost no changes.
   Checked both realistic array libraries for Pascal compatibility:
   **CuPy still supports compute capability >=3.0** (no deprecation
   signal), while **PyTorch dropped compute capability 6.1 in 2.8** (works
   on <=2.7 only, per user reports on a GTX 1080 Ti). CuPy is also already
   an unavoidable transitive dependency of `cuml-cu12`, so this doesn't add
   a new dependency for existing `pydreg[gpu]` users. Went with CuPy.

**Implementation** (`src/pydreg/backend.py`): `_build_cupy_predict_fn`
ports `DREGModel.predict`'s exact formula (same expanded squared-distance
trick, same SV-chunking loop) onto `cupy.asarray`-wrapped device arrays,
returning a `predict_fn(X_scaled) -> y_scaled` matching
`_wrap_sklearn_like`'s expected interface exactly -- so the cupy tier reuses
that same wrapper (scaling/unscaling + first-batch smoke test) for free,
with zero new wrapping code. Added `"cupy"` to `DEFAULT_QUERY_CHUNK`
(reusing the NumPy tier's `4096` default rather than cuML's `2**20`, since
this tier's own Python code materializes the `(query_chunk, sv_chunk)`
intermediate directly on GPU the same way NumPy does on CPU -- cuML's
`2**20` default assumed its internal C++ tiling, which this tier doesn't
have). `--backend cupy` added to the CLI; `cupy-cuda12x` added explicitly
to the `gpu` extra in `pyproject.toml` (previously only an implicit
transitive dependency via `cuml-cu12`).

Because it's the same formula as the already-validated NumPy tier and not
a separate from-scratch kernel, there's no independent implementation for
it to disagree with -- unlike the cuml/sklearn tiers, which depend on a
private-attribute conversion step and a genuinely separate kernel-eval
codebase (libsvm/cuML's own C++) agreeing with `DREGModel.predict`.

**Explicitly unvalidated**: written and tested entirely on a machine with
no GPU. `tests/test_backend.py` covers the chunking/formula logic against
NumPy arrays standing in for CuPy ones (`_build_cupy_predict_fn` matches
`DREGModel.predict` to ~1e-10 on a tiny synthetic SVR, both single-chunk and
multi-chunk-over-support-vectors), and `build_scorer(..., "cupy")` wiring,
but this says nothing about real GPU throughput, memory behavior, or
whether CuPy's own compiled ops have some other hardware quirk. Per the
user's request, this work lives on its own branch (`cupy-svr-backend`, off
`main` post-#2) specifically so it can be dropped cleanly if real-hardware
testing shows it's slower than `cuml` or doesn't work at all -- not merged
into `detect_backend()`'s auto-selection, only reachable via an explicit
`--backend cupy`. 51 tests pass (17 in `test_backend.py`).

## 2026-07-15 — cupy tier: correct on real hardware, but slow -- fused the elementwise glue, made batch size tunable

First real-GPU report on the cupy tier from the 2026-07-15 entry above:
no smoke-test divergence (formula/wiring confirmed correct), but slower
than wanted. Two independent levers identified and implemented, in the
order they matter:

1. **Kernel fusion.** The two matmuls (`X @ SV.T`, `K @ coefs`) are already
   cuBLAS GEMM calls -- near-optimal already. The elementwise formula
   between them (`exp(-gamma * (sq_x + sq_sv - 2*cross))`) was ~5 separate
   elementwise kernel launches, each round-tripping a full
   `(query_chunk, sv_chunk)` array through GPU global memory -- pure
   memory-bandwidth overhead on what's fundamentally a memory-bound step
   (same underlying reason the NumPy tier itself is memory-bandwidth-bound;
   see the "Batching" section of `docs/OPTIMIZATION.md`). Wrapped it in
   `cupy.fuse()`, which JIT-compiles the whole chain into a single kernel
   reading its inputs once and writing `K` once -- same formula, no
   precision change, ~5x less memory traffic for that step. Bonus: it also
   eliminates one live `(query_chunk, sv_chunk)`-shaped buffer (the old
   separate `sqdist` intermediate no longer exists as its own array), which
   freed up enough headroom to raise `_build_cupy_predict_fn`'s default
   `sv_chunk` from `20_000` to `32_768` without exceeding the pre-fusion
   tier's peak memory footprint.
2. **Made batch size tunable rather than guessing at a fixed value.**
   Unlike `cuml.svm` (tiles the kernel matrix internally in C++, never
   materializing the whole thing), this tier's own Python code
   materializes the `(query_chunk, sv_chunk)` intermediate directly -- so
   for a fixed GPU memory budget, total kernel-launch iterations scale as
   `total_queries * n_sv * 8 bytes / budget`, independent of how the budget
   is split between the two chunk dimensions. Growing the budget (either
   knob) is what reduces iteration count; rebalancing an unchanged budget
   between the two dimensions does nothing. Added `cupy_sv_chunk` to
   `backend.build_scorer()`/`pipeline.run()` and `--cupy-sv-chunk` to the
   CLI (mirroring `--cuml-query-chunk`'s existing pattern) so the actual
   ceiling can be found empirically per-GPU rather than guessed at from a
   machine with no GPU at all.

Also researched and explicitly deferred a third, bigger lever: **float32
instead of float64**. This SVR is inherently float64 (exported that way
from R); the smoke test's `rtol=atol=1e-4` tolerance would likely still
pass at float32, but arithmetic-precision changes need their own explicit
validation pass, unlike scheduling/fusion changes which are provably
equivalent. Flagged as particularly relevant here since consumer Pascal
GPUs (the exact hardware this tier exists for) have ~1:32 float64:float32
throughput -- likely the single largest lever available, but not attempted
without a way to validate it.

Tests: `tests/test_backend.py`'s fake cupy module gained a `fuse` stand-in
(calls the decorated function directly -- these tests exercise
`_build_cupy_predict_fn`'s formula/chunking logic on NumPy, not cupy's own
JIT, which needs a real GPU to mean anything) and a new test confirming
`cupy_sv_chunk` threads from `build_scorer()` through to
`_build_cupy_predict_fn`. 52 tests pass (18 in `test_backend.py`). No
production math changed -- both levers are scheduling/batching changes
only, verified against the same NumPy reference as before.

## 2026-07-15 — cp.fuse() introduced a real ~3.5e-4 divergence on real GPU hardware; switched to cupy.ElementwiseKernel

The fusion change above (`cp.fuse()`) tripped `_wrap_sklearn_like`'s own
smoke test on real hardware: `max_abs_diff=0.000353735` against the NumPy
reference, with sklearn (CPU libsvm, on the same sample) agreeing with
NumPy to ~5.5e-11 -- pinning the divergence specifically to the fused cupy
path, the same triage pattern the smoke-test cross-check was built for back
in the original Pascal investigation, now paying off a second time on a
bug of pydreg's own making rather than cuML's.

`cp.fuse()` was the only change between the last known-good state and this
one, so it's the confirmed cause -- but *why* couldn't be pinned down
without GPU access to test either hypothesis directly. Two suspects:

1. `gamma` was passed as a runtime Python-float **argument** into the
   fused function. `cp.fuse()`'s JIT tracer infers argument types at
   first call and compiles a fixed kernel from that -- it may not apply
   the same dtype-promotion guarantees eager CuPy array ops do for a bare
   Python scalar mixed with float64 arrays.
2. `n_sv = 605,187` isn't divisible by `sv_chunk = 32,768`: the last of 19
   chunks per predict() call has a different (smaller, 15,363-wide) shape
   than the other 18. If `cp.fuse()`'s kernel cache keys loosely on shape
   across repeated calls within the same chunking loop, a stale cached
   kernel could plausibly mishandle the differently-shaped last chunk.

Rather than debug fuse's internals blind, switched to
`cupy.ElementwiseKernel` -- CuPy's older, far more battle-tested mechanism
for exactly this "broadcast several arrays into one elementwise formula"
pattern (used throughout cupy's own internals). It sidesteps both suspects
at once: every argument's dtype is declared explicitly in the kernel
signature (`"float64 cross, float64 sq_x, float64 sq_sv"` -- no promotion
ambiguity possible), and `gamma` is baked directly into the kernel source
as a literal (`f"out = exp(-{gamma!r} * ...)"`) rather than passed through
at all, removing it as a per-call argument entirely. `ElementwiseKernel`
also has no shape-based tracing/caching the way `fuse` does -- it's one
compiled kernel, invoked generically for any broadcastable shape, so
suspect 2 can't apply either.

Same formula, same fusion benefit (one kernel launch, one read of each
input, one write of `K`) -- just via a different, more explicit mechanism.
`tests/test_backend.py`'s fake cupy module swapped its `fuse` stand-in for
a `_FakeElementwiseKernel` that `eval()`s the kernel's operation string
directly (valid since the CUDA C expression here happens to be
syntactically identical Python) against NumPy arrays -- still can't catch
this specific class of bug (it lives entirely in cupy's own kernel
compilation, not in pydreg's formula/wiring), which is exactly why real
end-to-end hardware validation caught it and the NumPy-standin unit tests
didn't, and why this kind of change needs that real validation loop before
being trusted. 52 tests pass.

## 2026-07-15 — cupy tier downcast to float32; cuml tier deliberately not touched

Motivated by confirming that the current pretrained dREG models are
trained via Rgtsvm (not `e1071` -- `e1071` is now purely an
S3-compatibility shim around it), and that Rgtsvm's own CUDA
implementation has no double-precision path at all (traced through
`Rgtsvm.cpp`/`svm.hpp`/`cuda_helpers.hpp`/`configure.ac` on GitHub -- see
`docs/OPTIMIZATION.md`'s cupy-tier section for the full citation trail).
The exported alphas/support-vectors/rho were themselves produced by a
float32-limited training process, so their real accuracy ceiling was
already float32 before ever reaching pydreg's float64 storage -- inference
at float32 doesn't trade away precision the model actually has.

Also checked libsvm's actual source (`cjlin1/libsvm`) to confirm the
NumPy/scikit-learn tiers should NOT follow suit: `svm_node.value` is
`double`, and `Kernel::k_function`/`svm_predict_values` (the predict path)
operate in `double` throughout. `Qfloat` (`typedef float`) exists but only
for the training-time kernel cache, never at predict time. So `e1071`'s
CPU inference has always been genuinely double-precision, and pydreg's CPU
tiers already match that -- downcasting them would be a real fidelity
regression with no corresponding hardware motivation (the FP64-throughput
problem is GPU-specific).

**Implemented for `cupy`**: `_build_cupy_predict_fn`'s two GEMMs and fused
RBF kernel now run in float32 (`SV`/`coefs`/`X` cast to `cp.float32`, the
`ElementwiseKernel`'s types changed to `float32`, gamma's literal in the
kernel source given an explicit `f` suffix -- an unsuffixed float literal
is `double` in CUDA C, which would have silently promoted the whole
expression back to double and defeated the point). `y_scaled` still
accumulates in float64 (each chunk's small per-query contribution is
upcast before adding) as cheap insurance against cross-chunk summation
error, independent of the GEMM/kernel precision itself.

Tests: `tests/test_backend.py`'s fake `ElementwiseKernel` needed two fixes
to keep exercising this on NumPy -- the kernel body is CUDA C, not Python,
so `expf(...)` (not a Python name) and `f`-suffixed float literals (not
valid Python syntax) both get normalized away before `eval`. Tolerances in
the four cupy-path tests loosened from `atol=1e-10`/`1e-8` to `1e-5`,
reflecting genuinely-expected float32 rounding rather than a bug -- same
reasoning as the earlier Cholesky pivot-order tolerance fix, not a
weakening of the safety net. 52 tests pass.

**Deliberately not implemented for `cuml`**: investigated whether
`cuml.svm.SVR.from_sklearn()` could be coaxed into float32 and concluded
it can't be done safely without real hardware to validate against.
`from_sklearn(cls, model)` takes no dtype parameter at all
(`cuml/internals/base.pyx`), and `dtype` is only ever set internally
during `.fit()`/`cpu_to_gpu()` (hardcoded to `np.float64` when converting
from a CPU model, per the earlier Pascal investigation's source read of
`_attrs_from_cpu`). A working float32 path would require bypassing
`from_sklearn()` and manually replicating that private attribute-setting
logic -- and `_get_svm_model()` picks its C++ template
(`SvmModel<float>`/`SvmModel<double>`) from the `dtype` flag, then
raw-pointer-reinterprets the array's memory accordingly. Getting the flag
and the actual underlying array dtype out of sync wouldn't just lose
precision, it would read the wrong bytes entirely -- a real risk with zero
way to catch it without a GPU. Recommended `cupy` as the float32 GPU path
instead, since it's the same math and already correct.

## 2026-07-15 — float32 tripped the smoke test as expected; loosened its tolerance for cupy specifically, with a real number behind it

The float32 downcast above hit `_wrap_sklearn_like`'s own smoke test on
real hardware: `max_abs_diff=0.000227605` against the NumPy reference,
with sklearn (CPU libsvm) on the same sample agreeing with that reference
to ~5.5e-11 -- same triage pattern as the original Pascal investigation,
this time confirming the divergence is cupy's own float32 arithmetic
rather than a bug.

Root-caused rather than just accepted: the expanded-form squared-distance
formula (`sq_x + sq_sv - 2*cross`) is a textbook catastrophic-cancellation
setup for nearby feature vectors -- two large, nearly-equal terms whose
small difference is what actually matters. Negligible in float64 (~15-16
significant digits, losing a few to cancellation still leaves plenty), but
float32 only has ~7 to begin with, so the same absolute cancellation
consumes a much larger fraction of the available precision. Checked
whether upcasting the subtract/exp step to float64 would help (cheap,
since that step is memory-bound not compute-bound, so it wouldn't cost the
FP64 GEMM throughput penalty this whole tier exists to avoid) -- it
wouldn't: `cross`'s own rounding error is already present the moment the
float32 GEMM produces it, so promoting arithmetic *after* that point can't
recover precision already lost. A real fix would need a fundamentally
different mixed-precision GEMM technique (e.g. FP32 accumulation with an
FP32 residual-correction pass); not attempted, no hardware to validate it
against.

Naive "float32 epsilon accumulated over 605,187 independent terms" napkin
math lands around ~1e-4 -- the observed 2.28e-4 is consistent with that
order of magnitude plus the cancellation amplification on top, not
wildly larger in a way that would suggest a second, unrelated bug.

**Fix**: `_wrap_sklearn_like` gained `rtol`/`atol` parameters (previously
hardcoded to `1e-4`/`1e-4`). `build_scorer()` now passes a new
`CUPY_SMOKE_TEST_ATOL = 5e-4` constant for the `cupy` tier specifically --
`sklearn`/`cuml` keep the default `1e-4`, since both are genuinely
float64 and nothing about their expected precision changed. The new
constant's value comes directly from the real measured divergence plus
margin, not a guess -- same principle as every other tolerance decision
this project has made (e.g. the `_permuted_cholesky` pivot-order test
fix): characterize the real mechanism first, confirm it's not masking an
actual bug, then loosen deliberately with a documented number behind it.

Tests: two new cases in `tests/test_backend.py` place a smoke-test
divergence at a precise, known distance from the reference (via a
`_build_cupy_predict_fn` wrapper that adds a fixed offset, since the
NumPy-standin fake doesn't reproduce real float32 rounding) -- one in the
1e-4..5e-4 band (must not raise, proving the looser tolerance actually
applies to `cupy`), one past 5e-4 (must still raise, proving the looser
tolerance doesn't disable the check entirely). 54 tests pass.

## 2026-07-15 — float32 exposed feature extraction as the new bottleneck; overlapped it with GPU scoring via a one-chunk prefetch

Real-hardware confirmation of the float32 result above: TITAN X throughput
went 794 -> 9732 pos/s (~12.3x). Checked the implied FLOPS both times as a
sanity check -- 794 pos/s implies ~346 GFLOPS (matches the TITAN X's
~343 GFLOPS FP64 ceiling almost exactly, the same number from the original
Pascal investigation); 9732 pos/s implies ~4.2 TFLOPS, roughly 40-45% of
the card's ~9.5-11 TFLOPS FP32 peak -- a reasonable real-world GEMM
efficiency, well short of a naive 32x (FP64:FP32 ratio) because the
elementwise/memory-bound portion of the kernel doesn't scale the same way,
and because of the bottleneck found next.

Per the earlier prediction (this was flagged as a real risk before there
were numbers to confirm it, back when discussing whether a P100 would
even be GPU-bound): `nvidia-smi` on this same TITAN X run now showed
utilization cycling 0-90% between chunks, instead of staying maxed out.
Classic Amdahl's-law consequence -- cutting the GPU-bound step's cost by
~12x exposed the *other* per-chunk step, CPU-bound feature extraction
(bigWig I/O + binning), as the new relatively-larger bottleneck. Same
issue independently observed on an A100 running `cuml` (unrelated to the
float32 work, confirming this is backend-agnostic, not cupy-specific) --
`_score_positions`'s extract-then-predict loop was always strictly
sequential, it just didn't matter while the GPU step was slow enough to
dominate regardless.

**Fix**: `pipeline._score_positions` now runs a one-chunk-ahead prefetch --
a single background thread (`ThreadPoolExecutor(max_workers=1)`) extracts
the *next* chunk's features while the main thread's `scorer.predict()`
call blocks on the current chunk's GPU work. Relies on `scorer.predict()`
releasing the GIL during that block (true of CuPy/cuML's device-sync
calls) for the overlap to actually help; harmless either way on CPU
backends. Safe specifically because only one thread ever touches the
bigWig readers at a time -- the main thread never reads a bigWig while a
background extraction is in flight, and `max_workers=1` guarantees at
most one extraction is ever in progress. Pure scheduling change: same
extraction/scoring calls, same inputs, same order, so no output change is
expected. `_score_positions`'s per-chromosome nested loop was flattened
into `_iter_score_chunks`, a flat generator across every chromosome
group -- needed so the prefetch has one uniform "next chunk" boundary to
reason about, including the last-chunk-of-one-chromosome ->
first-chunk-of-the-next-one transition, not two different loop shapes to
special-case.

Tests: `tests/test_pipeline.py` gained three cases -- `_iter_score_chunks`
flattening across chromosomes/chunk boundaries, `_score_positions`
producing the exact expected result for a checkable (not just
zeros-shaped) fake scorer across multiple chunks and chromosomes, and a
deterministic overlap-guarantee test (using `threading.Event`s to prove
the second chunk's extraction actually starts *while* the first chunk's
predict() is still blocked, not just that the whole call eventually
finishes correctly -- ran 5x locally with no flakiness). 57 tests pass.

**Confirmed on the same real TITAN X**: GPU utilization now runs
consistently 90-100%, up from cycling 0-90% before this fix -- the
prefetch is doing what it was supposed to. Given that, the remaining gap
is small enough (well under 10%) that a full port of feature extraction
to GPU (discussed as a possible follow-up, see the cupy-tier-vs-cuml
scoring conversation) isn't obviously worth its cost right now -- that
would be a genuinely bigger, riskier change (features.py is shared by
every backend, would need a dual NumPy/CuPy path, and would need its own
correctness validation the same way the RBF kernel did) for a much
smaller remaining return than this fix already captured.

**Update, same day**: a closer look at the utilization graph showed it's
actually spikier than "steady 90-100%" -- and, notably, landing in a
similar 50-80% band on *both* the TITAN Xp and an A100. That cross-hardware
consistency is itself informative: it argues against a GPU-architecture-
specific explanation (e.g. Pascal's weak FP64, irrelevant to the A100) and
for the CPU-extraction-throughput hypothesis instead -- if feature
extraction's throughput is now the shared limiting factor, any GPU fast
enough to be waiting on it should show a similar pattern regardless of
its own architecture.

Mechanism: the prefetch is one chunk deep with a single background
worker. If per-chunk extraction now takes *longer* on average than a
chunk's `scorer.predict()` (very plausible after cutting predict's cost by
~12x while extraction is unchanged), every cycle looks like: GPU busy
during predict(), then idle while the main thread waits for that cycle's
already-behind extraction -- a spiky trace, not a plateau. Deepening the
prefetch queue with the same single worker wouldn't fix this: a bigger
queue only smooths *variance* between chunks, it can't fix a *systematic*
throughput mismatch where one worker's total extraction rate is below the
GPU's now-much-faster consumption rate. The two real levers would be
bigger chunks (spreads `_extract_features_cluster`'s fixed per-chunk setup
cost over more positions, though not guaranteed to fix a rate mismatch if
both stages scale ~linearly with chunk size) or parallelizing extraction
across multiple background workers (directly addresses a throughput
mismatch, but reopens a correctness question the single-worker design
deliberately avoided: whether pybigtools bigWig reads are safe to run
concurrently from multiple threads against the same reader object --
unverified, and getting it wrong would be a real bug, not just a
performance one).

Rather than guess further from a utilization percentage, added direct
timing instead: `_score_positions` now accumulates `extract_seconds`
(summed on the background worker) and `predict_seconds` (timed on the main
thread) and logs both once at the end of the call (INFO level, one line,
not a repeated progress log -- explicitly not the kind of noise just
demoted to DEBUG elsewhere in this same session). Since the two run
concurrently by design, they intentionally don't sum to the call's wall
time -- the log message says so directly, to head off reading them as a
serial breakdown. New test in `tests/test_pipeline.py` injects real
`time.sleep()` delays into fake extract/predict functions and asserts the
logged numbers reflect them (loose lower bounds, since this is real
wall-clock timing, not a mocked clock; ran 3x locally with no flakiness).
58 tests pass. Next real run's log line will give the actual
extract-vs-predict ratio instead of inferring it from `nvidia-smi`.

## 2026-07-15 — real extract-vs-predict numbers from TITAN Xp and A100; researched (but didn't implement) parallel extraction

Real log lines from the new instrumentation, both cards, all three
`_score_positions` call sites:

TITAN Xp:
- informative positions: 196.90s extract / 733.57s predict
- gap-filled positions: 69.33s extract / 14.46s predict (reversed)
- 10bp-densified positions: 154.40s extract / 534.75s predict
- aggregate: 420.63s extract / 1282.78s predict -- 3.05x predict:extract

A100:
- informative positions: 193.81s extract / 338.67s predict
- gap-filled positions: 58.15s extract / 6.65s predict (reversed)
- 10bp-densified positions: 153.29s extract / 245.90s predict
- aggregate: 405.25s extract / 591.22s predict -- 1.46x predict:extract

Two of three steps are predict-dominated on both cards (extraction mostly
hides behind the GPU wait, confirming the prefetch fix works as intended).
`gap-filled positions` is a real, isolated exception on both cards --
extraction dominates there specifically, most likely because those points
are scattered into sparse gaps by construction (`peaks.find_gap_infp`),
defeating `_extract_features_cluster`'s shared-fetch batching (which only
pays off for nearby, clusterable points). Small in absolute terms either
way: ~5-10% of total scoring time on these runs.

The more important signal: the *aggregate ratio* dropped from 3.05x
(TITAN Xp) to 1.46x (A100) -- extraction time barely moved between cards
(CPU-bound, GPU-independent: 196.90->193.81s, 69.33->58.15s,
154.40->153.29s), while predict time shrank substantially on the faster
card, so the same fixed CPU cost is a growing fraction of the total as
scoring gets faster. Rough wall-clock-with-overlap estimates (predict time
plus a small first-chunk lag, for the two predict-dominated steps, plus
extract time plus a small lag for the reversed one): ~1346s on the TITAN
Xp (~21% saved vs. fully serial 1703s) and ~651s on the A100 (~35% saved
vs. fully serial 996s). A full fix to the gap-fill step's poor overlap
would only add ~6-10% more on top of either -- not enough to justify
implementing it today, but the compressing-ratio trend (as GPUs get
faster, or once `cupy`+float32 is the default path everywhere) is worth
tracking rather than dismissing outright.

**Researched (not implemented) whether extraction could safely run across
multiple background workers if that trend continues.** Read pybigtools'
actual Rust source (`jackh726/bigtools`, `pybigtools/src/reader.rs`):
`BBIReader.values()`/`.intervals()`/`.zoom_intervals()` all take `&mut
self`, and since `BBIReader` is a normal `#[pyclass]` (not `unsendable`,
unlike its iterator types), PyO3 wraps it with a runtime borrow-check cell
enforcing Rust's aliasing rules independent of the GIL -- two threads
calling a read method concurrently on the *same* `BBIReader` object would
raise `PyBorrowMutError` (a safe, loud failure, not silent corruption, but
still unusable concurrently). The underlying reader types are generic
over `CachedBBIFileRead<ReopenableFile>`, built specifically around a
`Reopen` trait meant for independent handles onto the same file -- so the
safe pattern, if ever implemented, is one independently-opened `BBIReader`
per worker thread (`pydreg.io.open_bigwig()` is a thin wrapper around
`pybigtools.open()`, trivial to call once per worker) rather than sharing
`pipeline.run()`'s single `bw_plus`/`bw_minus` pair across threads.

Decision: document this (both the real numbers and the thread-safety
research) and stop here for now -- the current numbers don't justify
implementing parallel extraction, and this write-up means that decision
doesn't need re-deriving if a future run's ratio tips further. See
`docs/OPTIMIZATION.md`'s "Overlapping feature extraction with scoring"
section for the reader-facing summary.

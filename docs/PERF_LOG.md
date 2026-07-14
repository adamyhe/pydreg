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
verified to agree to ~1e-10), but `sklearn.svm.SVR.predict()` (libsvm) is a
single-threaded C loop over support vectors, while `DREGModel.predict`'s
chunked `X_scaled @ sv_block.T` / `K @ coefs` dispatches to a multithreaded
BLAS (Accelerate, on this Mac).

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

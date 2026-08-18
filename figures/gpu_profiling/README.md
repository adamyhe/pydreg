# GPU profiling: dREG (Rgtsvm) vs pydreg (cupy), SVR scoring of informative positions

Methodology for comparing GPU behavior during the SVR-scoring step specifically
(not the informative-position screen itself, which is CPU-only in both
implementations -- no GPU involved before scoring starts).

## Hypotheses under test

1. **Poor host/device overlap** (scheduling, not kernel quality): each scoring
   chunk does host feature-prep -> transfer -> kernel -> transfer-back
   serially, so the GPU sits idle between chunks. pydreg hit exactly this
   before its extract/predict prefetch fix -- see `docs/PERF_LOG.md`'s
   2026-07-14/15 entries, where utilization on both a TITAN Xp and an A100
   cycled between busy-during-predict and idle-during-the-next-chunk's-CPU-work.
   `eval_svm.R`/Rgtsvm scores chunks synchronously with no such overlap, so
   this is a live possibility for dREG's own reported "regular spikes up and
   down."
2. **Memory-bound kernels** (the original hypothesis): Rgtsvm's CUDA kernels,
   optimized around sparse SVM operations, do poorly-coalesced/high-traffic
   memory access even while busy, as opposed to pydreg's dense chunked-matmul
   kernel (`docs/OPTIMIZATION.md`).

These look similar in a coarse utilization trace (both show "not pegged at
100%") but need different evidence and imply different fixes, so the
methodology below is built to tell them apart rather than just confirm "yes,
utilization differs."

## Findings (2026-08-18)

Both hypotheses were confirmed, on real hardware, as two separate and
independently-measured mechanisms -- neither alone explains the whole gap.
Full narrative in `docs/PERF_LOG.md`'s 2026-08-18 entry; summary here:

- **Scheduling gap (hypothesis 1), confirmed from source and from the
  trace.** `eval_svm.R`'s GPU path serializes CPU-parallel feature
  extraction (a *freshly spun-up and torn-down* snowfall cluster every
  round) against a single big `Rgtsvm::predict.gtsvm` call, once per round
  of `ncores` batches, with zero overlap between rounds. Real `nsys`
  `cuda_gpu_trace` idle-gap analysis found exactly `n.loop - 1 = 7` large
  gaps (`[93.8s, 92.9s, 92.9s, 92.8s, 92.3s, 92.0s, 21.3s]`) matching the
  round count predicted from source precisely -- ~660s of a 2,431s
  nsys-covered span (27%) is pure GPU idle time from this alone.
- **Kernel-design gap (hypothesis 2), confirmed and sharper than
  expected -- numbers below are phase-exact** (both sides restricted to
  just the informative-positions phase, via the `start_epoch` windowing
  fix; see "nsys and windowing" below). For the identical 5,617,218
  positions: Rgtsvm's `GTSVM::CUDA::SparseEvaluateKernelKernel256` launches
  **653,677** times (~2.2ms each) vs. pydreg's `sgemm_128x128x8_NT_vec`
  (cuBLAS) in **26,281** calls (~11.4ms each) -- **~25x** fewer kernel
  launches for the same work. Memory transfers are starker still: dREG
  issues **1,345,274** separate Host-to-Device memcpys averaging **2.3us**
  each (clearly launch-overhead-dominated) vs. pydreg's **1,384** H2D
  memcpys averaging **478.5us** each -- **~972x** more transfer calls for
  only ~4.7x more total H2D time. Textbook confirmation of "many small
  sparse-oriented operations" vs. "few large batched dense operations."
- **Cleanest single number**: restricting to time the GPU is *actually
  doing something* (excluding the scheduling gap above entirely), dREG's
  total GPU-busy time is 1,771.0s vs. pydreg's 415.1s for the identical
  workload -- **~4.27x**, the kernel-design gap in isolation. dREG loses
  27.1% of its own window to scheduling idle time; pydreg loses only 5.4%.

(The first pass at this comparison used pydreg numbers spanning its whole
run, not just this phase, giving smaller ratios -- ~17x/~666x. Every
number above supersedes those; see `docs/PERF_LOG.md`'s two 2026-08-18
entries for the full before/after.)

Two figures, both regenerable from the real profiling data in `gpu_out/`:
`plot_gpu_utilization.py` (`figures/plots/gpu_utilization.svg`) is the
qualitative one -- dREG's sawtooth vs. pydreg's plateau, real dmon data,
own x-axis per panel since the phases differ ~6.5x in duration.
`plot_gpu_time_breakdown.py` (`figures/plots/gpu_time_breakdown.svg`) is
the quantitative one -- stacked idle/kernel/memcpy time per tool, with
kernel-launch and H2D-memcpy call counts as direct text annotations
(their *time* is too small a fraction of either bar to encode by height,
even though their *count* is one of the two headline findings).

## Isolating the right process/phase, with zero source modification

- **dREG**: run `run_predict.bsh` (the legacy score-only workflow -- steps 1
  (`get_informative_positions`), 2 (feature extraction), 4 (`eval_svm`
  scoring) per `CLAUDE.md`; no gap-filling/densification/peak-calling). That
  process's entire GPU footprint already **is** the informative-positions
  scoring step -- no windowing needed. (Not `run_dREG.bsh`, which also scores
  gap-filled and densified position sets in the same process, the same
  ambiguity pydreg has -- see below.)
- **pydreg**: run the normal full pipeline (`pydreg plus.bw minus.bw out -v
  --backend cupy`) and slice the profile down to just the "scoring
  informative positions" phase using `pydreg`'s own existing `-v` INFO log
  lines (`"scoring informative positions..."` / `"scoring informative
  positions done in Xs"`) -- these already bracket exactly the right window,
  with real wall-clock timestamps, with no code changes.

Both invocations should run against the **same bigWig pair, on the same GPU**
-- cross-hardware or cross-input comparisons aren't meaningful here (see
PERF_LOG's own note that utilization patterns were compared across a TITAN Xp
and an A100 specifically to rule out an architecture-specific explanation;
do the same here if hardware access allows it).

## Running

```
./profile_gpu.sh OUTDIR dreg   -- bash run_predict.bsh plus.bw minus.bw dreg_model.RData out_prefix 16 0
./profile_gpu.sh OUTDIR pydreg -- pydreg plus.bw minus.bw out_prefix --backend cupy -v
python3 analyze_gpu_profile.py OUTDIR dreg pydreg --whole-trace dreg --gpu-index dreg=1,pydreg=0
```

(`--whole-trace dreg` tells the analyzer not to look for pydreg-style log
markers in dREG's log -- there aren't any, and there don't need to be, since
the whole process already is the target phase. `--gpu-index` is explained
below -- don't skip it on a multi-GPU host.)

### A real gotcha: CUDA's GPU index isn't nvidia-smi's GPU index

Confirmed on real hardware (a 2-GPU server): `run_predict.bsh`'s `[gpu_id]`
argument (`eval_svm.R`'s `Rgtsvm::selectGPUdevice(gpu_id)`) is a **CUDA**
device index, which is not guaranteed to match `nvidia-smi`'s enumeration
(CUDA defaults to its own ordering heuristic unless
`CUDA_DEVICE_ORDER=PCI_BUS_ID` is set; `nvidia-smi` always uses PCI bus
order). Concretely: `run_predict.bsh ... 16 0` (`gpu_id=0`) logged `GPU ID:
0` and genuinely ran on CUDA device 0 -- which turned out to be
`nvidia-smi`'s GPU **1**, not GPU 0. Trusting the argument you passed in
without checking would have profiled the wrong (idle) card entirely.

On top of that, `analyze_gpu_profile.py`'s dmon auto-detect (pick the GPU
with the most total `fb`/`sm` activity) is **only a fallback** and is
unreliable on a shared multi-tenant node -- confirmed on the same run,
where another process's memory footprint on an unrelated GPU outranked the
actual job's `fb` total and got auto-selected instead. Always pass
`--gpu-index LABEL=INDEX` pairs once you know which physical
(`nvidia-smi`-numbered) GPU each job actually landed on -- don't rely on
the argument you *passed* to the job, and don't rely on auto-detect on a
shared host.

Verify `run_predict.bsh`'s actual argument order/model path against your
checkout before running -- `_reference/dREG` isn't available in this sandbox
to confirm exact CLI syntax, only the documented pipeline structure in
`CLAUDE.md`/`docs/PLANNING.md`.

Each run produces, under `OUTDIR`:
- `LABEL.dmon.csv` -- `nvidia-smi dmon` samples (1 Hz, wall-clock timestamped):
  `sm` (Volatile GPU-Util), `mem` (memory-controller utilization -- not
  memory used), `fb`/`bar1` (memory used).
- `LABEL.nsys-rep` -- an `nsys profile` trace (CUDA + NVTX + OS runtime), if
  `nsys` is on `PATH`.
- `LABEL.log` -- the command's own stdout/stderr.

## What the analyzer reports, and which hypothesis each part addresses

- **dmon utilization series** (`sm`/`mem` mean/median/p10/p90, coefficient of
  variation, `%` of samples below 10%): directly answers "how spiky does
  Volatile GPU-Util look," matching what you already observed qualitatively.
  High coefficient of variation and a large "idle <10%" fraction confirm the
  spikiness quantitatively but **can't tell you which hypothesis is
  responsible** -- a synchronous host round-trip and a slow-but-continuous
  memory-bound kernel can both show up as "not pegged at 100%."
- **nsys idle-gap analysis** (`cuda_gpu_trace`, merged into busy intervals,
  gaps measured between them): this is what distinguishes the two hypotheses.
  - Regular, similarly-sized gaps between short GPU bursts (low
    `gap_coeff_of_variation`) -> hypothesis 1 (host round-trip per chunk, no
    overlap) -- the same signature pydreg's own prefetch fix eliminated.
  - GPU mostly continuously busy (`pct_busy` high, few/no gaps) but overall
    slower than pydreg for the same position count -> hypothesis 2 (the
    kernels themselves are the bottleneck, not scheduling) -- at that point,
    `cuda_gpu_kern_sum`/`cuda_gpu_mem_time_sum` (also collected) tell you
    whether time is dominated by actual SVM kernels or by memcpy/memset,
    which is the direct test of "heavy memory traffic."
  - If it's genuinely kernel-bound and memcpy isn't the story, the next step
    (not yet built here) would be Nsight Compute (`ncu`) on the specific
    Rgtsvm kernel(s) for achieved-occupancy/DRAM-throughput metrics -- adding
    that is worth doing only if this first pass actually points at the
    kernels rather than scheduling.

## nsys and windowing (resolved, with one caveat for old captures)

`nsys`'s own per-event timestamps in `cuda_gpu_trace` are nanoseconds since
*nsys itself* started recording, not wall-clock time -- there's no way to
compare them against `LABEL.log`'s wall-clock phase markers without
knowing the wall-clock instant nsys started. `profile_gpu.sh` now writes
that instant to `LABEL.start_epoch` (via `date +%s.%N`, captured
immediately before launching `nsys profile`), and `analyze_gpu_profile.py`
uses it to convert every `cuda_gpu_trace` event's timestamp into wall-clock
time and filter to exactly the log-derived window -- the same idea as the
dmon windowing, just needing one extra piece of information since nsys's
clock isn't wall-clock to begin with.

This means `nsys_cuda_gpu_kern_sum`/`nsys_cuda_gpu_mem_time_sum`/
`nsys_idle_gaps` are now correctly restricted to just the informative-
positions phase for pydreg too, not its whole run -- confirmed with a
synthetic fixture (in-window/out-of-window events at known relative
timestamps) before trusting it on real data. One real bug found and fixed
while building this: `pandas.Timestamp.timestamp()` assumes a naive
timestamp is **UTC**, while the stdlib's `datetime.timestamp()` (and
`date +%s.%N`, which writes `start_epoch`) assumes **local time** -- using
pandas' version directly would have silently mis-windowed by the local UTC
offset on any non-UTC host. Fixed by converting via
`Timestamp.to_pydatetime().timestamp()` instead.

**Caveat**: this only works for captures made *after* `profile_gpu.sh`
started writing `LABEL.start_epoch`. A `.nsys-rep` from before that change
has no matching `.start_epoch` file, so the analyzer falls back to
whole-trace nsys numbers for it automatically (prints a NOTE when this
happens) -- the 2026-08-18 findings in `docs/PERF_LOG.md` were computed
this way, before this fix existed, so pydreg's kernel/memcpy counts there
are its whole run, not just the informative-positions phase (the entry
already says so).

`cuda_api_sum` (CPU-side CUDA API call timing) is still whole-trace-only --
windowing it the same way would need the raw `cuda_api_trace` report
aggregated locally exactly like `cuda_gpu_trace` is now, not done here
since the kernel/memcpy numbers were what the scheduling/kernel-design
comparison actually needed.

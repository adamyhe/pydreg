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
  `cuda_gpu_trace` idle-gap analysis found exactly `n.loop - 1` large
  gaps, matching the round count predicted from source precisely, on
  every library where the prediction is checkable: 9 of the 12 match
  exactly (5 gaps for K562_groseq up to 10 for G5 and Jurkat_ChROseq),
  and the other 3 predict 21/11/12 gaps against the analyzer's 10-slot
  `largest_gaps_ns` cap, so they're consistent but not confirmable from
  the summary alone. Jurkat_PROseq as the worked example (113 batches ->
  8 rounds -> 7 gaps): `[70.9s, 70.3s, 70.2s, 70.0s, 69.8s, 69.4s,
  19.0s]` -- six full 16-batch rounds plus one short gap from the
  undersized final round of a single batch, 439.5s of pure idle inside a
  2,433.5s span. dREG loses a median **22.3%** of its own window (range
  20.0-26.6%) to this; pydreg loses **4.2%** (3.7-4.8%).
- **Kernel-design gap (hypothesis 2), confirmed and sharper than
  expected -- numbers below are phase-exact** (both sides restricted to
  just the informative-positions phase, via the `start_epoch` windowing
  fix; see "nsys and windowing" below). Per library, on identical
  position counts both sides: Rgtsvm's
  `GTSVM::CUDA::SparseEvaluateKernelKernel256` does **8.57-9.05**
  positions per launch (~2.48ms each) vs. pydreg's
  `sgemm_128x128x8_NT_vec` (cuBLAS) at **108.4-110.4** (~5.38ms each) --
  a median **12.4x** fewer kernel launches for the same work. Memory
  transfers are starker still: dREG issues one Host-to-Device memcpy per
  **4.16-4.41** positions, averaging **2.1us** each (clearly
  launch-overhead-dominated), vs. pydreg's one per **3,958-4,085**
  positions averaging **495.5us** -- a median **942x** more transfer
  calls for only ~4x more total H2D time. Textbook confirmation of "many
  small sparse-oriented operations" vs. "few large batched dense
  operations."
- **Cleanest single number**: restricting to time the GPU is *actually
  doing something* (excluding the scheduling gap above entirely), dREG's
  total GPU-busy time is **4.28-4.63x** pydreg's (median **4.47x**) for
  the identical workload -- the kernel-design gap in isolation. Notably
  tight across a 4x range of library sizes, and trending mildly *against*
  size (4.28x on G1's 17.1M positions, 4.63x on K562_groseq's 4.19M).

**Both per-operation gaps depend on pydreg's own batching defaults, so
quote them with those attached.** dREG's per-position rates are
tuning-free constants -- a median **0.2338** H2D memcpys and **0.1137**
dominant-kernel launches per position, varying only ~5% across the 12
libraries with data density. pydreg issues one H2D per *query* chunk and
`ceil(n_sv / sv_chunk)` sgemms per query chunk. So both ratios are
really statements about pydreg's chunk sizes:

| gap | formula | at the shipped defaults | scales with |
|---|---|---|---|
| memcpy | `0.2338 * query_chunk` | 950x (measured 942x) | `query_chunk` |
| kernel launch | `0.1137 * query_chunk / ceil(605,187 / sv_chunk)` | 12.5x (measured 12.4x) | `query_chunk` **x** `sv_chunk` |

That is why the kernel number read ~25x in earlier versions of this
README: it was measured at the retired `sv_chunk=32,768` (19 sv-chunks
per query batch) rather than the shipped 16,384 (37). Both are correct
for their configuration. The memcpy gap held steady across both captures
only because `query_chunk` stayed at 4,096 in each -- not because it's
immune.

**So state the configuration and report the measurement**, rather than
arguing about which ratio is intrinsic: *at pydreg's default batching,
pydreg does 942x dREG's work per H2D memcpy and 12x per kernel launch.*
Both figures say exactly that on their face -- the subtitle carries the
claim and a second line names the `query_chunk` and sv-chunks-per-batch
they were measured at, read out of the runs' own logs rather than
hardcoded, so the figure can't silently drift the way the ~25x number
did.

**Where only one ratio can be shown, use memcpy.** Not because it's
knob-free -- it isn't -- but because its order of magnitude survives the
whole plausible `query_chunk` range (~120x at 512 up to ~11,700x at
50,000, never under two orders of magnitude), whereas the kernel gap has
no such floor: at `query_chunk=512` with the shipped `sv_chunk` it
computes to ~1.6x, i.e. the claim all but disappears. A number a
plausible retune can erase shouldn't carry the argument.

(The first pass at this comparison used pydreg numbers spanning its whole
run, not just this phase, giving smaller ratios -- ~17x/~666x. Every
number above supersedes those; see `docs/PERF_LOG.md`'s two 2026-08-18
entries for the full before/after.)

**Scope**: the numbers above are the full 12-library sweep
(`profile_gpu_all.sh`, run 2026-09-03 on `cbsugpu01`), and every figure
below is drawn from it.

**Counts transfer between captures; times don't.** Jurkat_PROseq is the
one library captured both in the original 2026-08-18 pair and in the
sweep, on two different hosts three weeks apart. Its dREG launch counts
came back *bit-identical* (**653,677** dominant-kernel launches,
**1,345,274** H2D memcpys, on 5,617,218 positions) -- these counts are
deterministic properties of the algorithm, which is what makes the
per-operation claim above solid. Per-kernel *times* moved ~13% on the
different silicon (2.2ms -> 2.48ms for the same launch count), so
absolute times and time-ratios from one capture should not be compared
against the other's. See `docs/PERF_LOG.md`'s 2026-09-03 entry.

Three figures, all regenerable from the real profiling data in `gpu_out/`,
each drawing **one row per library** across whatever the sweep has
captured.

`plot_gpu_utilization.py` (`figures/plots/gpu_utilization.svg`) is the
qualitative one -- dREG's sawtooth vs. pydreg's plateau, real dmon data,
as a separate track per library per tool. Every track keeps its own
x-axis: dREG's phase runs several times longer than pydreg's on the same
library (~6.5x on Jurkat_PROseq) and libraries differ again on top of
that, so one shared axis would crush the short tracks into slivers and
destroy the very shape the panel exists to show. Each track states its own
duration and mean utilization instead.

`plot_gpu_time_breakdown.py` (`figures/plots/gpu_time_breakdown.svg`) is
the quantitative counterpart -- stacked idle/kernel/memcpy time, a bar
pair per library, on one shared time axis so both the within-library
dREG-vs-pydreg gap and the between-library size differences read directly.
Horizontal bars, because the category axis carries names like
`Jurkat_ChROseq_1` that would need rotating as vertical-bar x-labels.
Kernel-launch and memcpy call counts live in the tooltips rather than as
annotations here: memcpy's *time* is a sliver of every bar (well under
1%), and 24 bars' worth of count labels would swamp the figure.

Neither of those two actually states an inefficiency, though -- a symptom
(idle GPU) and raw totals side by side both still require the reader to do
the division themselves. `plot_gpu_efficiency.py`
(`figures/plots/gpu_efficiency.svg`) is that division, done for them: a
log-scale dot plot of *positions processed per GPU operation* (kernel
launch, H2D memcpy call), one dumbbell per library, with each library's
own informative-position count read from its own logs rather than divided
by a single hardcoded constant. dREG does ~12x/~942x less useful work per
individual operation than pydreg, which is the actual "inefficient" claim,
not just "the GPU is idle a lot" or "there are more calls." `--panels
memcpy` emits just the lower panel as `gpu_efficiency_memcpy.svg`, for
the technical note and anywhere else that should quote the gap whose
order of magnitude our own retuning can't erase (see Findings). Built
as a
dumbbell dot plot rather than a bar chart specifically because a log-scale
axis distorts bar *length* perceptually (the same reason
`figures/_common.py`'s scatter panels use log-scale point marks, not bars).

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

### The whole 12-library sweep

`profile_gpu_all.sh` drives the full benchmark -- both tools, all 12
libraries, plus the per-library analyzer pass -- and produces exactly the
label convention the figure scripts discover on their own:

```
BW_DIR=/path/to/bigwigs DREG_DIR=/path/to/dREG \
  NVSMI_GPU=$IDX ./profile_gpu_all.sh    # IDX from --report-gpus, see below
```

It runs strictly sequentially, which is not incidental: two profiled jobs
sharing a node contaminate each other's dmon samples and idle-gap
analysis, which is the entire measurement. Budget roughly **8-15 hours of
exclusive GPU time** for all 12 -- the Jurkat_PROseq reference pair was
~43min (dREG) + ~22min (pydreg) -- and check free disk first, since each
dREG capture traces hundreds of thousands of kernel launches and over a
million memcpys. The sweep is resumable: a library whose artifacts already
exist is skipped, so an interrupted run can just be re-invoked. Pass
library names as arguments to do a subset.

Set `NVSMI_GPU` only after confirming which physical card the jobs land
on -- see the gotcha section below. The cheapest way to confirm it is
after the fact, from a capture you already have:

```
python3 analyze_gpu_profile.py gpu_out --report-gpus
```

`nvidia-smi dmon` samples *every* GPU into the csv, so which card a run
used is recoverable from the capture -- `--gpu-index` is an analysis-time
selection, not a capture-time one, and getting it wrong never costs you a
re-capture. Leaving it unset falls back to the analyzer's auto-detect, which
is unreliable on a shared node.

**Don't copy an index out of this README.** The CUDA-to-nvidia-smi
mapping is per-host, and two real hosts have now disagreed:

| host | CUDA device passed to dREG | nvidia-smi index it actually was |
|---|---|---|
| original 2-GPU capture host | 0 | **1** (dREG 61.8% / pydreg 93.7% `sm`; GPU 0 ~1%) |
| `cbsugpu01` | 0 | **0** (dREG 70.7% / pydreg 96.2% `sm`; GPU 1 at 0.0%) |

Carrying the first host's `1` over to the second is exactly how you pin
an idle card: every dmon stat then describes a GPU that did nothing, and
before this was hardened it crashed the analyzer outright rather than
saying so. Always derive `IDX` from `--report-gpus` on the host you're
actually running on.

`NVSMI_GPU` is a *single* value because this comparison requires both
tools on the same physical GPU; a cross-hardware ratio would be
meaningless. The reason the underlying `--gpu-index` spec is *per-label*
at all is that CUDA's device index doesn't match nvidia-smi's -- not that
the two tools use different cards. `NVSMI_DREG`/`NVSMI_PYDREG` override
individually for a host where they genuinely do; setting them to
different values otherwise means you're about to compare two different
GPUs.

Then draw the figures (each writes its own standalone SVG to
`figures/plots/`, per this repo's one-script-per-panel convention):

```
python3 plot_gpu_utilization.py   --outdir gpu_out --gpu-index IDX
python3 plot_gpu_time_breakdown.py --outdir gpu_out
python3 plot_gpu_efficiency.py     --outdir gpu_out
python3 plot_gpu_efficiency.py     --outdir gpu_out --panels memcpy
```

The last one writes `gpu_efficiency_memcpy.svg`, the memcpy panel alone
as a standalone figure -- `--panels {both,kernel,memcpy}`, defaulting to
`both`. Every variant states the batching it was measured at in its own
subtitle, parsed from the pydreg run's log (`query_chunk`) and measured
from the trace (sv-chunks per batch = kernel launches per H2D memcpy),
so no default is hardcoded into a figure. It exists because the two panels aren't equally robust (see the
batching caveat in Findings): where only one per-operation number can be
shown, the memcpy gap is the one whose order of magnitude holds across
any plausible chunk-size setting. The single-panel output
is pixel-identical to that panel in the two-panel figure, just cropped
to its own canvas with the subtitle naming only the median it shows.

All three discover libraries by globbing for the
`dreg_<LIB>`/`pydreg_<LIB>` labels and skip (with a note) any library not
captured yet, so a partially-finished sweep still plots what it has.
`--libraries` restricts them to a subset.

### One library at a time

The sweep script is a loop around these three commands; run them directly
to profile a single library:

```
./profile_gpu.sh OUTDIR dreg_LIB   -- bash run_predict.bsh plus.bw minus.bw out_prefix dreg_model.RData 16 0
./profile_gpu.sh OUTDIR pydreg_LIB -- pydreg plus.bw minus.bw out_prefix --backend cupy -v
python3 analyze_gpu_profile.py OUTDIR dreg_LIB pydreg_LIB \
    --whole-trace dreg_LIB --gpu-index IDX
```

(`--whole-trace dreg_LIB` tells the analyzer not to look for pydreg-style
log markers in dREG's log -- there aren't any, and there don't need to be,
since the whole process already is the target phase. `--gpu-index` is
explained below -- don't skip it on a multi-GPU host.)

Note `run_predict.bsh`'s argument order: **plus, minus, out-prefix, model,
cores, gpu**. An earlier version of this README listed the model before the
out-prefix, which does not match what the real capture actually ran; verify
against your own checkout regardless.

### Captures that predate the label convention

The original 2026-08-18 capture used bare `dreg`/`pydreg2` labels, from
before the sweep existed. All three figure scripts can still read it by
naming the pair explicitly:

```
python3 plot_gpu_efficiency.py --pair Jurkat_PROseq=dreg:pydreg2
```

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
`--gpu-index` explicitly once you know which physical
(`nvidia-smi`-numbered) GPU the jobs actually landed on -- don't rely on
the argument you *passed* to the job, and don't rely on auto-detect on a
shared host. A bare `--gpu-index IDX` pins every label to nvidia-smi GPU
`IDX`,
which is the normal case here since both tools must share a card; the
`LABEL=INDEX` form is for the rarer host where they don't.

`run_predict.bsh`'s argument order is **plus, minus, out-prefix, model,
cores, gpu** -- confirmed from the real 2026-08-18 capture's own
`dreg.log`, which records the exact invocation, rather than inferred from
the pipeline structure documented in `CLAUDE.md`/`docs/PLANNING.md`. Verify
against your own checkout anyway before starting a multi-hour sweep;
`profile_gpu_all.sh` preflights the model/bigWig paths but can't check
dREG's own argument order for you.

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

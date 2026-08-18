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

## nsys and windowing (a real limitation)

`nsys` timestamps are nanoseconds since the profile started, not wall-clock
time, and no profile-start epoch is currently captured to translate the
log-derived wall-clock window into that timeline. So:

- For **dREG**, this doesn't matter -- the whole recorded trace already is the
  target phase.
- For **pydreg**, `nsys_cuda_gpu_kern_sum`/`nsys_cuda_gpu_mem_time_sum` in the
  analyzer's output reflect the **whole run** (informative + gap-filled +
  densified scoring combined), not just the informative-positions phase, even
  though the dmon utilization numbers above them are correctly windowed. The
  analyzer prints a note when this applies. The idle-gap analysis is affected
  the same way, so a mixed gap pattern across all three phases is expected --
  treat pydreg's nsys numbers as "how cupy behaves on this workload overall,"
  not a phase-exact match to dREG's isolated number, when comparing.

Two ways to close this gap later if the coarser comparison isn't conclusive
enough, in increasing order of effort:
1. `nsys`'s external `nsys start`/`nsys stop` attach-to-running-process
   commands can bracket an arbitrary window of an already-launched process
   from a separate watcher script tailing pydreg's log for the same two
   marker lines -- no source change, but exact flags vary across `nsys`
   versions and this hasn't been validated here (no GPU/`nsys` available in
   this environment); check `nsys launch --help`/`nsys start --help` on the
   actual server before relying on it.
2. An NVTX range around just that phase in `pipeline.py` would make `nsys`
   natively phase-aware (considered and reverted for this pass -- avoided
   touching `src/pydreg/` per your preference).

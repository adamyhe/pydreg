# 1.0 re-benchmark plan

Written for whoever runs the 1.0 benchmarks, including Claude Code agents
on cluster nodes. Read this file top to bottom before starting. The goal
is to re-measure current pydreg for (a) the README's Performance section
and (b) the updated bioRxiv technical note. The published numbers came
from older code: every release since then changed scoring (fused GPU
kernels in 0.4.0, the M5 TF32 fix), so they are stale.

## What to run, where

| # | hardware | script | libraries | modes | status |
|---|---|---|---|---|---|
| A | NVIDIA Titan Xp node (the one the published dREG baselines ran on) | `timing_scripts_cuda.sh` | all 12 | fp32 only (compute capability 6.1 has no TF32) | **done** 2026-11-23, see `docs/PERF_LOG.md` (2026-11-24). **Headline figures.** |
| B | NVIDIA A100 | `timing_scripts_cuda.sh` | Jurkat_ChROseq_1 | fp32 + tf32 | to do |
| C | NVIDIA L40S | `timing_scripts_cuda.sh` | Jurkat_ChROseq_1 | fp32 + tf32 | to do |
| D | Apple M5 | `timing_scripts_apple.sh` | Jurkat_ChROseq_1 | mlx + mlx-tf32 | **done**, see below |
| E | NVIDIA RTX 3080 (desktop) | `timing_scripts_cuda.sh` | Jurkat_ChROseq_1 | fp32 + tf32 | **done**, see below |

Run A is the only one that has to cover all 12 libraries, because it is the
only one with a like-for-like dREG baseline. B-D feed a cross-hardware table
(walltime, peak RSS, Jaccard vs dREG, TF32 trade-off).

**Don't re-run dREG** unless the Titan Xp node's hardware or software has
changed since the published dREG runs. The baselines are in the HF dataset
(`adamyhe/pydreg-supporting-data`, `dreg/`). If they have to be re-run, use
`timing_scripts_dreg_apptainer.sh`, which took about 19 hours for all 12 last time.

## Code version

Benchmark the code that will ship as 1.0: `main`, once PR #35 (these
scripts) is merged, or the `cuda-benchmark-scripts` branch until then. Its
`src/` is identical to `main`'s (`a8eae1f`, which includes the input check
and the `raw.peak.bed.gz` fix). Don't modify anything under `src/` on a
benchmark checkout. `env.txt` records the commit and flags uncommitted
`src/` changes, and a run made with them is not usable.

## Setup on a node

```bash
git clone https://github.com/adamyhe/pydreg && cd pydreg
git checkout main            # or cuda-benchmark-scripts, see above
uv sync --extra gpu          # CUDA nodes; on the Mac: uv sync --extra mlx
DATA_DIR=/path/to/fast/storage bash figures/timing_scripts_download.sh   # skip if the 12 bigWigs are already there
```

`timing_scripts_download.sh` needs `wget`, `bigWigMerge` and
`bedGraphToBigWig` (for the merged Jurkat_ChROseq). Put `DATA_DIR` on fast
local storage if the node has it. The perf log found cold or networked
reads up to 5x slower for feature extraction, and slow enough I/O can
leave the GPU waiting on input.

## Running (CUDA, runs A-C)

```bash
# A: Titan Xp, all 12 libraries, fp32
DATA_DIR=... bash figures/timing_scripts_cuda.sh
# B/C: A100 or L40S, one library, both precisions
DATA_DIR=... MODES="fp32 tf32" LIBS=Jurkat_ChROseq_1 bash figures/timing_scripts_cuda.sh
```

Defaults are `--cores 16 --seed 0`, matching the published runs. Keep
them. Don't pass any `--pmv-laplace-*`, chunk-size or other tuning flags.

**The idle check.** Before every run the script waits until the GPU is
idle (max utilization over 5 samples at most `IDLE_UTIL`=5%, memory in use
at most `IDLE_MEM_MIB`=2048) and the node's 1-minute load average is at
most `MAX_LOAD`=2. It gives up after `WAIT_IDLE`=3600 s rather than
record a contaminated run. On a cluster:

- **Request the node exclusively if you can** (`#SBATCH --exclusive`).
  The load average covers the whole node, so other users' jobs on a shared
  node would keep the check from passing. If you can't get exclusivity,
  set `MAX_LOAD=off`. The GPU checks still apply to your allocated GPU.
  Then note "shared node" in your report, since CPU-side stages
  (extraction, peak calling) can be slowed by neighbours.
- The script uses the GPU SLURM allocated (the first entry of
  `CUDA_VISIBLE_DEVICES`) unless `GPU_ID` is set. Don't set `GPU_ID` under
  SLURM.
- A desktop card that drives a display idles at a few % utilization and
  ~1 GB of memory. The 3080 needed `IDLE_UTIL=15`. Raise the thresholds
  only as far as the measured idle baseline, never to get past a real job.

**Sizing a SLURM job.** Published pydreg times on the Titan Xp range from
13 min (K562_groseq) to 43 min (G1), 4.2 h for all 12, with a peak RSS
of 10 GiB (G1). Current code should be somewhat faster. Request 16 CPUs, 1
GPU, 32 GB of memory, and about 6 h for run A or 1 h for B/C. If a job
times out, resubmit the same command: completed libraries are skipped
(`.done` markers), and the one that was interrupted reruns from scratch.

```bash
#!/bin/bash
#SBATCH --job-name=pydreg-bench
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --exclusive          # drop it, and set MAX_LOAD=off, if the partition doesn't allow it
cd /path/to/pydreg
DATA_DIR=/path/to/data bash figures/timing_scripts_cuda.sh
```

(Add the `--partition`/constraint your cluster uses to land on the right
GPU model, and check `env.txt` afterwards to confirm you got it.)

## Running (Apple, run D)

```bash
BACKENDS="mlx mlx-tf32" LIBS=Jurkat_ChROseq_1 bash figures/timing_scripts_apple.sh
uv run python3 figures/compare_apple_backends.py --libs Jurkat_ChROseq_1
```

Run it under `caffeinate -ims`. An earlier run without it slept and
inflated every stage ~1.9x (`docs/PERF_LOG.md`, 2026-11-21).

## Checking the results

```bash
cd figures && uv run python3 compare_cuda_runs.py    # needs bedtools on PATH
```

For fp32, expect:

- Jaccard vs dREG at least the published run's minus 0.001, and probably
  higher. On the 3080: 0.999793 vs 0.999363 published.
- A max score difference vs the published run under 1e-3.
- **The "positions" check may fail slightly, and that's expected.** The
  published run predates 0.4.0's fused kernel, whose score shifts change
  which gap-filled/densified positions are added near the score
  thresholds. On the 3080, 1.60e-4 of positions were in only one run,
  against a 1e-4 tolerance, while Jaccard improved. Report it. Don't
  loosen the tolerance or rerun to make it pass. Anything worse than a
  few 1e-4, or a Jaccard drop, is a real problem: stop and report it.

For tf32, there is no pass/fail. Expect Jaccard vs dREG around 0.996 and
1.2-1.5x faster end to end.

## What to hand back

1. Leave the full outputs in `OUT_DIR` (default `figures/output/`, which
   git ignores). The bigWigs and BED files go to the HF dataset, and the
   maintainer uploads them. **Don't upload to Hugging Face, push to
   `main`, tag or publish anything.**
2. Copy only the small text artifacts into the repo, mirroring run E:
   `figures/benchmark_results/1.0/<gpu-label>/<mode>/{env.txt,<lib>.time.log,<lib>.log}`
   plus `<gpu-label>/summary.csv` from `compare_cuda_runs.py --gpus <gpu-label>`.
   Commit them on a branch named `bench-1.0-<gpu-label>`, push that
   branch, and stop there.
3. Report: GPU and node (from `env.txt`), whether the node was exclusive,
   the per-library table and the medians `compare_cuda_runs.py` prints,
   any check failures with their numbers, and anything unusual in the
   logs (idle-check waits, retries, warnings beyond the known numba TBB
   one).

## After all runs are in (maintainer)

- Upload run A's outputs to the HF dataset's `pydreg/` folder, or copy them
  to `figures/benchmark_data/pydreg/` for a local preview, and rerun
  `plot_walltime.py`, `plot_memory.py`, `plot_peak_agreement.py` and
  `plot_score_exactness.py`.
- Update the README's Performance numbers from run A's medians, add the
  cross-hardware table from B-E, append a `docs/PERF_LOG.md` entry, and
  update the technical note.
- Then cut 1.0.0: version bump, `Development Status :: 5 - Production/Stable`,
  CHANGELOG, `CITATION.cff`.

## Run E: RTX 3080 (done 2026-11-23)

Ryzen 7 9800X3D (16 threads), WSL2, driver 621.14, cupy 14.1.1,
Jurkat_ChROseq_1 on local NVMe, `--cores 16 --seed 0`, `IDLE_UTIL=15`. The
code is `src/`-identical to `a8eae1f`: `env.txt` names `a80c1ca`, a local
merge that was never pushed. Artifacts are in
`figures/benchmark_results/1.0/nvidia-geforce-rtx-3080/`.

| | fp32 | tf32 |
|---|---|---|
| wall time | 380.7 s | 302.5 s (1.26x) |
| scoring informative positions | 157.5 s | 115.8 s |
| densify + merge (incl. gap-fill/densified scoring) | 137.3 s | 98.9 s |
| peak calling | 83.6 s | 83.5 s |
| peak RSS | 6.18 GiB | 6.18 GiB |
| Jaccard vs dREG | 0.999793 | 0.996183 |
| max score diff vs published | 5.0e-4 | 2.35e-3 |
| positions in only one run | 1.60e-4 (check fails, expected, see above) | 1.16e-3 |

## Run D: Apple M5 (done 2026-11-24)

pydreg 1.0.0 on Mac17,4 (Apple M5, 10-core CPU, 16 GiB), macOS 27.2.1, on
AC power, mlx 0.32.0, Jurkat_ChROseq_1 on the internal SSD, `--cores 10
--seed 0` (all physical cores; the CUDA runs used 16). Provenance: the
runs used `3e2f384` (`env.txt` records it as 0.4.0), whose `src/` is
identical to `a8eae1f`. The two backends ran as separate invocations, mlx
first, with a 5 min cooldown in between, under `caffeinate -ims`. `pmset
-g log` shows no sleep during the window. `timing_scripts_apple.sh`
doesn't write `env.txt`, so those were written by hand. Artifacts are in
`figures/benchmark_results/1.0/apple-m5/`.

| | mlx | mlx-tf32 |
|---|---|---|
| wall time | 1634.0 s | 1088.6 s (1.50x) |
| scoring informative positions | 863.5 s | 531.3 s (1.63x) |
| densify + merge (incl. gap-fill/densified scoring) | 622.5 s | 403.7 s |
| peak calling | 128.2 s | 135.0 s |
| peak RSS | 4.93 GiB | 6.00 GiB |
| Jaccard vs dREG | 0.999513 | 0.994692 |
| max score diff vs published | 2.25e-4 | 3.75e-3 |
| positions in only one run | 6.84e-5 (passes) | 1.54e-3 |

Checked against the 1.0.0 TITAN Xp run (run A). mlx's Jaccard vs dREG
(0.999513) is a little below cupy's (0.999793), within tolerance: the two
kernels' float32 scores differ at the 1e-4 level.

TF32 costs more agreement on the M5 than on the NVIDIA cards (0.994692
vs 0.996183 Jaccard) and buys more speed (1.50x vs 1.21-1.26x end to end).

#!/usr/bin/env bash
# Sweeps the whole 12-library benchmark through the dREG-vs-pydreg GPU
# profiling harness: for each library, one dREG run, one pydreg run, and
# one analyze_gpu_profile.py pass over the pair. Produces exactly the
# label convention (dreg_<LIB> / pydreg_<LIB>) that _gpu_common.py's
# figure scripts discover automatically.
#
# Usage:
#   ./profile_gpu_all.sh [LIBRARY...]      # default: all 12
#
# Configure via environment (defaults match the 2026-08-18 capture host):
#   BW_DIR      directory holding <LIB>.pl.bw / <LIB>.mn.bw  (default: .)
#   OUTDIR      where artifacts land                         (default: gpu_out)
#   DREG_DIR    dREG checkout with run_predict.bsh           (default: /home2/ayh8/dREG)
#   DREG_MODEL  asvm .rdata                                  (default: $DREG_DIR/asvm.gdm.6.6M.20170828.rdata)
#   CORES       cores for both tools                         (default: 16)
#   CUDA_GPU    CUDA device index passed to dREG             (default: 0)
#   NVSMI_GPU   nvidia-smi index BOTH tools land on          (default: unset -> analyzer auto-detects)
#   NVSMI_DREG / NVSMI_PYDREG   per-tool override of NVSMI_GPU, for the
#               unusual case where the two runs really are on different
#               cards                                        (default: NVSMI_GPU)
#
# IMPORTANT -- read README.md's "A real gotcha" section before trusting
# these: CUDA_GPU is a *CUDA* device index, which is not guaranteed to
# equal nvidia-smi's enumeration (on the original 2-GPU capture host,
# CUDA 0 was nvidia-smi 1). Run one library, check `nvidia-smi` while it's
# live, then set NVSMI_GPU for the rest. Leaving it unset falls back to
# the analyzer's auto-detect, which is unreliable on a shared
# multi-tenant node.
#
# NVSMI_GPU is ONE value because this comparison requires both tools on
# the same physical card -- a cross-hardware ratio would be meaningless,
# and on the original capture both dREG and pydreg were confirmed on
# nvidia-smi GPU 1. The separate NVSMI_DREG/NVSMI_PYDREG overrides exist
# only for a host where that genuinely isn't true; if you find yourself
# setting them to different values, check that's really what you want
# rather than an artifact of the CUDA-vs-nvidia-smi index gap above.
#
# Runs strictly sequentially, and that is not incidental: two profiled
# jobs sharing a node would contaminate each other's dmon samples and
# idle-gap analysis, which is the entire measurement. Budget roughly
# 8-15 hours of exclusive GPU time for all 12 (the Jurkat_PROseq
# reference pair was ~43min dREG + ~22min pydreg).
#
# Resumable: a library whose artifacts already exist is skipped, so an
# interrupted sweep can just be re-run. Delete a label's files to redo it.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

BW_DIR=${BW_DIR:-.}
OUTDIR=${OUTDIR:-$HERE/gpu_out}
DREG_DIR=${DREG_DIR:-/home2/ayh8/dREG}
DREG_MODEL=${DREG_MODEL:-$DREG_DIR/asvm.gdm.6.6M.20170828.rdata}
CORES=${CORES:-16}
CUDA_GPU=${CUDA_GPU:-0}

# The same 12 libraries figures/_common.py's LIBRARIES lists, read from it
# rather than duplicated here so the two can't drift.
if [[ $# -gt 0 ]]; then
  LIBRARIES=("$@")
else
  # read-loop rather than `mapfile`, which is bash 4+ only (macOS still
  # ships bash 3.2, and this script gets sanity-checked there).
  LIBRARIES=()
  while IFS= read -r lib; do
    LIBRARIES+=("$lib")
  done < <(
    python3 -c "import sys; sys.path.insert(0, '$HERE/..'); from _common import LIBRARIES; print('\n'.join(LIBRARIES))"
  )
fi

# Preflight: check every input up front rather than discovering a missing
# bigWig eleven hours into an overnight sweep.
missing=()
[[ -x "$DREG_DIR/run_predict.bsh" || -f "$DREG_DIR/run_predict.bsh" ]] || missing+=("$DREG_DIR/run_predict.bsh")
[[ -f "$DREG_MODEL" ]] || missing+=("$DREG_MODEL")
for lib in "${LIBRARIES[@]}"; do
  for suffix in pl mn; do
    [[ -f "$BW_DIR/$lib.$suffix.bw" ]] || missing+=("$BW_DIR/$lib.$suffix.bw")
  done
done
if [[ ${#missing[@]} -gt 0 ]]; then
  printf 'missing required input(s):\n' >&2
  printf '  %s\n' "${missing[@]}" >&2
  exit 1
fi
command -v pydreg >/dev/null || { echo "pydreg not on PATH" >&2; exit 1; }
command -v nsys >/dev/null || echo "WARNING: nsys not on PATH -- dmon utilization only, no kernel/memcpy/idle-gap data (the figures need it)" >&2

mkdir -p "$OUTDIR"
echo "[sweep] ${#LIBRARIES[@]} librar$([[ ${#LIBRARIES[@]} == 1 ]] && echo y || echo ies) -> $OUTDIR"
failed_analysis=()

already_captured() {  # $1 = label
  [[ -e "$OUTDIR/$1.nsys-rep" ]] || { ! command -v nsys >/dev/null && [[ -e "$OUTDIR/$1.log" ]]; }
}

for lib in "${LIBRARIES[@]}"; do
  dreg_label="dreg_$lib"
  pydreg_label="pydreg_$lib"

  echo
  echo "===== $lib ====="

  # dREG: run_predict.bsh, the legacy score-only path -- steps 1, 2, 4
  # only, so the whole process already IS the informative-positions
  # scoring phase and needs no windowing. NOT run_dREG.bsh, which also
  # scores gap-filled and densified position sets in the same process.
  # Argument order is plus, minus, out-prefix, model, cores, gpu.
  if already_captured "$dreg_label"; then
    echo "[skip] $dreg_label already captured"
  else
    "$HERE/profile_gpu.sh" "$OUTDIR" "$dreg_label" -- \
      bash "$DREG_DIR/run_predict.bsh" \
        "$BW_DIR/$lib.pl.bw" "$BW_DIR/$lib.mn.bw" "$OUTDIR/$lib.eval" \
        "$DREG_MODEL" "$CORES" "$CUDA_GPU"
  fi

  # pydreg: the normal full pipeline. There's no score-only mode, so this
  # also runs CPU peak calling afterward; the analyzer windows the profile
  # down to the scoring phase using pydreg's own -v log markers.
  if already_captured "$pydreg_label"; then
    echo "[skip] $pydreg_label already captured"
  else
    "$HERE/profile_gpu.sh" "$OUTDIR" "$pydreg_label" -- \
      pydreg "$BW_DIR/$lib.pl.bw" "$BW_DIR/$lib.mn.bw" "$OUTDIR/$lib" \
        --backend cupy -v -p "$CORES"
  fi

  nvsmi_dreg=${NVSMI_DREG:-${NVSMI_GPU:-}}
  nvsmi_pydreg=${NVSMI_PYDREG:-${NVSMI_GPU:-}}
  if [[ -z "$nvsmi_dreg" && -n "$nvsmi_pydreg" ]] || [[ -n "$nvsmi_dreg" && -z "$nvsmi_pydreg" ]]; then
    echo "set NVSMI_GPU (or both NVSMI_DREG and NVSMI_PYDREG) -- pinning one side and auto-detecting the other is how you end up comparing two different cards" >&2
    exit 1
  fi
  # An analyzer failure must NOT abort the sweep. The captures are the
  # expensive, unrepeatable part (hours of exclusive GPU time); analysis
  # is cheap and re-runnable against artifacts already on disk. Losing a
  # night of captures to a summarizing bug would be the worst possible
  # trade, so record the failure and keep capturing.
  # Built as one array rather than expanding a possibly-empty
  # "${gpu_index_args[@]}" inline: under `set -u`, expanding an empty
  # array that way is an "unbound variable" error on bash 3.2, so the
  # unpinned (auto-detect) path would break on older shells.
  analyzer_cmd=(
    python3 "$HERE/analyze_gpu_profile.py" "$OUTDIR" "$dreg_label" "$pydreg_label"
    --whole-trace "$dreg_label"
  )
  if [[ -n "$nvsmi_dreg" && -n "$nvsmi_pydreg" ]]; then
    analyzer_cmd+=(--gpu-index "$dreg_label=$nvsmi_dreg,$pydreg_label=$nvsmi_pydreg")
  fi
  if ! "${analyzer_cmd[@]}"; then
    echo "[warn] analysis FAILED for $lib -- captures are intact in $OUTDIR;" \
      "re-run this script (or the analyzer alone) after fixing to recover it" >&2
    failed_analysis+=("$lib")
  fi
done

echo
if [[ ${#failed_analysis[@]} -gt 0 ]]; then
  echo "[sweep] captures complete, but analysis failed for: ${failed_analysis[*]}" >&2
  echo "[sweep] their raw artifacts are intact -- re-run this script to retry analysis only" >&2
fi
echo "[sweep] done. Now draw the figures:"
echo "  python3 $HERE/plot_gpu_utilization.py --outdir $OUTDIR${NVSMI_GPU:+ --gpu-index $NVSMI_GPU}"
echo "  python3 $HERE/plot_gpu_time_breakdown.py --outdir $OUTDIR"
echo "  python3 $HERE/plot_gpu_efficiency.py --outdir $OUTDIR"

# Non-zero exit if any library's analysis failed, so an unattended sweep
# doesn't look like a clean success -- the captures are fine, but the
# summaries the figures read are incomplete until it's re-run.
[[ ${#failed_analysis[@]} -eq 0 ]]

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
#   NVSMI_DREG  nvidia-smi index dREG actually lands on      (default: unset -> analyzer auto-detects)
#   NVSMI_PYDREG  same, for pydreg                           (default: unset)
#
# IMPORTANT -- read README.md's "A real gotcha" section before trusting
# NVSMI_*: CUDA_GPU is a *CUDA* device index, which is not guaranteed to
# equal nvidia-smi's enumeration (on the original 2-GPU capture host,
# CUDA 0 was nvidia-smi 1). Run one library, check `nvidia-smi` while it's
# live, then set NVSMI_DREG/NVSMI_PYDREG for the rest. Leaving them unset
# falls back to the analyzer's auto-detect, which is unreliable on a
# shared multi-tenant node.
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

  gpu_index_args=()
  if [[ -n "${NVSMI_DREG:-}" && -n "${NVSMI_PYDREG:-}" ]]; then
    gpu_index_args=(--gpu-index "$dreg_label=$NVSMI_DREG,$pydreg_label=$NVSMI_PYDREG")
  fi
  python3 "$HERE/analyze_gpu_profile.py" "$OUTDIR" "$dreg_label" "$pydreg_label" \
    --whole-trace "$dreg_label" "${gpu_index_args[@]}"
done

echo
echo "[sweep] done. Now draw the figures:"
echo "  python3 $HERE/plot_gpu_utilization.py --outdir $OUTDIR${NVSMI_DREG:+ --gpu-index <label=idx,...>}"
echo "  python3 $HERE/plot_gpu_time_breakdown.py --outdir $OUTDIR"
echo "  python3 $HERE/plot_gpu_efficiency.py --outdir $OUTDIR"

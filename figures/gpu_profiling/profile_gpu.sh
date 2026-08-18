#!/usr/bin/env bash
# Wraps an arbitrary command (a dREG run_predict.bsh invocation, or a
# `pydreg ... --backend cupy -v` invocation) with GPU profiling, so the same
# harness works for both tools -- see README.md for the full methodology.
#
# Usage: profile_gpu.sh OUTDIR LABEL -- COMMAND [ARGS...]
#
# Produces, all under OUTDIR:
#   LABEL.dmon.csv   nvidia-smi dmon samples (wall-clock timestamped SM/
#                    memory-controller utilization + memory used), 1 Hz,
#                    for the command's full lifetime
#   LABEL.nsys-rep   nsys profile trace (CUDA + NVTX + OS runtime), if nsys
#                    is on PATH -- skipped with a warning otherwise
#   LABEL.log        the command's own stdout/stderr, unmodified (pydreg's
#                    `-v` INFO log lines are read back by
#                    analyze_gpu_profile.py to window this trace down to
#                    just the "scoring informative positions" phase; dREG's
#                    run_predict.bsh has no such phases to separate -- see
#                    README.md)
set -euo pipefail

if [[ $# -lt 4 || "$3" != "--" ]]; then
  echo "Usage: $0 OUTDIR LABEL -- COMMAND [ARGS...]" >&2
  exit 1
fi

OUTDIR=$1
LABEL=$2
shift 3
COMMAND=("$@")

mkdir -p "$OUTDIR"
DMON_CSV="$OUTDIR/$LABEL.dmon.csv"
LOG_FILE="$OUTDIR/$LABEL.log"
NSYS_REP="$OUTDIR/$LABEL.nsys-rep"

command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found on PATH" >&2; exit 1; }
if [[ -e "$NSYS_REP" ]]; then
  echo "refusing to overwrite existing $NSYS_REP -- pick a new LABEL or remove it" >&2
  exit 1
fi

echo "[profile_gpu] sampling nvidia-smi dmon -> $DMON_CSV"
# -s um: SM + memory-controller utilization, plus fb/bar1 memory used.
# -o DT: prepend wall-clock Date and Time columns (what analyze_gpu_profile.py
# correlates against pydreg's own log timestamps). Redirected via shell
# rather than dmon's own -f flag, since -f isn't present on all driver
# versions.
nvidia-smi dmon -s um -o DT >"$DMON_CSV" 2>&1 &
DMON_PID=$!
sleep 1  # let the first sample land before the command starts

cleanup() {
  kill "$DMON_PID" 2>/dev/null || true
  wait "$DMON_PID" 2>/dev/null || true
}
trap cleanup EXIT

if command -v nsys >/dev/null; then
  echo "[profile_gpu] running under nsys profile -> $NSYS_REP"
  nsys profile --trace=cuda,nvtx,osrt --output="$OUTDIR/$LABEL" -- "${COMMAND[@]}" \
    2>&1 | tee "$LOG_FILE"
else
  echo "[profile_gpu] WARNING: nsys not found on PATH -- skipping kernel-level trace," \
    "dmon utilization sampling only" >&2
  "${COMMAND[@]}" 2>&1 | tee "$LOG_FILE"
fi

echo "[profile_gpu] done -- $DMON_CSV, $LOG_FILE" \
  "$( [[ -e "$NSYS_REP" ]] && echo ", $NSYS_REP" )"

#!/usr/bin/env bash
# Times `pydreg` end-to-end on an Apple Silicon Mac with the mlx (Metal GPU)
# scoring backend against the numpy (CPU) one, across the 12-library
# benchmark set. Run figures/timing_scripts_download.sh first to fetch the
# bigWigs, then figures/compare_apple_backends.py on the outputs to check
# equivalence against the published results and summarize the timings.
#
# Both backends run with the same --cores and --seed, so their outputs
# differ only through scoring (float32 on the GPU vs float64 on the CPU).
# Backend order alternates per library, so neither backend always runs on
# a cooler machine. A completed run leaves a .done marker and is skipped on
# re-runs, so an interrupted sweep resumes where it stopped (FORCE=1 reruns
# everything).
#
# Environment overrides:
#   CORES     --cores value for both backends (default: all physical cores)
#   SEED      --seed value for both backends (default: 0)
#   LIBS      space-separated subset of libraries (default: all 12)
#   BACKENDS  space-separated backends to run (default: "numpy mlx")
#   FORCE     set to 1 to rerun runs that already completed
#   DATA_DIR / OUT_DIR, as in timing_scripts_common.sh
#
# Writes $OUT_DIR/apple/<backend>/<lib>.* : pydreg's outputs, <lib>.log
# (pydreg's -v log, including per-stage timings), and <lib>.time.log
# (`/usr/bin/time -l`: wall time and peak RSS, macOS format).
#
# Usage: bash figures/timing_scripts_apple.sh
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/timing_scripts_common.sh"

if [[ "$(uname -s)/$(uname -m)" != "Darwin/arm64" ]]; then
    echo "error: this benchmark needs an Apple Silicon Mac (got $(uname -s)/$(uname -m))" >&2
    exit 1
fi

CORES="${CORES:-$(sysctl -n hw.physicalcpu)}"
SEED="${SEED:-0}"
read -ra LIBS <<< "${LIBS:-${LIBRARIES[*]}}"
read -ra BACKENDS <<< "${BACKENDS:-numpy mlx}"

# Resolve the venv's own pydreg once (syncing the mlx extra into it), so the
# timings don't include `uv run`'s per-call environment check.
cd "$REPO_ROOT"
PYTHON="$(uv run --extra mlx python -c 'import sys; print(sys.executable)')"
PYDREG="$(dirname "$PYTHON")/pydreg"
if [[ " ${BACKENDS[*]} " == *" mlx "* ]] &&
    ! "$PYTHON" -c 'import mlx.core as mx, sys; sys.exit(0 if mx.metal.is_available() else 1)'; then
    echo "error: mlx is installed but sees no usable Metal GPU" >&2
    exit 1
fi

echo "pydreg $("$PYTHON" -c 'import pydreg; print(pydreg.__version__)') |" \
    "$(sysctl -n machdep.cpu.brand_string) | --cores $CORES --seed $SEED"

i=0
for lib in "${LIBS[@]}"; do
    if [[ ! -f "$(pl_bw "$lib")" || ! -f "$(mn_bw "$lib")" ]]; then
        echo "skipping $lib: bigWigs not found in $DATA_DIR" >&2
        continue
    fi

    order=("${BACKENDS[@]}")
    if (( i % 2 == 1 )); then
        order=()
        for (( j = ${#BACKENDS[@]} - 1; j >= 0; j-- )); do order+=("${BACKENDS[j]}"); done
    fi
    i=$((i + 1))

    for backend in "${order[@]}"; do
        dir="$OUT_DIR/apple/$backend"
        out="$dir/$lib"
        mkdir -p "$dir"
        if [[ -f "$out.done" && "${FORCE:-0}" != 1 ]]; then
            echo "== $lib ($backend): already done, skipping"
            continue
        fi
        rm -f "$out.done"
        echo "== $lib ($backend)"
        if /usr/bin/time -l -o "$out.time.log" \
            "$PYDREG" "$(pl_bw "$lib")" "$(mn_bw "$lib")" "$out" \
            -v --no-progress --backend "$backend" --cores "$CORES" --seed "$SEED" \
            2> "$out.log"; then
            touch "$out.done"
        else
            echo "   failed; see $out.log" >&2
        fi
    done
done

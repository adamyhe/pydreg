#!/usr/bin/env bash
# Times `pydreg` end-to-end on a Linux machine with an NVIDIA GPU (cupy
# backend) across the 12-library benchmark set, for the README/preprint
# figures and the cross-GPU table. Run figures/timing_scripts_download.sh
# first to fetch the bigWigs, then figures/compare_cuda_runs.py on the
# outputs to check them against the published results and summarize.
#
# Before every run it waits until the GPU is idle (utilization and memory
# below a threshold, sampled several times) and the 1-minute load average
# is low, so another job sharing the machine doesn't end up in the
# timings. If the machine doesn't go idle within WAIT_IDLE seconds, the
# sweep stops rather than recording a contaminated run.
#
# MODES runs each library once per precision mode: fp32 (the default,
# float32 GEMMs) and/or tf32 (--tf32; needs compute capability 8.0+). With
# both, the mode order alternates per library, so neither always runs on a
# cooler GPU, and there's a COOLDOWN pause between runs. A completed run
# leaves a .done marker and is skipped on re-runs, so an interrupted sweep
# resumes where it stopped (FORCE=1 reruns everything).
#
# Environment overrides:
#   GPU_ID        GPU index, as nvidia-smi numbers them (default: 0)
#   GPU_LABEL     output directory name for this card (default: derived from
#                 the GPU name, e.g. nvidia-titan-xp)
#   CORES         --cores value (default: 16, matching the published runs)
#   SEED          --seed value (default: 0)
#   LIBS          space-separated subset of libraries (default: all 12)
#   MODES         space-separated precision modes (default: "fp32")
#   COOLDOWN      seconds to pause between runs (default: 60)
#   IDLE_UTIL     max GPU utilization %, to count as idle (default: 5)
#   IDLE_MEM_MIB  max GPU memory in use, to count as idle (default: 2048;
#                 raise it on a desktop card that also drives a display)
#   MAX_LOAD      max 1-minute load average, to count as idle (default: 2)
#   WAIT_IDLE     seconds to wait for idle before giving up (default: 3600)
#   FORCE         set to 1 to rerun runs that already completed
#   DATA_DIR / OUT_DIR, as in timing_scripts_common.sh
#
# Writes $OUT_DIR/cuda/<gpu-label>/<mode>/<lib>.* : pydreg's outputs,
# <lib>.log (pydreg's -v log, including per-stage timings) and
# <lib>.time.log (`/usr/bin/time -v`: wall time and peak RSS), plus
# env.txt recording the code version and hardware. Those file names match
# the HF dataset's pydreg/ folder, so a run directory can be uploaded as-is
# or copied to figures/benchmark_data/pydreg/ to feed the plot_*.py scripts.
#
# Usage: bash figures/timing_scripts_cuda.sh
#        GPU_ID=1 MODES="fp32 tf32" LIBS=Jurkat_ChROseq_1 bash figures/timing_scripts_cuda.sh
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/timing_scripts_common.sh"

GPU_ID="${GPU_ID:-0}"
CORES="${CORES:-16}"
SEED="${SEED:-0}"
COOLDOWN="${COOLDOWN:-60}"
IDLE_UTIL="${IDLE_UTIL:-5}"
IDLE_MEM_MIB="${IDLE_MEM_MIB:-2048}"
MAX_LOAD="${MAX_LOAD:-2}"
WAIT_IDLE="${WAIT_IDLE:-3600}"
read -ra LIBS <<< "${LIBS:-${LIBRARIES[*]}}"
read -ra MODES <<< "${MODES:-fp32}"

for mode in "${MODES[@]}"; do
    if [[ "$mode" != fp32 && "$mode" != tf32 ]]; then
        echo "error: unknown mode '$mode' in MODES (expected fp32 and/or tf32)" >&2
        exit 1
    fi
done
command -v nvidia-smi > /dev/null || { echo "error: nvidia-smi not found" >&2; exit 1; }

# nvidia-smi numbers GPUs in PCI bus order; make CUDA (and so cupy) agree,
# then expose only the chosen card to pydreg.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_ID"

gpu_query() { nvidia-smi -i "$GPU_ID" --query-gpu="$1" --format=csv,noheader,nounits | head -1; }
# Integer reading of a numeric field; "[N/A]" (e.g. under WSL) counts as 0.
gpu_int() { local v; v="$(gpu_query "$1")"; v="${v//[^0-9]/}"; echo "${v:-0}"; }
GPU_NAME="$(gpu_query name)"
GPU_LABEL="${GPU_LABEL:-$(echo "$GPU_NAME" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-|-$//g')}"

# Resolve the venv's own pydreg once (syncing the gpu extra into it), so the
# timings don't include `uv run`'s per-call environment check.
cd "$REPO_ROOT"
PYTHON="$(uv run --extra gpu python -c 'import sys; print(sys.executable)')"
PYDREG="$(dirname "$PYTHON")/pydreg"
if ! "$PYTHON" -c 'from pydreg import backend; import sys; sys.exit(0 if backend.detect_backend() == "cupy" else 1)'; then
    echo "error: pydreg doesn't see a usable CUDA device through cupy" >&2
    exit 1
fi

# Echoes why the machine isn't idle, or nothing if it is. GPU utilization
# is sampled 5 times over ~5s and the maximum taken, since a single
# reading can land in a gap between another job's kernels.
busy_reason() {
    local max_util=0 util mem load
    for _ in 1 2 3 4 5; do
        util="$(gpu_int utilization.gpu)"
        (( util > max_util )) && max_util=$util
        sleep 1
    done
    mem="$(gpu_int memory.used)"
    load="$(cut -d' ' -f1 /proc/loadavg)"
    if (( max_util > IDLE_UTIL )); then
        echo "GPU utilization ${max_util}% > ${IDLE_UTIL}%"
    elif (( mem > IDLE_MEM_MIB )); then
        echo "GPU memory in use ${mem} MiB > ${IDLE_MEM_MIB} MiB"
    elif awk -v l="$load" -v m="$MAX_LOAD" 'BEGIN { exit !(l > m) }'; then
        echo "load average ${load} > ${MAX_LOAD}"
    fi
}

wait_for_idle() {
    local waited=0 reason
    while reason="$(busy_reason)"; [[ -n "$reason" ]]; do
        if (( waited >= WAIT_IDLE )); then
            echo "error: machine still busy after ${WAIT_IDLE}s ($reason); stopping" \
                "so no contaminated run gets recorded" >&2
            nvidia-smi -i "$GPU_ID" >&2 || true
            exit 1
        fi
        echo "   waiting for idle: $reason"
        sleep 30
        waited=$((waited + 35))
    done
}

write_env() {
    # write_env <dir> <mode>
    local commit dirty=""
    commit="$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
    git -C "$REPO_ROOT" diff --quiet HEAD -- src || dirty=" (uncommitted changes in src/)"
    {
        echo "date: $(date -Iseconds)"
        echo "host: $(hostname)"
        echo "pydreg: $("$PYTHON" -c 'import pydreg; print(pydreg.__version__)') @ ${commit}${dirty}"
        echo "cupy: $("$PYTHON" -c 'import cupy; print(cupy.__version__)')"
        echo "gpu: ${GPU_NAME} (index ${GPU_ID}, compute capability $(gpu_query compute_cap), $(gpu_query memory.total) MiB)"
        echo "driver: $(gpu_query driver_version)"
        echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //') ($(nproc) threads)"
        echo "flags: --backend cupy --cores ${CORES} --seed ${SEED}$([[ "$2" == tf32 ]] && echo ' --tf32')"
    } > "$1/env.txt"
}

echo "pydreg $("$PYTHON" -c 'import pydreg; print(pydreg.__version__)') | ${GPU_NAME} (${GPU_LABEL}) |" \
    "--cores $CORES --seed $SEED | modes: ${MODES[*]}"

first_run=1
i=0
for lib in "${LIBS[@]}"; do
    if [[ ! -f "$(pl_bw "$lib")" || ! -f "$(mn_bw "$lib")" ]]; then
        echo "skipping $lib: bigWigs not found in $DATA_DIR" >&2
        continue
    fi

    order=("${MODES[@]}")
    if (( i % 2 == 1 )); then
        order=()
        for (( j = ${#MODES[@]} - 1; j >= 0; j-- )); do order+=("${MODES[j]}"); done
    fi
    i=$((i + 1))

    for mode in "${order[@]}"; do
        dir="$OUT_DIR/cuda/$GPU_LABEL/$mode"
        out="$dir/$lib"
        mkdir -p "$dir"
        [[ -f "$dir/env.txt" ]] || write_env "$dir" "$mode"
        if [[ -f "$out.done" && "${FORCE:-0}" != 1 ]]; then
            echo "== $lib ($mode): already done, skipping"
            continue
        fi
        rm -f "$out.done"

        if (( ! first_run )); then sleep "$COOLDOWN"; fi
        first_run=0
        wait_for_idle

        extra=()
        [[ "$mode" == tf32 ]] && extra=(--tf32)
        echo "== $lib ($mode) [$(date +%H:%M:%S)]"
        if /usr/bin/time -v -o "$out.time.log" \
            "$PYDREG" "$(pl_bw "$lib")" "$(mn_bw "$lib")" "$out" \
            -v --no-progress --backend cupy --cores "$CORES" --seed "$SEED" ${extra[@]+"${extra[@]}"} \
            2> "$out.log"; then
            touch "$out.done"
            grep -E "Elapsed|Maximum resident" "$out.time.log" | sed 's/^\s*/   /'
        else
            echo "   failed; see $out.log" >&2
        fi
    done
done

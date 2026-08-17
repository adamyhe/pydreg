#!/usr/bin/env bash
# Times trogdor (github.com/adamyhe/trogdor) scoring across the 12-library
# benchmark set, for reference alongside the pydreg/dREG comparison. Run
# figures/timing_scripts_download.sh first to fetch the bigWigs.
#
# `-s 0` disables trogdor's storage threshold so every scored position is
# written, matching how pydreg/dREG report scores at all informative
# positions rather than only high-confidence ones.
#
# Requires: trogdor installed (`uv run trogdor` below assumes it's a
# dependency of this project's environment; adjust if run from trogdor's
# own venv instead) and simple_gpu_scheduler on PATH.
#
# Env vars:
#   GPUS   comma-separated GPU ids to schedule across (default: 0)
#
# Usage: bash figures/timing_scripts_trogdor.sh
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/timing_scripts_common.sh"
mkdir -p "$OUT_DIR/trogdor"

GPUS="${GPUS:-0}"

for lib in "${LIBRARIES[@]}"; do
    echo "/usr/bin/time -v -o $OUT_DIR/trogdor/${lib}.time.log \
        uv run trogdor score -v -p $(pl_bw "$lib") -m $(mn_bw "$lib") -o $OUT_DIR/trogdor/${lib}.prob.bw -s 0"
done | simple_gpu_scheduler --gpus "$GPUS"

#!/usr/bin/env bash
# Times `pydreg` end-to-end (scan -> features -> score -> peak call -> write)
# across the 12-library benchmark set. Run figures/timing_scripts_download.sh
# first to fetch the bigWigs.
#
# Usage: bash figures/timing_scripts_pydreg.sh
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/timing_scripts_common.sh"
mkdir -p "$OUT_DIR/pydreg"

for lib in "${LIBRARIES[@]}"; do
    /usr/bin/time -v -o "$OUT_DIR/pydreg/${lib}.time.log" \
        uv run pydreg "$(pl_bw "$lib")" "$(mn_bw "$lib")" "$OUT_DIR/pydreg/${lib}" -v --cores 16
done

#!/usr/bin/env bash
# Times real dREG (via the Danko-Lab/dREG-apptainer image) across the
# 12-library benchmark set, for comparison against
# figures/timing_scripts_pydreg_only.sh. Run figures/timing_scripts_download.sh
# first to fetch the bigWigs and the dREG SVR model.
#
# Build the image once (see https://github.com/Danko-Lab/dREG-apptainer):
#   git clone https://github.com/Danko-Lab/dREG-apptainer
#   cd dREG-apptainer
#   apptainer build --fakeroot --ignore-fakeroot-command dreg.sif dreg.def
#
# Env vars:
#   DREG_SIF   path to the built dreg.sif (default: ./dreg.sif)
#   GPU_ID     GPU device index passed to run_dREG (default: 0)
#
# Usage: bash figures/timing_scripts_dreg_apptainer.sh
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/timing_scripts_common.sh"
mkdir -p "$OUT_DIR/dreg"

DREG_SIF="${DREG_SIF:-./dreg.sif}"
GPU_ID="${GPU_ID:-0}"
MODEL="$DATA_DIR/asvm.gdm.6.6M.20170828.rdata"

for lib in "${LIBRARIES[@]}"; do
    /usr/bin/time -v -o "$OUT_DIR/dreg/${lib}.time.log" \
        apptainer exec --nv --bind "$DATA_DIR:/data,$OUT_DIR/dreg:/out" "$DREG_SIF" \
            dreg run_dREG "/data/$(basename "$(pl_bw "$lib")")" "/data/$(basename "$(mn_bw "$lib")")" \
            "/out/${lib}" "/data/$(basename "$MODEL")" 16 "$GPU_ID"
done

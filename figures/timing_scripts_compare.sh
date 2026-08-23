#!/usr/bin/env bash
# Reports per-library peak agreement between real dREG and pydreg via
# `bedtools jaccard` on their *.dREG.peak.prob.bed.gz outputs. Run
# figures/timing_scripts_dreg_apptainer.sh and
# figures/timing_scripts_pydreg.sh first.
#
# Requires: bedtools on PATH.
#
# Usage: bash figures/timing_scripts_compare.sh
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/timing_scripts_common.sh"

for lib in "${LIBRARIES[@]}"; do
    echo "== $lib =="
    bedtools jaccard \
        -a "$OUT_DIR/dreg/${lib}.dREG.peak.prob.bed.gz" \
        -b "$OUT_DIR/pydreg/${lib}.dREG.peak.prob.bed.gz"
done

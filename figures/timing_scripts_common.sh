#!/usr/bin/env bash
# Shared config sourced by the other figures/timing_scripts_*.sh scripts.
# Not runnable on its own.
#
# The 12 libraries are the finalized 0.2.7 benchmark set (see _common.py's
# LIBRARIES) -- same order, so per-library logs line up across scripts.
# Jurkat_ChROseq is not downloaded directly: timing_scripts_download.sh
# builds it by merging the Jurkat_ChROseq_1/_2 replicates.
LIBRARIES=(
    G1
    G3
    G5
    G6
    G7
    GM12878_groseq
    K562_groseq
    Jurkat_PROseq
    Jurkat_ChROseq_1
    Jurkat_ChROseq_2
    Jurkat_ChROseq
    Jurkat_leChROseq
)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/figures/data}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/figures/output}"

pl_bw() { echo "${DATA_DIR}/$1.pl.bw"; }
mn_bw() { echo "${DATA_DIR}/$1.mn.bw"; }

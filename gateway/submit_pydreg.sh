#!/usr/bin/env bash
# SLURM submission script for the pydreg gateway on Bridges-2 (PSC).
# Intended to be called by Airavata (or manually) with:
#   sbatch submit_pydreg.sh <plus.bw> <minus.bw> <out_prefix>
#
# Build the container once:
#   apptainer build pydreg.sif pydreg.def

#SBATCH -p GPU-shared
#SBATCH --gpus=v100-16:1
#SBATCH -A YOUR_ACCESS_ALLOCATION
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o pydreg_%j.out
#SBATCH -e pydreg_%j.err

set -euo pipefail

PLUS_BW="$1"
MINUS_BW="$2"
OUT_PREFIX="$3"
CORES="${SLURM_CPUS_PER_TASK:-16}"

PYDREG_SIF="${PYDREG_SIF:-./pydreg.sif}"

apptainer exec --nv \
    --bind "$(dirname "$PLUS_BW"),$(dirname "$OUT_PREFIX")" \
    "$PYDREG_SIF" \
    pydreg "$PLUS_BW" "$MINUS_BW" "$OUT_PREFIX" --cores "$CORES" --verbose

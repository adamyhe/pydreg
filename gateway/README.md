# pydreg ACCESS Gateway

Apptainer container and SLURM submission script for running pydreg on
Bridges-2 (PSC), intended as a drop-in replacement for the dREG gateway
backend.

## Setup

Build the container once on the cluster:

```bash
apptainer build pydreg.sif pydreg.def
```

This bakes in Python, CuPy, pydreg, and the pretrained model weights — fully
isolated from whatever modules or libraries the cluster has loaded.

## Running a job

```bash
sbatch submit_pydreg.sh /path/to/plus.bw /path/to/minus.bw /path/to/output_prefix
```

## SLURM configuration

The script defaults to a single V100-16GB on the `GPU-shared` partition.
Edit the `#SBATCH` directives to change resources:

| Directive | Default | Notes |
|---|---|---|
| `-p` | `GPU-shared` | Shared GPU partition (1 GPU). Use `GPU` for a full 8-GPU node. |
| `--gpus` | `v100-16:1` | Also available: `v100-32:1`, `l40s-48:1`, `h100-80:1`. |
| `-A` | `YOUR_ACCESS_ALLOCATION` | Your ACCESS allocation ID. |
| `--cpus-per-task` | `8` | pydreg's `--cores` is set to match. More cores = faster peak calling. |
| `--mem` | `16G` | Sufficient for all benchmarked libraries (peak RSS ≤10.5 GB). |
| `-t` | `02:00:00` | Conservative; largest benchmarked library finishes in ~45 min with 8 cores. |

## Cost

A typical pydreg job uses ~30 min of GPU time. ACCESS Credit cost per job
varies by cluster:

| Cluster | Credits / GPU-hr | Credits per pydreg job |
|---|---|---|
| Bridges-2 (PSC) | 53.846 | ~27 |
| Expanse (SDSC) | 53.846 | ~27 |
| Delta (NCSA) | 66 | ~33 |
| DeltaAI (NCSA) | 132 | ~66 |

Allocation capacity on Bridges-2 at ~27 credits/run:

| Allocation tier | Credits | Approx. pydreg runs | Review |
|---|---|---|---|
| Explore | 400,000 | ~14,800 | Automated |
| Discover | 1,500,000 | ~55,500 | Staff reviewed |
| Accelerate | 3,000,000 | ~111,000 | Panel reviewed |

## Airavata integration

The existing dREG gateway calls `apptainer exec --nv ... dreg run_dREG` with
bind-mounted input/output directories. The pydreg equivalent:

```
# dREG (old):
apptainer exec --nv --bind "$DATA:/data,$OUT:/out" dreg.sif \
    dreg run_dREG /data/plus.bw /data/minus.bw /out/prefix /data/model.rdata 16 0

# pydreg (new):
apptainer exec --nv --bind "$DATA,$OUT" pydreg.sif \
    pydreg $DATA/plus.bw $DATA/minus.bw $OUT/prefix --cores 16 --verbose
```

No model file argument — pretrained weights are baked into the container
(downloaded from Hugging Face at build time and cached in the image).

## Upgrading pydreg

Rebuild the container:

```bash
apptainer build pydreg.sif pydreg.def
```

To pin a specific version, edit `pydreg.def` to `pip3 install pydreg[gpu]==0.3.2`.

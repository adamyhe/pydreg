---
license: gpl-3.0
tags:
  - genomics
  - bioinformatics
  - pro-seq
  - gro-seq
  - chro-seq
  - bigwig
  - benchmark
  - transcription
pretty_name: pydreg vs. dREG benchmark outputs
---

# pydreg vs. dREG benchmark outputs

Raw benchmark artifacts backing the performance and accuracy comparisons in
[pydreg](https://github.com/adamyhe/pydreg), a from-scratch Python port of
[dREG](https://github.com/Danko-Lab/dREG) (Danko Lab). This dataset holds the
paired outputs of running both tools' full peak-calling pipeline
(`run_dREG`/`pydreg`) on the same 12 real PRO-seq/GRO-seq/ChRO-seq libraries,
plus the `/usr/bin/time -v` logs used to compare wall-clock time and peak
memory. It is data, not code — see the pydreg repo for the package itself and
for the scripts (`figures/timing_scripts_*.sh`) that produced everything here.

## Dataset structure

Two top-level directories, one per tool, each with one 8-file group per
library:

```
{tool}/{library}.dREG.infp.bw          # raw SVR score at every informative position, genome-wide
{tool}/{library}.dREG.raw.peak.bed.gz  # broad candidate peaks, before RF-assisted splitting
{tool}/{library}.dREG.peak.full.bed.gz # final called peaks, all columns (coords, score, prob, ...)
{tool}/{library}.dREG.peak.score.bed.gz
{tool}/{library}.dREG.peak.score.bw    # final called peaks, score column only
{tool}/{library}.dREG.peak.prob.bed.gz
{tool}/{library}.dREG.peak.prob.bw     # final called peaks, prob column only (1 - FDR-style p-value)
{tool}/{library}.time.log              # `/usr/bin/time -v` output for that run
```

`{tool}` is `dreg` (original R dREG) or `pydreg`. Filenames and the
`.dREG.`-infixed suffix convention match pydreg's own CLI output exactly, so
every `{tool}/{library}.dREG.*` pair is a direct, position-for-position
comparison of the two tools on identical input.

## Libraries

All 12 are published PRO-seq/GRO-seq/ChRO-seq libraries from GEO:

| Library            | GEO accession                    | Biosample | Assay      | Source                               |
| ------------------ | -------------------------------- | --------- | ---------- | ------------------------------------ |
| `G1`               | GSM1480327                       | K562      | PRO-seq    | Core et al., *Nat Genet* 2014        |
| `G3`               | GSM3452725                       | K562      | PRO-seq    | Wang et al., *Genome Res* 2019       |
| `G5`               | GSE89230                         | K562      | PRO-seq    | Vihervaara et al., *Nat Commun* 2017 |
| `G6`               | GSM2545324                       | K562      | PRO-seq    | Dukler et al., *Genome Res* 2017     |
| `G7`               | GSM2545325                       | K562      | PRO-seq    | Dukler et al., *Genome Res* 2017     |
| `GM12878_groseq`   | GSM1480326                       | GM12878   | GRO-seq    | Core et al., *Nat Genet* 2014        |
| `K562_groseq`      | GSM1480325                       | K562      | GRO-seq    | Core et al., *Nat Genet* 2014        |
| `Jurkat_PROseq`    | GSM3309955                       | Jurkat    | PRO-seq    | Chu et al., *Nat Genet* 2018         |
| `Jurkat_ChROseq_1` | GSM3309957                       | Jurkat    | ChRO-seq   | Chu et al., *Nat Genet* 2018         |
| `Jurkat_ChROseq_2` | GSM3309956                       | Jurkat    | ChRO-seq   | Chu et al., *Nat Genet* 2018         |
| `Jurkat_ChROseq`   | GSM3309956 + GSM3309957 (pooled) | Jurkat    | ChRO-seq   | Chu et al., *Nat Genet* 2018         |
| `Jurkat_leChROseq` | GSM3309958                       | Jurkat    | leChRO-seq | Chu et al., *Nat Genet* 2018         |

## How this was produced

- **dREG**: the original R package (`run_dREG.R`), scored with its 2017 SVR
  model (`asvm.gdm.6.6M.20170828.rdata`). The runs archived in this dataset
  were all generated against a raw, directly-installed R/CUDA/Rgtsvm stack
  (a bare `run_dREG.bsh` call), not the container described below.
- **pydreg**: this repo's Python port, same model weights (converted to
  safetensors, see `adamyhe/pydreg` on the Hub), run with `--cores 16`.
- Both tools ran on the same machine (NVIDIA P100 GPU + 16 Intel Xeon Silver
  4108 CPU cores), same bigWig inputs, one library at a time, wrapped in
  `/usr/bin/time -v` for wall-clock and peak-RSS logging.

Exact invocations: [`figures/timing_scripts_download.sh`](https://github.com/adamyhe/pydreg/blob/main/figures/timing_scripts_download.sh)
(fetches/rebuilds the 12 input bigWigs from GEO) and
[`figures/timing_scripts_pydreg_only.sh`](https://github.com/adamyhe/pydreg/blob/main/figures/timing_scripts_pydreg_only.sh).
For dREG, the repo's own script instructions now favor
[`figures/timing_scripts_dreg_apptainer.sh`](https://github.com/adamyhe/pydreg/blob/main/figures/timing_scripts_dreg_apptainer.sh)
(runs dREG inside the [Danko-Lab/dREG-apptainer](https://github.com/Danko-Lab/dREG-apptainer)
image) for reproducibility going forward: dREG's R/CUDA/Rgtsvm dependency
stack is fragile and version-sensitive to install by hand, which containerizing
avoids. The runs archived here predate that script and were produced with an
equivalent direct, non-containerized `run_dREG.bsh` call instead; a fresh
reproduction of this dataset should use the apptainer script.

Consumed by [`figures/plot_walltime.py`](https://github.com/adamyhe/pydreg/blob/main/figures/plot_walltime.py),
[`plot_memory.py`](https://github.com/adamyhe/pydreg/blob/main/figures/plot_memory.py),
[`plot_score_exactness.py`](https://github.com/adamyhe/pydreg/blob/main/figures/plot_score_exactness.py),
and [`plot_peak_agreement.py`](https://github.com/adamyhe/pydreg/blob/main/figures/plot_peak_agreement.py)
to produce the timing note figures, via the shared fetch helper in
[`figures/_common.py`](https://github.com/adamyhe/pydreg/blob/main/figures/_common.py).

## License

GPL-3.0, matching the [pydreg](https://github.com/adamyhe/pydreg) source
license these benchmark outputs support.

## Citation

If you use this data, please cite pydreg, dREG, and the original data
sources listed in the table above:

```bibtex
@article{wang2019dreg,
  author  = {Wang, Zhong and Chu, Tinyi and Choate, Lauren A. and Danko, Charles G.},
  title   = {Identification of regulatory elements from nascent transcription using dREG},
  journal = {Genome Research},
  year    = {2019},
  volume  = {29},
  number  = {2},
  pages   = {293--303},
  doi     = {10.1101/gr.238279.118}
}
```

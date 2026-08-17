#!/usr/bin/env bash
# Downloads and preprocesses the 12-library bigWig set used by the other
# figures/timing_scripts_*.sh benchmarks, plus the dREG SVR model needed by
# timing_scripts_dreg_apptainer.sh. Ported from the trogdor repo's
# scripts/data/download_training_data.sh + download_test_data.sh (GEO/GDS
# download URLs and the Jurkat_ChROseq replicate-merge steps), trimmed to
# just the 12 libraries in timing_scripts_common.sh's LIBRARIES.
#
# Requires: wget, bigWigMerge and bedGraphToBigWig (UCSC tools, e.g. via
# `conda install -c bioconda ucsc-bigwigmerge ucsc-bedgraphtobigwig`).
#
# Usage: bash figures/timing_scripts_download.sh
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/timing_scripts_common.sh"
mkdir -p "$DATA_DIR"

fetch() {
    # fetch <url> <output-path>
    wget --no-check-certificate --quiet --show-progress -O "$2" "$1"
}

# --- hg19 chromosome sizes (needed to rebuild the merged Jurkat_ChROseq bigWig) ---
fetch https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.chrom.sizes \
    "$DATA_DIR/hg19.chrom.sizes"

# --- dREG SVR model, for timing_scripts_dreg_apptainer.sh ---
# Zenodo archive, not the dreg.dnasequence.org gateway -- more persistent.
fetch "https://zenodo.org/records/10113379/files/asvm.gdm.6.6M.20170828.rdata?download=1" \
    "$DATA_DIR/asvm.gdm.6.6M.20170828.rdata"

# --- G1: K562 PRO-seq (GSM1480327) ---
fetch "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1480nnn/GSM1480327/suppl/GSM1480327%5FK562%5FPROseq%5Fplus.bw" "$(pl_bw G1)"
fetch "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1480nnn/GSM1480327/suppl/GSM1480327%5FK562%5FPROseq%5Fminus.bw" "$(mn_bw G1)"

# --- G3: K562 GRO-seq (GSM3452725) ---
fetch "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3452nnn/GSM3452725/suppl/GSM3452725%5FK562%5FNuc%5FNoRNase%5Fplus.bw" "$(pl_bw G3)"
fetch "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3452nnn/GSM3452725/suppl/GSM3452725%5FK562%5FNuc%5FNoRNase%5Fminus.bw" "$(mn_bw G3)"

# --- G5: K562 PRO-seq, combined replicates (GSE89230) ---
fetch "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE89nnn/GSE89230/suppl/GSE89230%5FNormalized%5FPRO%2Dseq%5FK562%5Fcombined%5Freplicates%5FNHS%5FplusStrand.bigWig" "$(pl_bw G5)"
fetch "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE89nnn/GSE89230/suppl/GSE89230%5FNormalized%5FPRO%2Dseq%5FK562%5Fcombined%5Freplicates%5FNHS%5FminusStrand.bigWig" "$(mn_bw G5)"

# --- G6: K562 PRO-seq, 0min celastrol rep1 (GSM2545324) ---
fetch "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM2545nnn/GSM2545324/suppl/GSM2545324%5F6045%5F7157%5F27170%5FHNHKJBGXX%5FK562%5F0min%5Fcelastrol10uM%5Frep1%5FGB%5FATCACG%5FR1%5Fplus.primary.bw" "$(pl_bw G6)"
fetch "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM2545nnn/GSM2545324/suppl/GSM2545324%5F6045%5F7157%5F27170%5FHNHKJBGXX%5FK562%5F0min%5Fcelastrol10uM%5Frep1%5FGB%5FATCACG%5FR1%5Fminus.primary.bw" "$(mn_bw G6)"

# --- G7: K562 PRO-seq, 0min celastrol rep2 (GSM2545325) ---
fetch "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM2545nnn/GSM2545325/suppl/GSM2545325%5F6045%5F7157%5F27176%5FHNHKJBGXX%5FK562%5F0min%5Fcelastrol10uM%5Frep2%5FGB%5FCAGATC%5FR1%5Fplus.primary.bw" "$(pl_bw G7)"
fetch "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM2545nnn/GSM2545325/suppl/GSM2545325%5F6045%5F7157%5F27176%5FHNHKJBGXX%5FK562%5F0min%5Fcelastrol10uM%5Frep2%5FGB%5FCAGATC%5FR1%5Fminus.primary.bw" "$(mn_bw G7)"

# --- GM12878 GRO-seq (GSM1480326) ---
fetch "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1480nnn/GSM1480326/suppl/GSM1480326%5FGM12878%5FGROseq%5Fplus.bigWig" "$(pl_bw GM12878_groseq)"
fetch "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1480nnn/GSM1480326/suppl/GSM1480326%5FGM12878%5FGROseq%5Fminus.bigWig" "$(mn_bw GM12878_groseq)"

# --- K562 GRO-seq (GSM1480325) ---
fetch "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1480nnn/GSM1480325/suppl/GSM1480325%5FK562%5FGROseq%5Fplus.bigWig" "$(pl_bw K562_groseq)"
fetch "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1480nnn/GSM1480325/suppl/GSM1480325%5FK562%5FGROseq%5Fminus.bigWig" "$(mn_bw K562_groseq)"

# --- Jurkat PRO-seq (GSM3309955) ---
fetch "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3309nnn/GSM3309955/suppl/GSM3309955%5F5587%5F5598%5F24204%5FHGC2FBGXX%5FJ%5FNUC%5FTTAGGC%5FR1%5Fplus%2Ebw" "$(pl_bw Jurkat_PROseq)"
fetch "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3309nnn/GSM3309955/suppl/GSM3309955%5F5587%5F5598%5F24204%5FHGC2FBGXX%5FJ%5FNUC%5FTTAGGC%5FR1%5Fminus%2Ebw" "$(mn_bw Jurkat_PROseq)"

# --- Jurkat leChRO-seq, RNase (GSM3309958) ---
fetch "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3309nnn/GSM3309958/suppl/GSM3309958%5FJurkat%5FChRO%5FRNase%5Fplus%2Ebw" "$(pl_bw Jurkat_leChROseq)"
fetch "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3309nnn/GSM3309958/suppl/GSM3309958%5FJurkat%5FChRO%5FRNase%5Fminus%2Ebw" "$(mn_bw Jurkat_leChROseq)"

# --- Jurkat ChRO-seq replicates, no RNase (GSM3309957, GSM3309956) ---
fetch "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3309nnn/GSM3309957/suppl/GSM3309957%5FJurkat%5FChRO%5FNoRNase%5Fplus%2Ebw" "$(pl_bw Jurkat_ChROseq_1)"
fetch "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3309nnn/GSM3309957/suppl/GSM3309957%5FJurkat%5FChRO%5FNoRNase%5Fminus%2Ebw" "$(mn_bw Jurkat_ChROseq_1)"
fetch "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3309nnn/GSM3309956/suppl/GSM3309956%5F5587%5F5598%5F24205%5FHGC2FBGXX%5FJ%5FCHR%5FTGACCA%5FR1%5Fplus%2Ebw" "$(pl_bw Jurkat_ChROseq_2)"
fetch "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3309nnn/GSM3309956/suppl/GSM3309956%5F5587%5F5598%5F24205%5FHGC2FBGXX%5FJ%5FCHR%5FTGACCA%5FR1%5Fminus%2Ebw" "$(mn_bw Jurkat_ChROseq_2)"

# Jurkat_ChROseq = the two no-RNase replicates merged, matching the
# original trogdor benchmark's pooled-replicate library.
HG19_SIZES="$DATA_DIR/hg19.chrom.sizes"
bigWigMerge "$(pl_bw Jurkat_ChROseq_1)" "$(pl_bw Jurkat_ChROseq_2)" "$DATA_DIR/Jurkat_ChROseq.pl.bg"
bigWigMerge -threshold=-10000000 "$(mn_bw Jurkat_ChROseq_1)" "$(mn_bw Jurkat_ChROseq_2)" "$DATA_DIR/Jurkat_ChROseq.mn.bg"
sort -k1,1 -k2,2n "$DATA_DIR/Jurkat_ChROseq.pl.bg" > "$DATA_DIR/Jurkat_ChROseq.sort.pl.bg"
sort -k1,1 -k2,2n "$DATA_DIR/Jurkat_ChROseq.mn.bg" > "$DATA_DIR/Jurkat_ChROseq.sort.mn.bg"
bedGraphToBigWig "$DATA_DIR/Jurkat_ChROseq.sort.pl.bg" "$HG19_SIZES" "$(pl_bw Jurkat_ChROseq)"
bedGraphToBigWig "$DATA_DIR/Jurkat_ChROseq.sort.mn.bg" "$HG19_SIZES" "$(mn_bw Jurkat_ChROseq)"
rm -f "$DATA_DIR"/Jurkat_ChROseq.*.bg "$DATA_DIR"/Jurkat_ChROseq.sort.*.bg

echo "Done. Downloaded/built bigWigs for: ${LIBRARIES[*]}"

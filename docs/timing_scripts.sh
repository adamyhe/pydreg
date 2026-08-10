GM12878_groseq G1 G2 G5 G7 G3 G6 Jurkat_ChROseq_2 Jurkat_leChROseq K562_groseq Jurkat_ChROseq_1 Jurkat_ChROseq Jurkat_PROseq K562_mnetseq K562_mnetseq_1 K562_mnetseq_2

for f in GM12878_groseq G1 G2 G5 G7 G3 G6 Jurkat_ChROseq_2 Jurkat_leChROseq K562_groseq Jurkat_ChROseq_1 Jurkat_ChROseq Jurkat_PROseq K562_mnetseq K562_mnetseq_1 K562_mnetseq_2;
do
  /usr/bin/time -v -o pydreg/${f}.time.log \
    pydreg ${f}.pl.bw ${f}.mn.bw pydreg/${f} -v --cores 16
done

f="PROseq_merged_QC_end"; /usr/bin/time -v -o pydreg/${f}.time.log \
    pydreg ${f}_plus.bw ${f}_minus.bw pydreg/${f} -v --cores 16

f="Sample_K562UNT_121109_proseq_1_QC"; /usr/bin/time -v -o pydreg/${f}.time.log \
    pydreg ${f}_plus.bw ${f}_minus.bw pydreg/${f} -v -p 16

for f in GM12878_groseq G1 G2 G5 G7 G3 G6 Jurkat_ChROseq_2 Jurkat_leChROseq K562_groseq Jurkat_ChROseq_1 Jurkat_ChROseq Jurkat_PROseq K562_mnetseq K562_mnetseq_1 K562_mnetseq_2;
do
  /usr/bin/time -v -o dreg/${f}.time.log \
    bash /home2/ayh8/dREG/run_dREG.bsh ${f}.pl.bw ${f}.mn.bw dreg/${f} /home2/ayh8/dREG/asvm.gdm.6.6M.20170828.rdata 16 0
done

f="PROseq_merged_QC_end"; /usr/bin/time -v -o dreg/${f}.time.log \
    bash /home2/ayh8/dREG/run_dREG.bsh ${f}_plus.bw ${f}_minus.bw dreg/${f} /home2/ayh8/dREG/asvm.gdm.6.6M.20170828.rdata 16 0

f="Sample_K562UNT_121109_proseq_1_QC"; /usr/bin/time -v -o dreg/${f}.time.log \
    bash /home2/ayh8/dREG/run_dREG.bsh ${f}_plus.bw ${f}_minus.bw dreg/${f} /home2/ayh8/dREG/asvm.gdm.6.6M.20170828.rdata 16 0

for f in GM12878_groseq G1 G2 G5 G7 G3 G6 Jurkat_ChROseq_2 Jurkat_leChROseq K562_groseq Jurkat_ChROseq_1 Jurkat_ChROseq Jurkat_PROseq K562_mnetseq K562_mnetseq_1 K562_mnetseq_2;
do 
    bedtools jaccard -a dreg/${f}.dREG.peak.prob.bed.gz -b pydreg/${f}.dREG.peak.prob.bed.gz;
done

for f in G1 G2 G5 G3 G6 K562_groseq Jurkat_ChROseq_1 Jurkat_ChROseq_2 Jurkat_ChROseq; do
echo "/usr/bin/time -v -o trogdor/${f}.time.log \
    trogdor score -v -p ${f}.pl.bw -m ${f}.mn.bw -o trogdor/${f}.prob.bw -s 0";
done | simple_gpu_scheduler --gpus 0,1

f="PROseq_merged_QC_end"; /usr/bin/time -v -o trogdor/${f}.time.log \
    trogdor score -v -p ${f}_plus.bw -m ${f}_minus.bw -o trogdor/${f}.prob.bw -s 0

for f in G1 G2 G5 G3 G6 K562_groseq Jurkat_ChROseq_1 Jurkat_ChROseq_2 Jurkat_ChROseq; do
echo "/usr/bin/time -v -o trogdor/${f}.time.log \
    trogdor score -v -p ${f}.pl.bw -m ${f}.mn.bw -o trogdor/${f}.prob.bw -s 0";
done | simple_gpu_scheduler --gpus 0,1

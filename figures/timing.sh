for f in K562_groseq Jurkat_PROseq G1 G3 G5 G6 G7 GM12878_groseq Jurkat_ChROseq_1 Jurkat_ChROseq_2 Jurkat_ChROseq Jurkat_leChROseq;
do
    /usr/bin/time -v -o pydreg/${f}.time.log \
        uv run pydreg ${f}.pl.bw ${f}.mn.bw pydreg/${f} -v -p 16 --pmv-laplace-fast
done

for f in K562_groseq Jurkat_PROseq G1 G3 G5 G6 G7 GM12878_groseq Jurkat_ChROseq_1 Jurkat_ChROseq_2 Jurkat_ChROseq Jurkat_leChROseq;
do
    /usr/bin/time -v -o dreg/${f}.time.log \
        bash /workdir/ayh8/dREG/run_dREG.bsh ${f}.pl.bw ${f}.mn.bw dreg/${f} /workdir/ayh8/dREG/asvm.gdm.6.6M.20170828.rdata 16 0
done

for f in K562_groseq Jurkat_PROseq G1 G3 G5 G6 G7 GM12878_groseq Jurkat_ChROseq_1 Jurkat_ChROseq_2 Jurkat_ChROseq Jurkat_leChROseq;
do
    bedtools jaccard -a pydreg/${f}.dREG.peak.prob.bed.gz -b ../../../trogdor_runs/dreg/${f}.dREG.peak.prob.bed.gz
done
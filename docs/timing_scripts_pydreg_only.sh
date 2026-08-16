for f in G1 G2 G3 G5 G6 G7 GM12878_groseq K562_groseq Jurkat_PROseq Jurkat_ChROseq_1 Jurkat_ChROseq_2 Jurkat_ChROseq Jurkat_leChROseq;
do
    /usr/bin/time -v -o pred/${f}.time.log \
        uv run pydreg data/${f}.pl.bw data/${f}.mn.bw pred/${f} -v --cores 16
done

for f in GM12878_groseq Jurkat_PROseq Jurkat_ChROseq_1 Jurkat_ChROseq_2 Jurkat_ChROseq Jurkat_leChROseq;
do
    /usr/bin/time -v -o pred/${f}.time.log \
        uv run pydreg data/${f}.pl.bw data/${f}.mn.bw pred/${f} -v --cores 16
done

for f in "PROseq_merged_QC_end" "Sample_K562UNT_121109_proseq_1_QC";
do
    /usr/bin/time -v -o pred/${f}.time.log \
        uv run pydreg data/${f}_plus.bw data/${f}_minus.bw pred/${f} -v --cores 16
done

Jurkat_leChROseq; /usr/bin/time -v -o pred/${f}.time.log \
        uv run pydreg data/${f}.pl.bw data/${f}.mn.bw pred/${f} -v --cores 16

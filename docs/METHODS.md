# How pydreg works

This page explains what pydreg does to a pair of PRO-seq/GRO-seq bigWig
files, step by step, without assuming you've read the source code.

pydreg is a port of [dREG](https://github.com/Danko-Lab/dREG) (Danko Lab).
It uses dREG's pretrained models and reproduces its recommended
peak-calling pipeline (`run_dREG.R`) step by step. It doesn't change the
method and can't train new models. Across 12 real
PRO-seq/GRO-seq/ChRO-seq libraries, the peaks it calls agree with dREG's
at a **median Jaccard index of 0.99967** (range 0.99840-0.99991). The
Jaccard index is the size of the overlap between the two peak sets divided
by the size of their union, so 1.0 would be identical. The small remaining
difference is what you'd expect from the random sampling in the p-value
step (see [step 4](#4-peak-calling)). The per-library breakdown is in
`figures/plots/peak_agreement.svg`.

The pipeline has four steps.

## 1. Find positions worth scoring

Most of the genome has no PRO-seq signal at all, so the first step finds
the small fraction of positions that do. pydreg moves along each
chromosome in 50 bp steps and keeps a position if either:

- the 100 bp window around it has more than 2 reads in total across both
  strands, or
- the 1,000 bp window around it has at least one read on *each* strand.

The positions that pass are called **informative positions**. Only these
are scored, which saves a great deal of work in the later, more expensive
steps.

## 2. Describe each position's surroundings

The model needs a fixed-size description of the transcription around each
informative position. pydreg builds one by counting reads in bins on both
sides of the position, at five different scales: narrow bins close in
(10 bp) and progressively wider bins further out (up to 5,000 bp), reaching
100 kb in each direction. It does this separately for each strand.

Each set of bins is passed through a logistic (S-shaped) transform, so one
extremely high bin can't drown out everything else. The result is a list
of 360 numbers per position that captures both the fine local shape of
transcription and the broader context.

## 3. Score each position

A pretrained support vector regression model turns each 360-number
description into a **dREG score**, roughly between 0 and 1. Higher scores
mean the position is more likely to be part of an active promoter or
enhancer. This is the same model dREG ships, unchanged.

The model is large (605,187 support vectors), and scoring is the most
expensive step in the pipeline. That's why pydreg can run it on an NVIDIA
or Apple Silicon GPU. Every backend computes the same thing; see
[`OPTIMIZATION.md`](OPTIMIZATION.md) for how they differ.

## 4. Peak calling

Per-position scores are noisy. Peak calling turns them into a clean list of
significant regions, in three stages.

**Find broad candidate regions.** pydreg first estimates what pure noise
looks like. It assumes the negative scores are all noise and fits a
Laplace distribution to them. Neighboring positions that score above a low
threshold derived from that fit are joined into broad candidate regions.
Each region probably contains at least one real regulatory element, but
possibly several.

**Split regions into individual peaks.** Two regulatory elements can sit
close enough together to end up in one broad region. pydreg smooths the
scores within each region and finds the local high points. For each pair
of neighboring high points, a small pretrained random-forest model decides
whether they belong to one peak or two. It looks at features such as their
heights and how deep the dip between them is. This is the same model and
logic as dREG.

**Test each peak for significance.** Each peak gets a p-value: the chance
that pure noise would produce scores at least this high at five points
around the peak's center. The noise model accounts for the fact that
nearby scores are correlated. This probability has no exact formula, so it
is estimated by randomized sampling, as in the original R code. That makes
it the **only random step in the pipeline**, and it is why two runs can
give very slightly different p-values. In practice this almost never
changes which peaks are called. `--seed` makes runs exactly repeatable (see
[`CLI.md`](CLI.md#randomness-and---seed)).

Finally, all p-values across the genome are corrected for multiple testing
(Benjamini-Hochberg false discovery rate by default), and peaks below the
corrected threshold (0.05 by default) form the final peak set.

## Output

The main result is `{out_prefix}.dREG.peak.full.bed.gz`: the significant
peaks with their score, p-value and center. pydreg also writes every
informative position's score as a bigWig, the candidate peaks from before
the significance test, and score and probability tracks for viewing in a
genome browser. [`CLI.md`](CLI.md#output-files) lists every file.

## Things to know

- **Strand sign doesn't matter.** pydreg takes the absolute value of both
  strands, as dREG does, so a minus-strand bigWig with negative values
  works the same as one with positive values.
- **Chromosomes come from the plus-strand file.** As in dREG, the list of
  chromosomes to scan is taken from `plus.bw`. A chromosome that appears
  only in `plus.bw` is still scored, with the minus strand treated as
  empty. A chromosome that appears only in `minus.bw` is skipped.
- **Some of dREG's quirks are kept on purpose.** A few things in the
  original code look like bugs, for example how one function's arguments
  are bound in the p-value calculation, or an off-by-one error that drops
  the last candidate region on each chromosome. pydreg reproduces them
  rather than fixing them, because the goal is to give the same answers as
  dREG. [`PLANNING.md`](PLANNING.md) lists each one and explains why it
  was kept.

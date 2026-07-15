# How pydreg works

This is a plain-language walkthrough of the pipeline `pydreg` runs on a
plus/minus-strand bigWig pair, for anyone who wants to understand the
algorithm without reading source code. For the full algorithmic
specification this was ported from (exact formulas, edge cases, every
upstream R quirk and why it's kept) see `docs/PLANNING.md`; for the
performance work layered on top without changing any of this, see
`docs/OPTIMIZATION.md`.

pydreg is a faithful, inference-only port of [dREG](https://github.com/Danko-Lab/dREG)
(Danko Lab) — it reproduces the original R/C package's `run_dREG.R` pipeline
end to end, including several upstream quirks that would look like bugs in
isolation but are kept deliberately because the pretrained model's expected
output depends on them. This has been validated directly: on real test data,
pydreg's called peaks agree with real dREG's at a **0.999728 Jaccard index**
(the fraction of the union of both peak sets that both implementations
agree on) — near-total agreement, with only the small residual expected from
inherent randomization in the p-value calculation (see "Peak calling"
below).

## 1. Informative-position scan

Most of the genome has no PRO-seq/GRO-seq signal at all, so the first step
finds the small fraction of positions worth scoring. `pydreg` tiles each
chromosome at a 50bp step and keeps a candidate position if either:

- a 100bp window around it has more than 2 combined reads across both
  strands ("OR" pass — catches low-signal loci where either strand
  contributes), or
- a 1000bp window around it has at least one read on *both* strands
  independently ("AND" pass — catches broader, lower-density transcription).

This produces a genome-wide set of "informative positions" — every position
downstream steps will actually score — restricting the (much more expensive)
feature extraction and model scoring to loci with any signal at all.

## 2. Multi-scale feature extraction

For each informative position, `pydreg` builds a fixed-length feature vector
describing the local transcription profile at multiple spatial scales
("zoom levels"). Each zoom level bins nearby read counts (both strands,
sign-stripped) into a fixed number of equal-width, non-overlapping bins on
either side of the position — small windows close in, progressively larger
windows further out (the pretrained model uses 5 zoom levels ranging from
10bp to 5000bp bins, spanning ±100kb total). Each zoom/strand combination is
independently rescaled with a logistic (sigmoid) transform, so a single very
high bin doesn't dominate the whole feature vector the way a raw count
would.

The result is a 360-dimensional feature vector per position (5 zoom levels
× 2 strands × up to 36 bins per zoom/strand) that captures both fine local
structure and broader regional context in one shot.

## 3. Scoring: the SVR model

A pretrained RBF-kernel support vector regressor (605,187 support vectors,
trained on the original dREG training data) maps each 360-dim feature vector
to a "dREG score" in roughly [0, 1] — higher means more likely to be part of
an active regulatory element. This is a large, fixed, non-retrainable model:
`pydreg` only implements *inference* against it, never training.

Because evaluating an RBF kernel against 605K support vectors for every
informative position is the single most expensive step in the pipeline,
`pydreg` supports three interchangeable scoring backends (GPU via cuML, CPU
via scikit-learn, and a dependency-free NumPy implementation) — see
`docs/OPTIMIZATION.md` for why the NumPy tier, not scikit-learn, is the
default CPU choice.

## 4. Peak calling

Raw per-position scores are noisy; peak calling turns them into a clean set
of significant regions with a controlled false-discovery rate. This has
three parts:

**Broad candidate regions.** Adjacent informative positions scoring above a
low significance floor (derived from fitting a Laplace noise model to the
negative-score tail, which is assumed to be pure noise) are merged into
broad candidate peaks — coarse regions likely to contain one or more real
regulatory elements, but not yet resolved into individual peaks.

**Splitting broad peaks into individual summits.** A broad peak can contain
more than one real regulatory element close together. `pydreg` smooths the
score profile inside each broad peak, finds local maxima, and uses a small
pretrained random-forest model (10 hand-engineered features describing each
pair of adjacent local maxima — their heights, the valley between them,
etc.) to decide whether each pair should be merged into one peak or kept
separate. This is the same random-forest-based splitting logic as the
original R implementation, just re-implemented for Python.

**Per-summit significance.** Each resulting summit gets a p-value from a
5-dimensional multivariate-Laplace tail probability (the null hypothesis is
that scores at 5 representative points around the summit are just
correlated Laplace-distributed noise, using a genome-wide autocorrelation
model fit once per run). This integral has no closed form and is evaluated
via quasi-Monte-Carlo sampling — the same algorithm family the original R
implementation uses (`mvtnorm::pmvnorm`), and, like the original, inherently
randomized: **re-running peak calling on the same input can produce
very slightly different p-values from run to run**, though this essentially
never changes which peaks clear the final significance threshold in
practice. See `docs/OPTIMIZATION.md` for how this specific step, initially
the dominant cost in peak calling, was made ~150x faster without changing
its statistical behavior.

Finally, all candidate summits' p-values are adjusted for multiple testing
(Benjamini-Hochberg FDR by default) across the whole genome at once, and
only peaks clearing the adjusted threshold (0.05 by default) are kept as the
final, significant peak set.

## Output

See the README's "Output files" table for the exact files written. In
short: every informative position's raw score (`*.dREG.infp.bed.gz`), every
candidate peak before FDR filtering (`*.dREG.raw.peak.bed.gz`), and the
final significant peaks with their scores/p-values/centers
(`*.dREG.peak.full.bed.gz`, plus score-only and probability-only variants as
both BED and bigWig tracks for visualization).

#!/usr/bin/env Rscript
#
# Extracts dREG's bundled peak-shape random forest
# (dREG/inst/extdata/rf-model-201803.RDS) into portable binary/JSON files,
# using only base R (no `randomForest` package required — this machine's R
# toolchain can't compile it at all, see notes below, but that package is
# only needed for its S3 predict() method; the underlying object is a plain
# S3 list we can read directly with readRDS() + unclass()).
#
# Background: unlike the per-position dREG score (an SVR over 360 genomic
# features, see export_dreg_model.R), this is a *separate*, much smaller
# model used only in the final peak-calling stage (peak_calling_rf.R:
# find_rf_peaks() / split_peak()) to decide whether two adjacent local
# maxima inside an already-called broad peak should be merged into one peak
# or split into two. It runs on a handful of hand-engineered peak-shape
# summary statistics per candidate split (order of thousands of calls per
# genome, not millions), not on genomic_data_model feature vectors.
#
# It's a `randomForest` R package regression forest (500 trees, <=153 nodes
# each, 10 continuous features: dist, r1, r2, y1, y2, maxy, d1, d2, d3, dr —
# confirmed against the package's actual C source, src/regTree.c's
# predictRegTree(), not just docs/memory). Prediction per tree:
#
#   k = 0  (root)
#   while nodestatus[k] != -1 (NODE_TERMINAL):
#     m = bestvar[k] - 1                      # bestvar is 1-indexed
#     k = leftDaughter[k]-1 if x[m] <= xbestsplit[k] else rightDaughter[k]-1
#   leaf value = nodepred[k]
#
# and the forest's prediction is the mean leaf value over all 500 trees.
# split_peak() then applies pred > 0.5 as a merge/split decision threshold.
#
# All 10 predictors are continuous (ncat all == 1 in this model), so there's
# no need to port the categorical-split (bit-packed) branch of
# predictRegTree() at all.
#
# This model is tiny (~460K numbers across 6 arrays) compared to the 1.75GB
# SVR, so — unlike that one — it's small enough to check into git directly
# if you'd rather do that than regenerate it from this script.
#
# Usage:
#   Rscript export_dreg_rf_model.R <path/to/rf-model.RDS> <output_dir>

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Usage: Rscript export_dreg_rf_model.R <path/to/rf-model.RDS> <output_dir>")
}
rds_path <- args[1]
out_dir <- args[2]
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

rf <- unclass(readRDS(rds_path))
stopifnot(rf$type == "regression")
f <- rf$forest
stopifnot(all(f$ncat == 1))  # all-continuous predictors; see note above

n_nodes <- f$nrnodes
n_trees <- f$ntree
feature_names <- rownames(rf$importance)

# f$nodestatus etc. are (n_nodes x n_trees) matrices (rows=nodes, cols=trees)
# — the OPPOSITE orientation from the SVR's SV matrix (samples x features).
# R's default flatten (as.integer/as.double with no transpose) is
# column-major, which for THIS (nodes, trees) shape already walks all nodes
# of tree 1, then all nodes of tree 2, ... i.e. exactly tree-major order.
# (Do not add t() here — that was tried and produces a garbled, cyclic tree
# structure: an earlier version of this script did, and it hung Python in
# an infinite loop during validation.) Readable in Python as
#   np.fromfile(path, dtype="<i4"/"<f8").reshape(n_trees, n_nodes)
writeBin(as.integer(f$nodestatus), file.path(out_dir, "nodestatus_i32.bin"), size = 4)
writeBin(as.integer(f$bestvar), file.path(out_dir, "bestvar_i32.bin"), size = 4)
writeBin(as.integer(f$leftDaughter), file.path(out_dir, "left_daughter_i32.bin"), size = 4)
writeBin(as.integer(f$rightDaughter), file.path(out_dir, "right_daughter_i32.bin"), size = 4)
writeBin(as.double(f$xbestsplit), file.path(out_dir, "xbestsplit_f64.bin"))
writeBin(as.double(f$nodepred), file.path(out_dir, "nodepred_f64.bin"))
writeBin(as.integer(f$ndbigtree), file.path(out_dir, "ndbigtree_i32.bin"), size = 4)

meta <- list(
  n_trees = n_trees,
  n_nodes = n_nodes,
  n_features = length(feature_names),
  feature_names = feature_names
)
kv <- sapply(names(meta), function(k) {
  v <- meta[[k]]
  if (length(v) > 1) paste0('"', k, '": ["', paste(v, collapse = '","'), '"]')
  else paste0('"', k, '": ', v)
})
writeLines(paste0("{\n  ", paste(kv, collapse = ",\n  "), "\n}"), file.path(out_dir, "meta.json"))

cat("Exported", n_trees, "trees x", n_nodes, "max nodes,", length(feature_names), "features to", out_dir, "\n")

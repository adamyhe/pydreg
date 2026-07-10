#!/usr/bin/env Rscript
#
# Extracts the pretrained dREG SVR model (e.g. asvm.gdm.6.6M.20170828.rdata,
# https://zenodo.org/records/10113379) into portable, framework-agnostic files
# that can be loaded from Python without R, e1071, or Rgtsvm.
#
# Background: the "asvm" object saved in this RData is trained with Rgtsvm
# (class "gtsvm"), a GPU SVM library — but dREG's own R code treats it as
# interchangeable with an e1071 "svm" object (see eval_svm.R, which does
# `class(asvm) <- "svm"` / `<- "gtsvm"` depending on backend). Both are thin
# wrappers around the same libsvm dual-form representation, so this script
# only needs base R (no e1071/Rgtsvm/dREG package required) to pull out:
#
#   - SV        support vectors, already compacted to exactly tot.nSV rows
#   - coefs     dual coefficients, ONE PER SUPPORT VECTOR
#   - rho       bias term
#   - gamma     RBF kernel bandwidth
#   - x.scale   per-feature center/scale applied before the kernel (z-score)
#   - y.scale   center/scale applied to the *output* to undo internal scaling
#
# Gotcha: asvm$coefs and asvm$index are NOT length tot.nSV. They're padded to
# 2 * n_train (epsilon-SVR keeps two dual variables, alpha and alpha*, per
# training example internally before compaction). Only the first tot.nSV
# entries are nonzero, and they align 1:1, in order, with the rows of the
# already-compacted `SV` matrix — so `coefs[1:tot.nSV]` is what you want, not
# `coefs[coefs != 0]` (order matters and both happen to agree here, but the
# former is the documented invariant).
#
# The resulting model is an ordinary RBF epsilon-SVR:
#   y_scaled = sum_i coefs[i] * exp(-gamma * ||x_scaled - SV[i,]||^2) - rho
#   y        = y_scaled * y.scale$scale + y.scale$center
# with x_scaled = (x_raw - x.scale$center) / x.scale$scale.
#
# Usage:
#   Rscript export_dreg_model.R <path/to/asvm.rdata> <output_dir>

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Usage: Rscript export_dreg_model.R <path/to/asvm.rdata> <output_dir>")
}
rdata_path <- args[1]
out_dir <- args[2]
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

env <- new.env()
load(rdata_path, envir = env)
asvm <- env$asvm
gdm <- env$gdm

n_sv <- asvm$tot.nSV
coefs <- asvm$coefs[1:n_sv]
SV <- asvm$SV
stopifnot(nrow(SV) == n_sv)

# Row-major float64 dump: readable in Python as
#   np.fromfile(path, dtype="<f8").reshape(n_sv, n_features)
writeBin(as.double(t(SV)), file.path(out_dir, "sv_matrix_f64.bin"))
writeBin(as.double(coefs), file.path(out_dir, "coefs_f64.bin"))
writeBin(as.double(asvm$x.scale[["scaled:center"]]), file.path(out_dir, "x_center_f64.bin"))
writeBin(as.double(asvm$x.scale[["scaled:scale"]]), file.path(out_dir, "x_scale_f64.bin"))

# gdm is an S4 "genomic_data_model" (dREG package); attributes() reads its
# slots without needing the dREG package/class definition installed.
gdm_attrs <- attributes(gdm)

meta <- list(
  n_sv = n_sv,
  n_features = ncol(SV),
  rho = asvm$rho,
  gamma = asvm$gamma,
  cost = asvm$cost,
  epsilon = asvm$epsilon,
  kernel = asvm$kernel,
  type = asvm$type,
  y_center = asvm$y.scale[["scaled:center"]],
  y_scale = asvm$y.scale[["scaled:scale"]],
  gdm_window_sizes = gdm_attrs$window_sizes,
  gdm_half_nWindows = gdm_attrs$half_nWindows
)
kv <- sapply(names(meta), function(k) {
  v <- meta[[k]]
  if (length(v) > 1) paste0('"', k, '": [', paste(v, collapse = ","), "]")
  else paste0('"', k, '": ', v)
})
writeLines(paste0("{\n  ", paste(kv, collapse = ",\n  "), "\n}"), file.path(out_dir, "meta.json"))

cat("Exported", n_sv, "support vectors x", ncol(SV), "features to", out_dir, "\n")

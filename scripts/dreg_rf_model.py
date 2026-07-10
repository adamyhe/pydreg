"""Loads dREG's bundled peak-shape random forest (exported by
export_dreg_rf_model.R from dREG/inst/extdata/rf-model-201803.RDS) and
predicts with it in plain NumPy.

This is a *different* model from the one in dreg_model.py: a small
`randomForest` R package regression forest (500 trees, <=153 nodes each, 10
continuous peak-shape features), used only in the final peak-calling stage
to decide whether two adjacent local maxima in a broad peak should be merged
or split (see peak_calling_rf.R: find_rf_peaks()/split_peak()). It runs on a
handful of candidate peak splits per genome, not per genomic position, so
there is no GPU/cuML case to make here the way there was for the 605k-SV
SVR — plain NumPy tree traversal is more than fast enough.

The traversal below is taken directly from this package's actual C source
(src/regTree.c, predictRegTree()), not from memory/docs:

    k = 0
    while nodestatus[k] != -1:          # -1 == NODE_TERMINAL
        m = bestvar[k] - 1              # bestvar is 1-indexed
        k = leftDaughter[k] - 1 if x[m] <= xbestsplit[k] else rightDaughter[k] - 1
    leaf value = nodepred[k]

averaged over all trees. All 10 predictors in this model are continuous
(ncat == 1 everywhere), so the categorical/bit-packed split branch of
predictRegTree() is not needed and isn't implemented here.

Caveat: this was validated by reading the shipped C source and by internal
consistency checks (tree-array invariants, output range), not by comparing
against a live R predict() call — this machine's R toolchain can't compile
the `randomForest` package to get a ground-truth run (unrelated broken
linker; see conversation). Low risk given the algorithm is unambiguous and
taken verbatim from source, but worth a real cross-check if that matters for
your use case.
"""

import json
import os

import numpy as np


class DREGPeakSplitForest:
    def __init__(self, model_path):
        """model_path: either a directory of raw .bin files + meta.json (as
        produced directly by export_dreg_rf_model.R), or a single
        .safetensors or .safetensors.zst file (as produced by
        pack_safetensors.py — the preferred format for distributing this
        model, e.g. on HF)."""
        if model_path.endswith(".safetensors") or model_path.endswith(".safetensors.zst"):
            from _safetensors_io import open_safetensors

            with open_safetensors(model_path) as f:
                meta = {k: json.loads(v) if v[:1] in "[{" else v for k, v in f.metadata().items()}
                self.nodestatus = f.get_tensor("nodestatus")
                self.bestvar = f.get_tensor("bestvar")
                self.left_daughter = f.get_tensor("left_daughter")
                self.right_daughter = f.get_tensor("right_daughter")
                self.xbestsplit = f.get_tensor("xbestsplit")
                self.nodepred = f.get_tensor("nodepred")
        else:
            meta = json.load(open(os.path.join(model_path, "meta.json")))
            shape = (meta["n_trees"], meta["n_nodes"])

            def load(name, dtype):
                return np.fromfile(os.path.join(model_path, name), dtype=dtype).reshape(shape)

            self.nodestatus = load("nodestatus_i32.bin", "<i4")
            self.bestvar = load("bestvar_i32.bin", "<i4")
            self.left_daughter = load("left_daughter_i32.bin", "<i4")
            self.right_daughter = load("right_daughter_i32.bin", "<i4")
            self.xbestsplit = load("xbestsplit_f64.bin", "<f8")
            self.nodepred = load("nodepred_f64.bin", "<f8")

        self.n_trees = int(meta["n_trees"])
        self.n_nodes = int(meta["n_nodes"])
        self.n_features = int(meta["n_features"])
        self.feature_names = meta["feature_names"]

    def _predict_tree(self, t, X):
        """X: (n_queries, n_features). Returns (n_queries,) leaf values for tree t."""
        k = np.zeros(X.shape[0], dtype=np.int64)
        node_status = self.nodestatus[t]
        bestvar = self.bestvar[t]
        xbestsplit = self.xbestsplit[t]
        left = self.left_daughter[t]
        right = self.right_daughter[t]

        active = node_status[k] != -1
        while np.any(active):
            idx = np.nonzero(active)[0]
            kk = k[idx]
            m = bestvar[kk] - 1
            go_left = X[idx, m] <= xbestsplit[kk]
            k[idx] = np.where(go_left, left[kk] - 1, right[kk] - 1)
            active = node_status[k] != -1

        return self.nodepred[t, k]

    def predict(self, X):
        """X: (n_queries, n_features), columns ordered as self.feature_names
        (dist, r1, r2, y1, y2, maxy, d1, d2, d3, dr). Returns the forest's
        mean leaf value per query (compared against 0.5 by dREG's
        split_peak() to decide merge vs. split)."""
        X = np.asarray(X, dtype=np.float64)
        out = np.zeros(X.shape[0])
        for t in range(self.n_trees):
            out += self._predict_tree(t, X)
        return out / self.n_trees


if __name__ == "__main__":
    import sys

    model = DREGPeakSplitForest(sys.argv[1] if len(sys.argv) > 1 else ".")
    # invariant checks
    assert np.all(np.isin(model.nodestatus, (-3, -1, 0))), "unexpected nodestatus value"
    valid_var = model.bestvar[model.nodestatus == -3]
    assert valid_var.min() >= 1 and valid_var.max() <= model.n_nodes, "bestvar out of range"

    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, model.n_features))
    preds = model.predict(X)
    print("predictions:", preds)
    assert np.all((preds >= 0) & (preds <= 1)), "prediction outside expected [0,1] range"
    print("OK: predictions in [0, 1], tree-array invariants hold")

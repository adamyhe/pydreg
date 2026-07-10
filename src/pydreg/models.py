"""The two pretrained dREG models: the per-position SVR scorer (DREGModel)
and the peak-shape random forest used only during peak calling to decide
whether adjacent local maxima merge or split (DREGPeakSplitForest).

Both load from a directory of raw .bin files + meta.json, a single
.safetensors[.zst] file, or (via from_pretrained()) directly from the
`adamyhe/dREG` Hugging Face repo. Ported from scripts/dreg_model.py and
scripts/dreg_rf_model.py — see those files' original docstrings (preserved
below) for the full validation history.

DREGModel is an RBF epsilon-SVR dual sum:
    y_scaled = sum_i coefs[i] * exp(-gamma * ||x_scaled - SV[i]||^2) - rho
    y = y_scaled * y_scale + y_center
with x_scaled = (x_raw - x_center) / x_scale. It is trivial to batch on any
array framework (NumPy, PyTorch, CuPy) without depending on the internals of
any particular SVM library.

On scikit-learn and cuML:
- scikit-learn's SVR *can* be made to predict with these weights, but only by
  writing to its private, undocumented libsvm-facing attributes
  (support_vectors_, _dual_coef_, _intercept_, _n_support, ...) and skipping
  .fit() entirely -- see `to_sklearn_svr()` below. Verified to match
  DREGModel.predict()'s NumPy math to ~1e-9 (sklearn 1.8.0). One gotcha:
  `_n_support` must have shape (2,) even for regression, or libsvm's C code
  segfaults.
- cuML supports this too, via a real, documented, stable public API:
  `cuml.svm.SVR.from_sklearn(sklearn_svr)` (added in cuML 25.06). It reads
  `dual_coef_`, `support_vectors_`, `intercept_`, `support_`, `_gamma`,
  `_sparse`, `fit_status_`, `shape_fit_`, `n_iter_` off exactly the kind of
  sklearn SVR object `to_sklearn_svr()` builds, copies them to GPU arrays,
  and cuML's `_predict()` rebuilds its C++ model struct from those arrays on
  *every call* -- it is not tied to state built only during `.fit()`. Only
  dense RBF-kernel SVR is covered by this interop path, which is exactly this
  model's shape (605k x 360 dense support vectors). See pydreg/backend.py for
  the tiered dispatch that uses this.

DREGPeakSplitForest is a small `randomForest` R package regression forest
(500 trees, <=153 nodes each, 10 continuous peak-shape features), used only
in the final peak-calling stage (see pydreg/rfsplit.py) to decide whether two
adjacent local maxima in a broad peak should be merged or split. It runs on a
handful of candidate peak splits per genome, not per genomic position, so
there is no GPU/cuML case to make here -- plain NumPy tree traversal is more
than fast enough. The traversal is taken directly from this package's actual
C source (src/regTree.c, predictRegTree()):

    k = 0
    while nodestatus[k] != -1:          # -1 == NODE_TERMINAL
        m = bestvar[k] - 1              # bestvar is 1-indexed
        k = leftDaughter[k] - 1 if x[m] <= xbestsplit[k] else rightDaughter[k] - 1
    leaf value = nodepred[k]

averaged over all trees. All 10 predictors in this model are continuous
(ncat == 1 everywhere), so the categorical/bit-packed split branch of
predictRegTree() is not needed and isn't implemented here.

Caveat carried over from the original port: the RF traversal was validated
by reading the shipped C source and by internal consistency checks
(tree-array invariants, output range), not by comparing against a live R
predict() call -- the R toolchain available during porting couldn't compile
the `randomForest` package (unrelated broken linker). Low risk given the
algorithm is unambiguous and taken verbatim from source, but worth a real
cross-check if that matters for your use case. Likewise, cuML's
from_sklearn() path was verified by reading cuML's source, not by running it
on real GPU hardware -- do a real end-to-end run before depending on it in
production.
"""

import json
import os

import numba
import numpy as np

from ._safetensors_io import open_safetensors

DEFAULT_REPO_ID = "adamyhe/dREG"


@numba.njit(cache=True)
def _forest_predict(nodestatus, bestvar, left_daughter, right_daughter, xbestsplit, nodepred, X):
    """JIT-compiled literal translation of src/regTree.c's predictRegTree(),
    looped over trees x queries x node-depth in compiled code -- see
    DREGPeakSplitForest's module docstring. Called with tiny X (a handful of
    rows) many times per broad peak across thousands of broad peaks, so
    Python/NumPy per-call dispatch overhead (not FLOPs) is what this removes;
    a plain Python loop over trees is no faster to JIT-vectorize across
    (500 trees, small X) than to compile as nested loops directly."""
    n_trees, _ = nodestatus.shape
    n_queries = X.shape[0]
    out = np.zeros(n_queries)
    for q in range(n_queries):
        total = 0.0
        for t in range(n_trees):
            k = 0
            while nodestatus[t, k] != -1:
                m = bestvar[t, k] - 1
                if X[q, m] <= xbestsplit[t, k]:
                    k = left_daughter[t, k] - 1
                else:
                    k = right_daughter[t, k] - 1
            total += nodepred[t, k]
        out[q] = total / n_trees
    return out


class DREGModel:
    def __init__(self, model_path):
        """model_path: either a directory of raw .bin files + meta.json (as
        produced directly by scripts/export_dreg_model.R), or a single
        .safetensors or .safetensors.zst file."""
        if model_path.endswith(".safetensors") or model_path.endswith(
            ".safetensors.zst"
        ):
            with open_safetensors(model_path) as f:
                meta = {
                    k: json.loads(v) if v[:1] in "[{" else v
                    for k, v in f.metadata().items()
                }
                self.SV = f.get_tensor("support_vectors")
                self.coefs = f.get_tensor("dual_coef")
                self.x_center = f.get_tensor("x_center")
                self.x_scale = f.get_tensor("x_scale")
        else:
            meta = json.load(open(os.path.join(model_path, "meta.json")))
            self.SV = np.fromfile(
                os.path.join(model_path, "sv_matrix_f64.bin"), dtype="<f8"
            ).reshape(meta["n_sv"], meta["n_features"])
            self.coefs = np.fromfile(
                os.path.join(model_path, "coefs_f64.bin"), dtype="<f8"
            )
            self.x_center = np.fromfile(
                os.path.join(model_path, "x_center_f64.bin"), dtype="<f8"
            )
            self.x_scale = np.fromfile(
                os.path.join(model_path, "x_scale_f64.bin"), dtype="<f8"
            )

        self.n_sv = int(meta["n_sv"])
        self.n_features = int(meta["n_features"])
        self.gamma = float(meta["gamma"])
        self.rho = float(meta["rho"])
        self.y_center = float(meta["y_center"])
        self.y_scale = float(meta["y_scale"])
        # The genomic_data_model feature layout this SVR was trained against
        # (see pydreg.features) -- a model is only meaningful relative to the
        # exact feature layout it was trained on, so this travels with it.
        self.window_sizes = np.array(meta["gdm_window_sizes"], dtype=int)
        self.half_n_windows = np.array(meta["gdm_half_nWindows"], dtype=int)
        self._sq_sv = np.sum(self.SV**2, axis=1)
        self._scorer_cache = {}

    @classmethod
    def from_pretrained(
        cls, repo_id=DEFAULT_REPO_ID, filename="svm.model.safetensors.zst", **hf_kwargs
    ):
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=repo_id, filename=filename, **hf_kwargs)
        return cls(path)

    def predict(self, X_raw, chunk=20_000):
        """X_raw: (n_queries, n_features) in the original (unscaled) feature
        space produced by dREG's genomic_data_model feature extraction.
        Returns dREG scores in their native (~[0, 1]) range."""
        X_scaled = (X_raw - self.x_center) / self.x_scale
        sq_x = np.sum(X_scaled**2, axis=1)

        y_scaled = np.zeros(X_scaled.shape[0])
        for start in range(0, self.n_sv, chunk):
            end = min(start + chunk, self.n_sv)
            sv_block = self.SV[start:end]
            cross = X_scaled @ sv_block.T
            sqdist = sq_x[:, None] + self._sq_sv[None, start:end] - 2 * cross
            K = np.exp(-self.gamma * sqdist)
            y_scaled += K @ self.coefs[start:end]
        y_scaled -= self.rho

        return y_scaled * self.y_scale + self.y_center


def to_sklearn_svr(model):
    """Builds a real sklearn.svm.SVR whose .predict() reproduces
    DREGModel.predict()'s NumPy math to ~1e-9 (verified on sklearn 1.8.0).

    Also the intended input to `cuml.svm.SVR.from_sklearn()` for a GPU
    version of the same weights -- see pydreg/backend.py and the module
    docstring above."""
    from sklearn.svm import SVR as SkSVR

    svr = SkSVR(kernel="rbf", gamma=model.gamma, epsilon=0.1)
    # A tiny dummy fit lets sklearn allocate every internal (libsvm-facing)
    # attribute in the right shape/dtype; we overwrite them immediately
    # after with the real pretrained weights.
    svr.fit(model.SV[:2], np.array([0.0, 1.0]))

    svr.support_vectors_ = model.SV
    svr.dual_coef_ = svr._dual_coef_ = model.coefs.reshape(1, -1)
    svr.support_ = np.arange(model.n_sv, dtype=np.int32)
    # NB: shape must be (2,) even for regression, or libsvm's C code segfaults.
    svr._n_support = np.array([model.n_sv, model.n_sv], dtype=np.int32)
    svr.intercept_ = svr._intercept_ = np.array([-model.rho])
    svr._gamma = model.gamma
    return svr


class DREGPeakSplitForest:
    def __init__(self, model_path):
        """model_path: either a directory of raw .bin files + meta.json (as
        produced directly by scripts/export_dreg_rf_model.R), or a single
        .safetensors or .safetensors.zst file."""
        if model_path.endswith(".safetensors") or model_path.endswith(
            ".safetensors.zst"
        ):
            with open_safetensors(model_path) as f:
                meta = {
                    k: json.loads(v) if v[:1] in "[{" else v
                    for k, v in f.metadata().items()
                }
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
                return np.fromfile(os.path.join(model_path, name), dtype=dtype).reshape(
                    shape
                )

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

    @classmethod
    def from_pretrained(
        cls, repo_id=DEFAULT_REPO_ID, filename="rf.model.safetensors.zst", **hf_kwargs
    ):
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=repo_id, filename=filename, **hf_kwargs)
        return cls(path)

    def predict(self, X):
        """X: (n_queries, n_features), columns ordered as self.feature_names
        (dist, r1, r2, y1, y2, maxy, d1, d2, d3, dr). Returns the forest's
        mean leaf value per query (compared against 0.5 by pydreg.rfsplit's
        split_peak() to decide merge vs. split)."""
        X = np.asarray(X, dtype=np.float64)
        return _forest_predict(
            self.nodestatus, self.bestvar, self.left_daughter, self.right_daughter,
            self.xbestsplit, self.nodepred, X,
        )

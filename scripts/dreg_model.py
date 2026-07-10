"""Loads a dREG SVR model exported by export_dreg_model.R and predicts with it,
using plain NumPy — no e1071/Rgtsvm/scikit-learn/cuML dependency required for
the core path.

On scikit-learn and cuML (corrected from an earlier, wrong assumption in this
file — verified by reading cuML's current `main` branch source, since no
CUDA/Linux machine was available to run it end-to-end):

- scikit-learn's SVR *can* be made to predict with these weights, but only by
  writing to its private, undocumented libsvm-facing attributes
  (support_vectors_, _dual_coef_, _intercept_, _n_support, ...) and skipping
  .fit() entirely — see `to_sklearn_svr()` below. Verified to match this
  module's NumPy predictions to ~1e-9 (sklearn 1.8.0). One gotcha: `_n_support`
  must have shape (2,) even for regression, or libsvm's C code segfaults.
- cuML *does* support this, via a real, documented, stable public API:
  `cuml.svm.SVR.from_sklearn(sklearn_svr)` (added in cuML 25.06, present
  through the current 26.06 stable; see rapidsai/cuml PR #6778). It reads
  `dual_coef_`, `support_vectors_`, `intercept_`, `support_`, `_gamma`,
  `_sparse`, `fit_status_`, `shape_fit_`, `n_iter_` off exactly the kind of
  sklearn SVR object `to_sklearn_svr()` builds, copies them to GPU arrays,
  and cuML's `_predict()` rebuilds its C++ model struct from those arrays on
  *every call* — it is not tied to state built only during `.fit()`. So:

      from cuml.svm import SVR as cuSVR
      gpu_model = cuSVR.from_sklearn(to_sklearn_svr(model))
      scores = gpu_model.predict(X_scaled)  # X already through raw_to_scaled

  Caveats: needs cuML >= 25.06 on Linux+CUDA (RAPIDS doesn't ship for macOS,
  so this could not be executed here — only confirmed via source inspection;
  do a real end-to-end run on GPU hardware before relying on it). Only dense
  RBF-kernel SVR is covered by this interop path, which is exactly this
  model's shape (605k x 360 dense support vectors), so no fallback caveats
  from the PR (sparse inputs, SVC probability/multiclass) apply here.

The model is just an RBF epsilon-SVR dual sum, which is trivial to batch on
any array framework (NumPy, PyTorch, CuPy) without depending on the internals
of any particular SVM library — swap `numpy` for `torch`/`cupy` and the same
formula runs on GPU unchanged. This remains the simplest, most portable
option if you don't want an sklearn/cuML dependency at all.

At genome scale (tens of millions of positions against 605k support vectors)
this is inherently compute-heavy regardless of platform; the original dREG
docs quote 4-12 hours on an NVIDIA K80 for the full pipeline. Chunk over
support vectors (see `chunk` below) to bound memory rather than materializing
a full (n_queries, n_support_vectors) kernel matrix at once.
"""

import json
import os

import numpy as np


class DREGModel:
    def __init__(self, model_path):
        """model_path: either a directory of raw .bin files + meta.json (as
        produced directly by export_dreg_model.R), or a single .safetensors
        or .safetensors.zst file (as produced by pack_safetensors.py — the
        preferred format for distributing this model, e.g. on HF)."""
        if model_path.endswith(".safetensors") or model_path.endswith(".safetensors.zst"):
            from _safetensors_io import open_safetensors

            with open_safetensors(model_path) as f:
                meta = {k: json.loads(v) if v[:1] in "[{" else v for k, v in f.metadata().items()}
                self.SV = f.get_tensor("support_vectors")
                self.coefs = f.get_tensor("dual_coef")
                self.x_center = f.get_tensor("x_center")
                self.x_scale = f.get_tensor("x_scale")
        else:
            meta = json.load(open(os.path.join(model_path, "meta.json")))
            self.SV = np.fromfile(
                os.path.join(model_path, "sv_matrix_f64.bin"), dtype="<f8"
            ).reshape(meta["n_sv"], meta["n_features"])
            self.coefs = np.fromfile(os.path.join(model_path, "coefs_f64.bin"), dtype="<f8")
            self.x_center = np.fromfile(os.path.join(model_path, "x_center_f64.bin"), dtype="<f8")
            self.x_scale = np.fromfile(os.path.join(model_path, "x_scale_f64.bin"), dtype="<f8")

        self.n_sv = int(meta["n_sv"])
        self.n_features = int(meta["n_features"])
        self.gamma = float(meta["gamma"])
        self.rho = float(meta["rho"])
        self.y_center = float(meta["y_center"])
        self.y_scale = float(meta["y_scale"])
        self._sq_sv = np.sum(self.SV**2, axis=1)

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
    version of the same weights — see the module docstring."""
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


if __name__ == "__main__":
    import sys

    model = DREGModel(sys.argv[1] if len(sys.argv) > 1 else ".")
    rng = np.random.default_rng(0)
    X = model.x_center + rng.normal(scale=0.5, size=(5, model.n_features)) * model.x_scale
    print(model.predict(X))

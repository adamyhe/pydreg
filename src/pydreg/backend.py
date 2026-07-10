"""Tiered cuML -> scikit-learn -> NumPy scoring backend dispatch (see
docs/PLANNING.md "Backend dispatch" / "Batching"). pydreg.pipeline never
branches on backend -- it only ever calls a Scorer's uniform .predict().

Detection is lazy (never at import time -- importing cuml alone can take
seconds and drags in cupy/numba-cuda/rmm, a bad tax on every invocation
including --help) and cached once per process. The cuML tier is unvalidated
on real GPU hardware (none was available where this was written) -- the
default query-chunk size there especially should be re-checked on an actual
CUDA box.
"""

import functools
import logging

import numpy as np

from .models import to_sklearn_svr

logger = logging.getLogger(__name__)

# Default query-position chunk sizes per backend tier. Sized for the
# pretrained SVR's shape (605,187 support vectors x 360 features); see
# docs/PLANNING.md "Batching" for the memory-bound reasoning behind each.
DEFAULT_QUERY_CHUNK = {"numpy": 4096, "sklearn": 50_000, "cuml": 200_000}


class BackendUnavailable(RuntimeError):
    """Raised when an explicitly requested backend can't actually be used
    (rather than silently falling back to the next tier)."""


@functools.lru_cache(maxsize=1)
def detect_backend():
    """Probes once per process and returns "cuml" or "numpy" -- the best
    backend actually usable right now.

    "sklearn" is CPU-only and is never auto-selected: benchmarked at ~15x
    slower than the "numpy" tier (libsvm's predict loop is single-threaded C,
    vs. DREGModel.predict's chunked matmul, which dispatches to a
    multithreaded BLAS) despite computing the same math (agrees to ~1e-10).
    It remains selectable via --backend sklearn, and to_sklearn_svr() is
    still required as the input to cuml.svm.SVR.from_sklearn()."""
    try:
        import cuml.svm
    except ModuleNotFoundError:
        logger.info("cuml not installed -- install pydreg[gpu] for GPU scoring")
    else:
        try:
            probe = cuml.svm.SVR()
            probe.fit(np.zeros((2, 1)), np.array([0.0, 1.0]))
            probe.predict(np.zeros((1, 1)))
        except Exception as e:
            # cuml installs fine on a GPU-less box; only a real op proves a
            # usable device. CUDA init failures aren't one clean exception
            # type, hence the broad catch.
            logger.info(
                "cuml installed but no usable CUDA GPU detected at runtime (%s) "
                "-- falling back to CPU",
                e,
            )
        else:
            return "cuml"

    return "numpy"


class Scorer:
    """Uniform `.predict(X_chunk) -> np.ndarray` wrapper hiding backend
    differences from callers."""

    def __init__(self, backend, predict_fn):
        self.backend = backend
        self._predict_fn = predict_fn

    def predict(self, X):
        return self._predict_fn(X)


def _wrap_sklearn_like(dreg_model, sk_predict):
    """Both the sklearn and cuml tiers predict in the SVR's internal scaled
    feature space and need the same x-scale / y-unscale wrapping DREGModel
    itself does -- see pydreg.models.DREGModel.predict."""

    def predict_fn(X):
        X_scaled = (X - dreg_model.x_center) / dreg_model.x_scale
        y_scaled = np.asarray(sk_predict(X_scaled))
        return y_scaled * dreg_model.y_scale + dreg_model.y_center

    return predict_fn


def build_scorer(dreg_model, backend=None):
    """Builds (and caches, on `dreg_model._scorer_cache`) a Scorer for the
    requested backend. backend=None ("auto") picks the best available tier
    via detect_backend(). An explicit backend name that isn't usable raises
    BackendUnavailable rather than silently falling back -- a caller who
    asked for a specific backend wants a loud failure, not a silent
    slowdown on a job sized for that backend's throughput."""
    resolved = backend or detect_backend()
    if resolved not in DEFAULT_QUERY_CHUNK:
        raise ValueError(f"unknown backend {resolved!r}, expected one of {sorted(DEFAULT_QUERY_CHUNK)}")

    if resolved in dreg_model._scorer_cache:
        return dreg_model._scorer_cache[resolved]

    if resolved == "cuml":
        try:
            import cuml.svm
        except ModuleNotFoundError as e:
            raise BackendUnavailable("cuml is not installed (pip install 'pydreg[gpu]')") from e
        try:
            gpu_model = cuml.svm.SVR.from_sklearn(to_sklearn_svr(dreg_model))
        except Exception as e:
            raise BackendUnavailable(f"cuml is installed but could not build a GPU model: {e}") from e
        predict_fn = _wrap_sklearn_like(dreg_model, gpu_model.predict)

    elif resolved == "sklearn":
        try:
            sk_model = to_sklearn_svr(dreg_model)
        except ModuleNotFoundError as e:
            raise BackendUnavailable("scikit-learn is not installed") from e
        predict_fn = _wrap_sklearn_like(dreg_model, sk_model.predict)

    else:  # numpy
        predict_fn = dreg_model.predict

    scorer = Scorer(resolved, predict_fn)
    dreg_model._scorer_cache[resolved] = scorer
    return scorer

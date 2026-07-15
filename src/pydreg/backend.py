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
import importlib.util
import logging

import numpy as np

from .models import to_sklearn_svr

logger = logging.getLogger(__name__)

# Default query-position chunk sizes per backend tier. Sized for the
# pretrained SVR's shape (605,187 support vectors x 360 features); see
# docs/PLANNING.md "Batching" for the memory-bound reasoning behind each.
DEFAULT_QUERY_CHUNK = {"numpy": 4096, "sklearn": 50_000, "cuml": 2**20}


class BackendUnavailable(RuntimeError):
    """Raised when an explicitly requested backend can't actually be used
    (rather than silently falling back to the next tier)."""


def _cuml_installed():
    return importlib.util.find_spec("cuml") is not None


def _cuda_runtime_available():
    """Return whether CUDA is visible through CuPy, a cuML dependency."""
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        logger.debug("CuPy CUDA runtime availability probe failed", exc_info=True)
        return False


@functools.lru_cache(maxsize=1)
def detect_backend():
    """Probes once per process and returns "cuml" or "numpy" -- the best
    backend actually usable right now.

    "sklearn" is CPU-only and is never auto-selected: benchmarked at ~15x
    slower than the "numpy" tier despite computing the same math (agrees to
    ~1e-10). Not a threading gap (forcing single-threaded BLAS via
    VECLIB_MAXIMUM_THREADS=1 doesn't change DREGModel.predict's wall-clock
    time at all) -- libsvm's predict path (svm.cpp's predict_values ->
    k_function) mallocs a temp array and issues one tiny BLAS level-1 dot()
    per query-SV pair (605,187 of them), while DREGModel.predict's chunked
    `X_scaled @ sv_block.T` issues one BLAS level-3 GEMM call that computes
    all of them at once -- a genuinely different computational shape, not a
    parallelism difference (see docs/PERF_LOG.md's 2026-07-14 entry). It
    remains selectable via --backend sklearn, and to_sklearn_svr() is still
    required as the input to cuml.svm.SVR.from_sklearn()."""
    if not _cuml_installed():
        logger.info("cuml not installed -- install pydreg[gpu] for GPU scoring")
        return "numpy"

    if not _cuda_runtime_available():
        logger.info("cuml installed but no usable CUDA GPU detected at runtime -- falling back to CPU")
        return "numpy"

    return "cuml"


class Scorer:
    """Uniform `.predict(X_chunk) -> np.ndarray` wrapper hiding backend
    differences from callers."""

    def __init__(self, backend, predict_fn):
        self.backend = backend
        self._predict_fn = predict_fn

    def predict(self, X):
        return self._predict_fn(X)


def _wrap_sklearn_like(dreg_model, sk_predict, backend_name):
    """Both the sklearn and cuml tiers predict in the SVR's internal scaled
    feature space and need the same x-scale / y-unscale wrapping DREGModel
    itself does -- see pydreg.models.DREGModel.predict."""
    validated = False

    def predict_fn(X):
        nonlocal validated
        X_scaled = (X - dreg_model.x_center) / dreg_model.x_scale
        y_scaled = np.asarray(sk_predict(X_scaled))
        y = y_scaled * dreg_model.y_scale + dreg_model.y_center

        if not validated:
            sample = X[: min(len(X), 8)]
            reference = dreg_model.predict(sample)
            candidate = np.asarray(y[: len(sample)], dtype=float)
            if (
                candidate.shape != reference.shape
                or not np.all(np.isfinite(candidate))
                or not np.allclose(candidate, reference, rtol=1e-4, atol=1e-4)
            ):
                max_abs = (
                    float(np.max(np.abs(candidate - reference)))
                    if candidate.shape == reference.shape
                    else float("nan")
                )
                raise BackendUnavailable(
                    f"{backend_name} backend predictions do not match the NumPy reference "
                    f"on a first-batch smoke test (max_abs_diff={max_abs:.6g}); "
                    "use --backend numpy until this backend conversion is fixed"
                )
            validated = True
        return y

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
        predict_fn = _wrap_sklearn_like(dreg_model, gpu_model.predict, "cuml")

    elif resolved == "sklearn":
        try:
            sk_model = to_sklearn_svr(dreg_model)
        except ModuleNotFoundError as e:
            raise BackendUnavailable("scikit-learn is not installed") from e
        predict_fn = _wrap_sklearn_like(dreg_model, sk_model.predict, "sklearn")

    else:  # numpy
        predict_fn = dreg_model.predict

    scorer = Scorer(resolved, predict_fn)
    dreg_model._scorer_cache[resolved] = scorer
    return scorer

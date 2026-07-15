"""Tiered cuML -> scikit-learn -> NumPy scoring backend dispatch (see
docs/PLANNING.md "Backend dispatch" / "Batching"). pydreg.pipeline never
branches on backend -- it only ever calls a Scorer's uniform .predict().

Detection is lazy (never at import time -- importing cuml alone can take
seconds and drags in cupy/numba-cuda/rmm, a bad tax on every invocation
including --help) and cached once per process.

The cuML tier is now validated on real GPU hardware -- and that validation
surfaced a real, confirmed finding: RAPIDS/cuML dropped support for Pascal
GPUs (compute capability < 7.0) in the 24.02 release, and running a
Pascal-era cuML build on such hardware doesn't error, it silently returns
wrong predictions (RAPIDS's own deprecation notice: "use of a Pascal GPU
will either fail or return invalid results"). Confirmed end-to-end on a
real production run: cuml 26.06.00's SVR.from_sklearn()-built model
diverged from the NumPy reference by ~0.05 on an NVIDIA TITAN X (Pascal,
compute capability 6.1), while the *exact same bigWig inputs* on an A100
(compute capability 8.0) ran clean (Jaccard > 0.999 vs. real dREG). See
docs/OPTIMIZATION.md for the full investigation. This is exactly why
_wrap_sklearn_like's first-batch smoke test exists -- and why
detect_backend()/build_scorer() also check compute capability directly, so
unsupported hardware is caught before or instead of a confusing
mid-pipeline BackendUnavailable.

EXPERIMENTAL: a fourth tier, "cupy", evaluates DREGModel.predict's exact
RBF dual-sum formula directly on a CuPy device array instead of routing
through cuml.svm's own compiled kernel (see _build_cupy_predict_fn). CuPy's
own array ops support compute capability >=3.0 -- including the Pascal
hardware cuML dropped -- and being the same formula as the already-validated
NumPy tier, it carries none of the cross-library conversion risk the cuml
round-trip does. Not yet validated on real GPU hardware (this was written
on a machine with no GPU at all) and not auto-selected by detect_backend()
-- only reachable via an explicit --backend cupy, on its own branch, to be
dropped if it doesn't pan out. See docs/OPTIMIZATION.md."""

import functools
import importlib.util
import logging

import numpy as np

from .models import to_sklearn_svr

logger = logging.getLogger(__name__)

# Default query-position chunk sizes per backend tier. Sized for the
# pretrained SVR's shape (605,187 support vectors x 360 features); see
# docs/PLANNING.md "Batching" for the memory-bound reasoning behind each.
# "cupy" reuses the "numpy" tier's conservative default rather than cuml's
# 2**20 -- unlike cuml.svm (which tiles the kernel matrix internally in
# C++ without ever materializing the whole thing), _build_cupy_predict_fn
# materializes a (query_chunk, sv_chunk)-shaped intermediate directly on
# the GPU, same as the NumPy tier does on the CPU, so it needs the same
# kind of conservative sizing -- unvalidated on real GPU memory, tune this
# up once tested on real hardware.
DEFAULT_QUERY_CHUNK = {"numpy": 4096, "sklearn": 50_000, "cuml": 2**20, "cupy": 4096}


class BackendUnavailable(RuntimeError):
    """Raised when an explicitly requested backend can't actually be used
    (rather than silently falling back to the next tier)."""


def _cuml_installed():
    return importlib.util.find_spec("cuml") is not None


def _cupy_installed():
    return importlib.util.find_spec("cupy") is not None


def _cuda_runtime_available():
    """Return whether CUDA is visible through CuPy, a cuML dependency."""
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        logger.debug("CuPy CUDA runtime availability probe failed", exc_info=True)
        return False


# RAPIDS/cuML's documented minimum since the 24.02 release -- see this
# module's docstring for the real-hardware confirmation of what happens
# below this (silently wrong results, not an error).
MIN_CUDA_COMPUTE_CAPABILITY = 70


def _cuda_compute_capability():
    """Returns the current CUDA device's compute capability as an int
    (e.g. 70 for 7.0, matching CuPy's own '70'-style string format), or
    None if it can't be determined (no GPU, CuPy not installed, etc.)."""
    try:
        import cupy

        return int(cupy.cuda.Device().compute_capability)
    except Exception:
        logger.debug("CuPy compute-capability probe failed", exc_info=True)
        return None


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

    cc = _cuda_compute_capability()
    if cc is not None and cc < MIN_CUDA_COMPUTE_CAPABILITY:
        logger.info(
            "GPU compute capability %.1f is below RAPIDS/cuML's minimum of %.1f "
            "(older GPUs aren't just unsupported, they can silently return wrong "
            "predictions rather than erroring -- see pydreg.backend's module "
            "docstring) -- falling back to CPU",
            cc / 10, MIN_CUDA_COMPUTE_CAPABILITY / 10,
        )
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


def _sklearn_cross_check_detail(dreg_model, sample, reference):
    """On a non-sklearn backend's smoke-test failure, also runs the CPU
    libsvm (scikit-learn) path on the same sample -- to_sklearn_svr()'s
    conversion independently agrees with the NumPy reference to ~1e-9 (see
    its docstring), so this pinpoints whether a divergence is specific to
    the failing backend's own GPU/library conversion, or is instead shared
    with any libsvm-style kernel evaluation (which would point at
    DREGModel.predict's own expanded-form squared-distance formula being
    the side that's actually wrong on this particular input, not the
    backend being tested). Best-effort: swallows its own failures rather
    than masking the original error with a second one."""
    try:
        sk_svr = to_sklearn_svr(dreg_model)
        sample_scaled = (sample - dreg_model.x_center) / dreg_model.x_scale
        sk_y = np.asarray(sk_svr.predict(sample_scaled)) * dreg_model.y_scale + dreg_model.y_center
        sk_max_abs = float(np.max(np.abs(sk_y - reference)))
    except Exception:
        logger.debug("sklearn cross-check itself failed", exc_info=True)
        return ""

    if sk_max_abs > 1e-4:
        return (
            f"; sklearn (CPU libsvm) on the same sample also diverges from the "
            f"NumPy reference (max_abs_diff={sk_max_abs:.6g}) -- this looks like a "
            "NumPy-reference-side issue (e.g. DREGModel.predict's expanded-form "
            "squared-distance formula losing precision on this input), not specific "
            "to this backend"
        )
    return (
        f"; sklearn (CPU libsvm) on the same sample agrees with the NumPy reference "
        f"(max_abs_diff={sk_max_abs:.6g}) -- the divergence looks specific to this backend"
    )


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
                detail = ""
                if backend_name != "sklearn" and candidate.shape == reference.shape:
                    detail = _sklearn_cross_check_detail(dreg_model, sample, reference)
                raise BackendUnavailable(
                    f"{backend_name} backend predictions do not match the NumPy reference "
                    f"on a first-batch smoke test (max_abs_diff={max_abs:.6g}){detail}; "
                    "use --backend numpy until this backend conversion is fixed"
                )
            validated = True
        return y

    return predict_fn


def _build_cupy_predict_fn(dreg_model, sv_chunk=32_768):
    """Returns predict_fn(X_scaled) -> y_scaled (both host NumPy arrays --
    matching _wrap_sklearn_like's expected interface, so it composes with
    the same scaling/unscaling wrapper and smoke test as the sklearn/cuml
    tiers) that evaluates DREGModel.predict's exact RBF dual-sum formula
    on a CuPy device array, chunked over support vectors the same way
    DREGModel.predict itself is chunked over the CPU. This is the *same
    formula*, not a separate from-scratch kernel implementation, so (unlike
    the cuml tier) there is no cuml.svm/libsvm conversion step that could
    diverge -- and CuPy's own array ops support compute capability >=3.0,
    below the >=7.0 floor cuml.svm silently gets wrong (see this module's
    docstring).

    The two matmuls (X @ SV.T and K @ coefs) are already cuBLAS GEMM calls
    -- about as fast as this gets without touching precision. The glue
    between them (sq_x + sq_sv - 2*cross, then exp(-gamma*...)) was
    originally ~5 separate elementwise kernel launches each reading/writing
    a full (query_chunk, sv_chunk) array to GPU global memory -- pure
    memory-bandwidth overhead on what's fundamentally a memory-bound step
    (same reason the NumPy tier is memory-bandwidth-, not compute-, bound;
    see docs/OPTIMIZATION.md "Batching"). This is fused into a single kernel
    that reads cross/sq_x/sq_sv once and writes K once, cutting that memory
    traffic roughly 5x with no formula or precision change.

    Fusion was first tried via @cp.fuse(), which produced a confirmed real
    divergence on actual GPU hardware (~3.5e-4, tripping
    _wrap_sklearn_like's smoke test -- sklearn agreed with the NumPy
    reference to ~5.5e-11 on the same sample, pinning the bug specifically
    to the fused cupy path). Two suspects, either of which fuse's JIT
    tracer could plausibly get wrong and neither confirmable without GPU
    access: gamma was passed as a runtime Python-float *argument* (fuse's
    tracer may not apply the same dtype-promotion guarantees eager CuPy
    ops do), and n_sv=605,187 isn't divisible by sv_chunk (the last chunk
    is a different, smaller shape than the rest -- a case fuse's
    shape-based kernel caching could mishandle). cupy.ElementwiseKernel
    below sidesteps both at once: every argument's dtype is declared
    explicitly (no promotion ambiguity), gamma is baked in as a literal
    rather than passed at all, and it has no shape-based tracing/caching --
    one compiled kernel, invoked generically for any broadcastable shape,
    the same mature mechanism CuPy's own internals use for this pattern.

    It also drops one (query_chunk, sv_chunk)-shaped device buffer
    entirely (the old separate `sqdist` intermediate no longer exists) --
    two live buffers of that shape per iteration (`cross`, `K`) instead of
    three, which is why sv_chunk's default could grow here without
    exceeding the original tier's peak memory footprint. Still a
    conservative starting point (unvalidated on real GPU memory beyond
    "didn't OOM") -- tune via build_scorer's cupy_sv_chunk / --cupy-sv-chunk
    once you have real headroom numbers for your GPU."""
    import cupy as cp

    SV = cp.asarray(dreg_model.SV)
    coefs = cp.asarray(dreg_model.coefs)
    sq_sv = cp.sum(SV**2, axis=1)
    gamma = dreg_model.gamma
    rho = dreg_model.rho
    n_sv = dreg_model.n_sv

    _rbf_from_cross = cp.ElementwiseKernel(
        "float64 cross, float64 sq_x, float64 sq_sv",
        "float64 out",
        f"out = exp(-{gamma!r} * (sq_x + sq_sv - 2 * cross))",
        "pydreg_rbf_from_cross",
    )

    def predict(X_scaled):
        X = cp.asarray(X_scaled)
        sq_x = cp.sum(X**2, axis=1)[:, None]
        y_scaled = cp.zeros(X.shape[0])
        for start in range(0, n_sv, sv_chunk):
            end = min(start + sv_chunk, n_sv)
            cross = X @ SV[start:end].T
            K = _rbf_from_cross(cross, sq_x, sq_sv[None, start:end])
            y_scaled += K @ coefs[start:end]
        y_scaled -= rho
        return cp.asnumpy(y_scaled)

    return predict


def build_scorer(dreg_model, backend=None, cupy_sv_chunk=None):
    """Builds (and caches, on `dreg_model._scorer_cache`) a Scorer for the
    requested backend. backend=None ("auto") picks the best available tier
    via detect_backend(). An explicit backend name that isn't usable raises
    BackendUnavailable rather than silently falling back -- a caller who
    asked for a specific backend wants a loud failure, not a silent
    slowdown on a job sized for that backend's throughput.

    cupy_sv_chunk: only used by the "cupy" tier -- how many support vectors
    (of 605,187) to evaluate per GPU kernel/GEMM call; None uses
    _build_cupy_predict_fn's own default. This is the main lever for
    trading GPU memory for fewer, larger (better-amortized) kernel launches
    -- see that function's docstring. Real GPU memory headroom varies by
    card, so this is deliberately left tunable rather than hardcoded."""
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
        cc = _cuda_compute_capability()
        if cc is not None and cc < MIN_CUDA_COMPUTE_CAPABILITY:
            raise BackendUnavailable(
                f"GPU compute capability {cc / 10:.1f} is below RAPIDS/cuML's minimum "
                f"of {MIN_CUDA_COMPUTE_CAPABILITY / 10:.1f} -- on unsupported hardware "
                "(e.g. Pascal) cuML doesn't error, it silently returns wrong predictions "
                "(confirmed on a real NVIDIA TITAN X, see pydreg.backend's module "
                "docstring); use --backend numpy"
            )
        try:
            gpu_model = cuml.svm.SVR.from_sklearn(to_sklearn_svr(dreg_model))
        except Exception as e:
            raise BackendUnavailable(f"cuml is installed but could not build a GPU model: {e}") from e
        predict_fn = _wrap_sklearn_like(dreg_model, gpu_model.predict, "cuml")

    elif resolved == "cupy":
        try:
            import cupy  # noqa: F401
        except ModuleNotFoundError as e:
            raise BackendUnavailable("cupy is not installed (pip install 'pydreg[gpu]')") from e
        try:
            cupy_kwargs = {} if cupy_sv_chunk is None else {"sv_chunk": cupy_sv_chunk}
            cupy_predict = _build_cupy_predict_fn(dreg_model, **cupy_kwargs)
        except Exception as e:
            raise BackendUnavailable(f"cupy is installed but could not build a GPU predict function: {e}") from e
        predict_fn = _wrap_sklearn_like(dreg_model, cupy_predict, "cupy")

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

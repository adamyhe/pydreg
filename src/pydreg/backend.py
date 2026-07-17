"""Tiered cupy -> scikit-learn -> NumPy scoring backend dispatch (see
docs/PLANNING.md "Backend dispatch" / "Batching"). pydreg.pipeline never
branches on backend -- it only ever calls a Scorer's uniform .predict().

Detection is lazy (never at import time -- importing cupy alone can take
a noticeable moment, a bad tax on every invocation including --help) and
cached once per process.

There used to be a fourth, GPU-only tier here, "cuml" (cuml.svm.SVR built
via from_sklearn()). It was dropped after real-hardware testing found a
confirmed, serious problem: RAPIDS/cuML dropped support for Pascal GPUs
(compute capability < 7.0) in the 24.02 release, and running a Pascal-era
cuML build on such hardware doesn't error, it silently returns wrong
predictions (RAPIDS's own deprecation notice: "use of a Pascal GPU will
either fail or return invalid results"). Confirmed end-to-end on a real
production run: cuml 26.06.00's SVR.from_sklearn()-built model diverged
from the NumPy reference by ~0.05 on an NVIDIA TITAN X (Pascal, compute
capability 6.1), while the *exact same bigWig inputs* on an A100 (compute
capability 8.0) ran clean. `cupy` (below) -- pydreg's own RBF kernel
implementation, not a routed-through third-party SVM library -- ran
correctly on that same TITAN X, and after this session's fusion/batching/
float32 work is now faster than cuml ever was, on both a TITAN Xp and an
A100. See docs/OPTIMIZATION.md and docs/PERF_LOG.md for the full
investigation and the decision to drop cuml entirely rather than keep
maintaining two GPU tiers.

"cupy" evaluates DREGModel.predict's exact RBF dual-sum formula directly
on a CuPy device array (see _build_cupy_predict_fn) -- being the same
formula as the already-validated NumPy tier, it carries none of the
cross-library conversion risk a routed-through SVM library would. CuPy's
own array ops support compute capability >=3.0, well below the >=7.0
floor cuml.svm needed -- this is now the auto-selected GPU tier whenever a
usable CUDA device is present, with no compute-capability gate needed at
all. This is exactly why _wrap_sklearn_like's first-batch smoke test
exists -- to catch any future conversion/precision issue before it
reaches an output file, the same mechanism that caught the real cuml
divergence above and a real cp.fuse() bug during this tier's own
development (see docs/PERF_LOG.md)."""

import functools
import importlib.util
import logging

import numpy as np

from .models import to_sklearn_svr

logger = logging.getLogger(__name__)

# Default query-position chunk sizes per backend tier. Sized for the
# pretrained SVR's shape (605,187 support vectors x 360 features); see
# docs/PLANNING.md "Batching" for the memory-bound reasoning behind each.
# "cupy": _build_cupy_predict_fn materializes a (query_chunk, sv_chunk)
# -shaped intermediate directly on the GPU, same as the NumPy tier does on
# the CPU, so it gets the same kind of conservative sizing.
DEFAULT_QUERY_CHUNK = {"numpy": 4096, "sklearn": 50_000, "cupy": 4096}


class BackendUnavailable(RuntimeError):
    """Raised when an explicitly requested backend can't actually be used
    (rather than silently falling back to the next tier)."""


def _cupy_installed():
    return importlib.util.find_spec("cupy") is not None


def _cuda_runtime_available():
    """Return whether CUDA is visible through CuPy."""
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        logger.debug("CuPy CUDA runtime availability probe failed", exc_info=True)
        return False


# _wrap_sklearn_like's default smoke-test atol (1e-4) assumes
# near-double-precision agreement; the "cupy" tier's GEMMs/kernel are
# deliberately float32 (see _build_cupy_predict_fn), so it gets this looser
# one instead. Confirmed on real hardware: a genuine max_abs_diff of
# ~2.3e-4 to ~5.4e-4 against the float64 NumPy reference, with sklearn
# independently agreeing with that same reference to ~6e-11 on the same sample --
# pinning the divergence to cupy's own float32 arithmetic (specifically the
# expanded-form squared-distance formula's cancellation sensitivity,
# amplified by float32's much smaller precision budget), not a conversion
# bug. See docs/PERF_LOG.md's 2026-07-15 entry.
CUPY_SMOKE_TEST_ATOL = 1e-3


@functools.lru_cache(maxsize=1)
def detect_backend():
    """Probes once per process and returns "cupy" or "numpy" -- the best
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
    required for it, and as the input to _sklearn_cross_check_detail's
    cupy-smoke-test diagnostic."""
    if not _cupy_installed():
        logger.info("cupy not installed -- install pydreg[gpu] for GPU scoring")
        return "numpy"

    if not _cuda_runtime_available():
        logger.info(
            "cupy installed but no usable CUDA GPU detected at runtime -- falling back to CPU"
        )
        return "numpy"

    return "cupy"


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
        sk_y = (
            np.asarray(sk_svr.predict(sample_scaled)) * dreg_model.y_scale
            + dreg_model.y_center
        )
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


def _wrap_sklearn_like(dreg_model, sk_predict, backend_name, rtol=1e-4, atol=1e-4):
    """Both the sklearn and cupy tiers predict in the SVR's internal scaled
    feature space and need the same x-scale / y-unscale wrapping DREGModel
    itself does -- see pydreg.models.DREGModel.predict.

    rtol/atol: smoke-test tolerance against the NumPy reference. The
    default (1e-4/1e-4) assumes near-double-precision agreement, true for
    sklearn (genuinely float64). build_scorer() passes the looser
    CUPY_SMOKE_TEST_ATOL for "cupy" specifically -- see that constant's
    comment for why (deliberately float32, not a bug)."""
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
                or not np.allclose(candidate, reference, rtol=rtol, atol=atol)
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
    the same scaling/unscaling wrapper and smoke test as the sklearn tier)
    that evaluates DREGModel.predict's exact RBF dual-sum formula on a
    CuPy device array, chunked over support vectors the same way
    DREGModel.predict itself is chunked over the CPU. This is the *same
    formula*, not a separate from-scratch kernel implementation, so there
    is no separate SVM-library conversion step that could diverge -- and
    CuPy's own array ops support compute capability >=3.0 (see this
    module's docstring for why that matters).

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
    once you have real headroom numbers for your GPU.

    The two GEMMs (X @ SV.T, K @ coefs) and the fused RBF kernel run in
    float32, not float64 -- deliberately, not a shortcut. The current
    pretrained models are trained via Rgtsvm (dREG's GPU SVM tool; e1071
    is now just an S3-compatibility shim around it, not what actually fits
    the model), and Rgtsvm's own CUDA implementation has no double-
    precision code path at all: gamma/coef0/degree/cost are hard-downcast
    to C++ float (Rgtsvm.cpp), the support-vector matrix is stored
    internally as float (svm.hpp's SparseVector), and the optimizer's own
    CUDA_FLOAT_DOUBLE type defaults to float unless a build flag dREG's
    build never sets is defined (cuda_helpers.hpp) -- see
    docs/OPTIMIZATION.md for the full trace. So these weights' real
    accuracy ceiling was already float32 before ever reaching pydreg;
    float32 inference here isn't trading away precision the model
    actually has, and arguably matches how real GPU-accelerated dREG
    behaved more closely than float64 inference did. This does NOT extend
    to the CPU tiers (NumPy/scikit-learn) -- libsvm's actual predict path
    (what both e1071's CPU mode and sklearn.svm.SVR use) is genuinely
    double-precision throughout, so those stay float64.

    y_scaled accumulates in float64 despite K/coefs being float32 -- each
    chunk's small (query_chunk,)-sized contribution is upcast before
    adding, cheap insurance against cross-chunk summation error over the
    ~19 chunks this loop runs, independent of the float32 GEMM/kernel
    itself."""
    import cupy as cp

    SV = cp.asarray(dreg_model.SV, dtype=cp.float32)
    coefs = cp.asarray(dreg_model.coefs, dtype=cp.float32)
    sq_sv = cp.sum(SV**2, axis=1)
    gamma = dreg_model.gamma
    rho = dreg_model.rho
    n_sv = dreg_model.n_sv

    _rbf_from_cross = cp.ElementwiseKernel(
        "float32 cross, float32 sq_x, float32 sq_sv",
        "float32 out",
        # the trailing "f" on the gamma literal matters: an unsuffixed
        # float literal is `double` in C/CUDA, which would silently
        # promote this whole expression (and expf's argument) back to
        # double despite every array here being float32 -- defeating the
        # point.
        f"out = expf(-{gamma!r}f * (sq_x + sq_sv - 2 * cross))",
        "pydreg_rbf_from_cross",
    )

    def predict(X_scaled):
        X = cp.asarray(X_scaled, dtype=cp.float32)
        sq_x = cp.sum(X**2, axis=1)[:, None]
        y_scaled = cp.zeros(X.shape[0], dtype=cp.float64)
        for start in range(0, n_sv, sv_chunk):
            end = min(start + sv_chunk, n_sv)
            cross = X @ SV[start:end].T
            K = _rbf_from_cross(cross, sq_x, sq_sv[None, start:end])
            y_scaled += (K @ coefs[start:end]).astype(cp.float64)
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
        raise ValueError(
            f"unknown backend {resolved!r}, expected one of {sorted(DEFAULT_QUERY_CHUNK)}"
        )

    if resolved in dreg_model._scorer_cache:
        return dreg_model._scorer_cache[resolved]

    if resolved == "cupy":
        try:
            import cupy  # noqa: F401
        except ModuleNotFoundError as e:
            raise BackendUnavailable(
                "cupy is not installed (pip install 'pydreg[gpu]')"
            ) from e
        try:
            cupy_kwargs = {} if cupy_sv_chunk is None else {"sv_chunk": cupy_sv_chunk}
            cupy_predict = _build_cupy_predict_fn(dreg_model, **cupy_kwargs)
        except Exception as e:
            raise BackendUnavailable(
                f"cupy is installed but could not build a GPU predict function: {e}"
            ) from e
        predict_fn = _wrap_sklearn_like(dreg_model, cupy_predict, "cupy", atol=CUPY_SMOKE_TEST_ATOL)

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

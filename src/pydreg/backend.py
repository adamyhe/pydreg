"""Tiered cupy/mlx -> NumPy scoring backend dispatch (see
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

There is also "mlx", an Apple Silicon GPU tier for machines with neither
NVIDIA hardware nor cupy applicable at all -- see _build_mlx_predict_fn for
the same near-verbatim RBF dual-sum formula run through Apple's MLX array
framework instead of CuPy. Confirmed on real Apple M4 hardware during
development: ~22x faster than the NumPy tier at a realistic query/support-
vector batch size, agreeing with the NumPy reference to ~1e-5 on synthetic
data of realistic shape (605,187 SVs x 360 features) -- see
docs/OPTIMIZATION.md and docs/PERF_LOG.md.

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
# the CPU, so it gets the same kind of conservative sizing. "mlx" does the
# same thing (see _build_mlx_predict_fn), so it reuses the same default.
DEFAULT_QUERY_CHUNK = {"numpy": 4096, "sklearn": 50_000, "cupy": 4096, "mlx": 4096}


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


def _mlx_installed():
    return importlib.util.find_spec("mlx") is not None


def _mlx_gpu_available():
    """Return whether MLX sees a usable Metal GPU device. mlx's PyPI wheels
    only target macOS on Apple Silicon at all (see pyproject.toml's `mlx`
    extra), so unlike cupy there's no separate "installed but no runtime"
    split to speak of in practice -- this still probes for real rather than
    assuming, the same defensive shape as _cuda_runtime_available, in case a
    future build ever ships for a device without a working GPU (e.g. a
    software-rendering fallback, or running inside a VM without GPU
    passthrough)."""
    try:
        import mlx.core as mx

        return mx.metal.is_available()
    except Exception:
        logger.debug("MLX Metal availability probe failed", exc_info=True)
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

# Same float32 catastrophic-cancellation story as CUPY_SMOKE_TEST_ATOL above
# (the expanded-form squared-distance formula, see _build_cupy_predict_fn's
# docstring) -- MLX's Metal GPU backend hits the identical precision ceiling
# for the identical reason (it's the same formula), not a separately
# re-measured number. Unlike cupy, float32 isn't even a choice for this
# tier: Metal has no float64 support at all (confirmed on real Apple M4
# hardware -- a plain float64 matmul on the GPU device raises
# `ValueError: float64 is not supported on the GPU`), so _build_mlx_predict_fn
# couldn't run this computation in float64 even if that were desired.
MLX_SMOKE_TEST_ATOL = CUPY_SMOKE_TEST_ATOL


# Smoke-test atol for either GPU tier when TF32 is actually in effect
# (`--tf32`, build_scorer(tf32=True)). Measured on a real Apple M5 with
# MLX's TF32 matmuls: max |error| 2.59e-3 over 16,384 real Jurkat ChRO-seq
# positions, 1.6e-3 on the smoke test's own 8-row sample. The cupy tier's
# TF32 (cuBLAS CUBLAS_COMPUTE_32F_FAST_TF32) keeps the same 10-bit
# mantissa on the same formula, so the same bound is used there. Measured
# on a real L40S and A100 (identical on both): max |error| 1.84e-3 over
# 200,704 real Jurkat ChRO-seq positions with the fused RBF+GEMV kernel.
TF32_SMOKE_TEST_ATOL = 5e-3


def _configure_mlx_tf32(enabled):
    """Sets whether MLX's float32 GEMMs run at TF32 input precision, checks
    what actually took effect, and returns that (True = TF32 in effect).
    Called before the mlx tier's first matmul.

    MLX runs float32 matmuls with TF32-class inputs (10-bit mantissa) by
    default on GPUs that have the hardware for it -- the Apple M5's neural
    accelerators ("NAX" kernels), not the M4. pydreg turns that off unless
    asked (`--tf32`): on a real M5 (mlx 0.32.0) TF32 put scores on 16,384
    real Jurkat ChRO-seq positions up to 2.6e-3 off the float64 reference,
    failed the 1e-3 smoke test on every real run, and dropped full-run peak
    agreement with real dREG from 0.9997 to 0.9953 Jaccard. With TF32 off
    the GEMM is ~1.9x slower on the M5 (~1.57x end to end) but matches
    NumPy float32. See docs/PERF_LOG.md's 2026-11-20 entry.

    MLX reads MLX_ENABLE_TF32 once, at the first float32 matmul in the
    process, and caches it (mlx/utils.h's enable_tf32()). The flag is the
    source of truth, so this assigns the variable either way (overriding a
    value already in the environment), then measures with a small matmul
    whose exact answer TF32 can't represent. If TF32 was wanted off but is
    on (mlx already ran a TF32 matmul earlier in this process), raise
    BackendUnavailable with the fix rather than letting the smoke test fail
    unexplained. If it was wanted on but isn't (no TF32 hardware, e.g. an
    M4, or an earlier full-precision matmul locked it off), that only costs
    speed, so just log it. A (32, 360, 32) product is large enough to take
    the TF32 path on an M5 (anything from (8, 360, 8) up did; (1, 360, 8)
    did not)."""
    import os

    import mlx.core as mx

    os.environ["MLX_ENABLE_TF32"] = "1" if enabled else "0"
    k = 360
    a = np.full((32, k), 1 + 2.0**-12, dtype=np.float32)
    b = np.ones((32, k), dtype=np.float32)
    got = np.array(mx.array(a) @ mx.array(b).T, dtype=np.float64)
    err = float(np.max(np.abs(got - k * (1 + 2.0**-12))))
    in_effect = err > 1e-3
    if not enabled and in_effect:
        raise BackendUnavailable(
            f"mlx is running float32 matmuls at reduced (TF32) precision "
            f"(probe error {err:.3g}); set MLX_ENABLE_TF32=0 in the environment "
            "before starting Python, or pass --tf32 to accept it. MLX reads it "
            "once, at the first matmul."
        )
    if enabled and not in_effect:
        logger.info(
            "--tf32 has no effect on this mlx device (no TF32 hardware, or an "
            "earlier full-precision matmul in this process locked it off)"
        )
    return in_effect


def _set_cupy_compute_type(compute_type):
    """Sets cupy's float32 compute type and returns the one that actually
    took effect. CuPy falls back to DEFAULT, with a warning, on GPUs below
    compute capability 8.0; the warning is silenced here because callers
    check the returned value instead."""
    import warnings

    import cupy

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cupy._core.set_compute_type(np.float32, compute_type)
    return cupy._core.get_compute_type(np.float32)


def _configure_cupy_tf32(enabled):
    """Returns the float32 compute type a cupy scorer should run its
    `X @ SV.T` cuBLAS GEMM under, and whether that is TF32. Leaves cupy's
    own setting as it found it.

    CuPy reads its float32 compute type on every matmul
    (cupy._core._routines_linalg.get_compute_type): COMPUTE_TYPE_TF32
    selects CUBLAS_COMPUTE_32F_FAST_TF32, COMPUTE_TYPE_DEFAULT plain FP32.
    That setting is process-wide, so _build_cupy_predict_fn applies the
    returned type around each predict call and restores the previous one,
    rather than setting it once at build time. Setting it once let a
    float32 scorer silently run at TF32 after a TF32 scorer was built in the
    same process, while still reporting tf32=False (measured on a real L40S
    and A100: max score change 1.7e-3, identical to the TF32 scorer's
    output). Applying it per call also makes the flag win over CuPy's own
    CUPY_TF32 env var in both directions.

    TF32 is probed here by setting it and reading it back, because CuPy
    falls back to DEFAULT on GPUs below compute capability 8.0 (Pascal,
    Volta, Turing), which have no TF32. Off by default for the same fidelity
    reason as the mlx tier (_configure_mlx_tf32). The GEMM speedup is large
    on both datacenter and Ada cards: measured 4.9x on an A100 and 2.4x on
    an L40S, which with the fused RBF+GEMV kernel is 2.99x and 1.43x per
    scoring chunk, and 1.48x and 1.25x end to end on Jurkat_ChROseq_1. Peak
    agreement with real dREG drops from 0.99976 to 0.99620 Jaccard on both
    cards (see docs/PERF_LOG.md's 2026-11-22 entry)."""
    import cupy

    linalg = cupy._core._routines_linalg
    previous = cupy._core.get_compute_type(np.float32)
    wanted = linalg.COMPUTE_TYPE_TF32 if enabled else linalg.COMPUTE_TYPE_DEFAULT
    try:
        compute_type = _set_cupy_compute_type(wanted)
    finally:
        _set_cupy_compute_type(previous)
    in_effect = compute_type == linalg.COMPUTE_TYPE_TF32
    if enabled and not in_effect:
        logger.warning(
            "--tf32 has no effect on this GPU (TF32 needs compute capability 8.0+)"
        )
    return compute_type, in_effect


@functools.lru_cache(maxsize=1)
def detect_backend():
    """Probes once per process and returns "cupy", "mlx", or "numpy" -- the
    best backend actually usable right now. cupy (NVIDIA/CUDA) is checked
    first, then mlx (Apple Silicon/Metal) -- in practice at most one of the
    two is ever actually installable on a given machine (see each's
    pyproject.toml platform marker), so this ordering is mostly about which
    diagnostic message a mixed/unusual environment sees first, not a real
    preference between them.

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
        logger.debug("cupy not installed -- install pydreg[gpu] for GPU scoring")
    elif not _cuda_runtime_available():
        logger.warning(
            "cupy installed but no usable CUDA GPU detected at runtime -- falling back to CPU"
        )
    else:
        return "cupy"

    if not _mlx_installed():
        logger.debug("mlx not installed -- install pydreg[mlx] for Apple Silicon GPU scoring")
    elif not _mlx_gpu_available():
        logger.warning(
            "mlx installed but no usable Metal GPU detected at runtime -- falling back to CPU"
        )
    else:
        return "mlx"

    return "numpy"


class Scorer:
    """Uniform `.predict(X_chunk) -> np.ndarray` wrapper hiding backend
    differences from callers. `tf32` records whether TF32 GEMMs are actually
    in effect (requested and supported), for logging."""

    def __init__(self, backend, predict_fn, tf32=False):
        self.backend = backend
        self._predict_fn = predict_fn
        self.tf32 = tf32

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


# One thread block per query row of `cross` (C-contiguous, n_rows x n_cols):
# each thread strides over the row's support vectors accumulating
# coef_j * exp(-gamma * (sq_x + sq_sv_j - 2*cross_ij)) in float32, then the
# block reduces those partial sums in double and adds the row's total into
# y. Fuses what used to be an ElementwiseKernel (writing the full RBF matrix
# K) followed by a cuBLAS GEMV (reading it back), so K is never
# materialized -- see _build_cupy_predict_fn. Every argument's type is
# fixed by this signature, so the caller must pass gamma as np.float32 and
# n_cols as np.int32 (RawKernel does no conversion; a wrong scalar width
# would be read as garbage, not promoted).
_RBF_GEMV_THREADS = 256
_RBF_GEMV_SRC = r"""
extern "C" __global__
void pydreg_rbf_gemv(const float* __restrict__ cross,
                     const float* __restrict__ sq_x,
                     const float* __restrict__ sq_sv,
                     const float* __restrict__ coefs,
                     const float gamma,
                     const int n_cols,
                     double* __restrict__ y) {
    const int row = blockIdx.x;
    const float* crow = cross + (size_t)row * n_cols;
    const float sx = sq_x[row];
    float acc = 0.0f;
    for (int j = threadIdx.x; j < n_cols; j += blockDim.x) {
        acc += coefs[j] * expf(-gamma * (sx + sq_sv[j] - 2.0f * crow[j]));
    }
    __shared__ double partial[%d];
    partial[threadIdx.x] = (double)acc;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) partial[threadIdx.x] += partial[threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[row] += partial[0];
}
""" % _RBF_GEMV_THREADS


def _build_cupy_predict_fn(dreg_model, sv_chunk=16_384, tf32=False):
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

    Per sv-chunk, X @ SV.T is a cuBLAS sgemm writing `cross`, and a single
    fused RawKernel (_RBF_GEMV_SRC) then reads `cross` once, applies the RBF
    formula exp(-gamma * (sq_x + sq_sv - 2*cross)) and multiplies by coefs
    and reduces over support vectors in the same pass, accumulating straight
    into y_scaled. The (query_chunk, sv_chunk) kernel matrix K is never
    written to GPU memory.

    History, because each step was a measured fix: the glue between the two
    GEMMs was first ~5 separate elementwise launches, each round-tripping a
    full (query_chunk, sv_chunk) array; @cp.fuse() was tried and produced a
    real ~3.5e-4 divergence on GPU hardware (gamma passed as a runtime
    Python float, and a ragged final sv-chunk, were the suspects); then a
    cupy.ElementwiseKernel (explicit dtypes, gamma baked in as a literal)
    wrote K, followed by a cuBLAS GEMV reading it back. Fusing the RBF and
    GEMV steps into the RawKernel halves the per-sv-chunk memory traffic
    (the GEMM writes `cross` once and the kernel reads it once), and on a
    real RTX 3080 took a 4096-query chunk from 138.0 ms to 114.6 ms (1.20x),
    leaving ~13 ms above the sgemm's own 102 ms, about the cost of reading
    `cross` once at the card's bandwidth. It is also more accurate
    (max |error| vs the float64 NumPy reference 3.5e-5, against 2.6e-4),
    because K is no longer rounded to float32 before the dot product and
    the per-row reduction runs in double. RawKernel has the same
    no-ambiguity property ElementwiseKernel did: every argument type is
    fixed by the CUDA signature (gamma is passed as an explicit np.float32,
    n_cols as np.int32), with no tracing or shape-based caching. Fusing the
    GEMM itself as well was considered and not pursued: it could save at
    most the ~13 ms `cross` write, but only if a hand-written GEMM matched
    cuBLAS. See docs/PERF_LOG.md's 2026-11-18 entries.

    sv_chunk's default (16,384, lowered from an earlier, unvalidated
    32,768) is chosen from real measurements, not a guess: a full sweep
    on a real TITAN Xp (4,096 through 65,536) and a real A100 (4,096
    through 1,048,576, i.e. one single sv-chunk covering all 605,187
    support vectors) found `predict()`'s own time completely flat across
    the whole range on both cards -- this step is memory-bandwidth-bound,
    not launch-overhead-bound, so growing sv_chunk buys nothing, while
    16,384 (like 4,096-8,192 below it) already sits at the fixed
    pre-kernel-launch VRAM floor (~1844 MB measured on the TITAN Xp,
    vs. 3379 MB at the old 32,768 default) with no throughput cost -- see
    docs/PERF_LOG.md's 2026-08-20 entry for the full investigation. Still
    tunable via build_scorer's cupy_sv_chunk / --cupy-sv-chunk if your
    card's real headroom or bandwidth profile differs.

    The GEMM (X @ SV.T) and the fused RBF/GEMV kernel's per-element math
    run in float32, not float64 -- deliberately, not a shortcut. The current
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

    y_scaled accumulates in float64 despite cross/coefs being float32: each
    thread block reduces its float32 per-thread partial sums in double and
    adds them into the float64 y_scaled, cheap insurance against summation
    error across the 605,187 support vectors (37 sv-chunks at the default
    sv_chunk), independent of the float32 GEMM itself.

    tf32: whether the `X @ SV.T` GEMM runs at TF32. _configure_cupy_tf32
    picks the compute type once, and each predict call applies it and then
    restores cupy's previous setting, so scorers built with different tf32
    values can share a process. Off by default. The returned function's
    `.tf32` says whether TF32 is in effect."""
    import cupy as cp

    compute_type, tf32_in_effect = _configure_cupy_tf32(tf32)
    SV = cp.asarray(dreg_model.SV, dtype=cp.float32)
    coefs = cp.asarray(dreg_model.coefs, dtype=cp.float32)
    sq_sv = cp.sum(SV**2, axis=1)
    gamma = np.float32(dreg_model.gamma)
    rho = dreg_model.rho
    n_sv = dreg_model.n_sv

    _rbf_gemv = cp.RawKernel(_RBF_GEMV_SRC, "pydreg_rbf_gemv")

    def predict(X_scaled):
        # cupy's compute type is process-wide; apply this scorer's for the
        # duration of the call only (see _configure_cupy_tf32).
        previous = cp._core.get_compute_type(np.float32)
        _set_cupy_compute_type(compute_type)
        try:
            return _predict(X_scaled)
        finally:
            _set_cupy_compute_type(previous)

    def _predict(X_scaled):
        X = cp.asarray(X_scaled, dtype=cp.float32)
        n = X.shape[0]
        sq_x = cp.sum(X**2, axis=1)
        y_scaled = cp.zeros(n, dtype=cp.float64)
        if n == 0:
            return cp.asnumpy(y_scaled)
        for start in range(0, n_sv, sv_chunk):
            end = min(start + sv_chunk, n_sv)
            # The kernel indexes cross as dense row-major; ascontiguousarray
            # is a no-op for cuBLAS's C-ordered output but guards the layout.
            cross = cp.ascontiguousarray(X @ SV[start:end].T)
            _rbf_gemv(
                (n,),
                (_RBF_GEMV_THREADS,),
                (
                    cross,
                    sq_x,
                    sq_sv[start:end],
                    coefs[start:end],
                    gamma,
                    np.int32(end - start),
                    y_scaled,
                ),
            )
        y_scaled -= rho
        return cp.asnumpy(y_scaled)

    predict.tf32 = tf32_in_effect
    return predict


# Metal counterpart to _RBF_GEMV_SRC, the body of an mx.fast.metal_kernel
# (MLX generates the signature from input_names/output_names, so argument
# types come from the arrays passed in -- gamma is a float32 (1,) array, the
# same explicitly-typed-scalar role np.float32 plays for the RawKernel). One
# threadgroup per query row of `cross`: each thread strides over the row's
# support vectors accumulating coef_j * exp(-gamma * (sq_x + sq_sv_j -
# 2*cross_ij)), then simd_sum reduces within each 32-wide SIMD group and
# the first group reduces the per-group partials. Metal has no double, so
# unlike the CUDA kernel the in-row reduction is float32; cross-chunk
# accumulation is still float64, on the host (see _build_mlx_predict_fn).
_RBF_GEMV_METAL_THREADS = 256
_RBF_GEMV_METAL_SRC = """
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint n_cols = cross_shape[1];
    const device float* crow = cross + (size_t)row * n_cols;
    float sx = sq_x[row];
    float g = gamma[0];
    float acc = 0.0f;
    for (uint j = tid; j < n_cols; j += %(threads)d) {
        acc += coefs[j] * metal::exp(-g * (sx + sq_sv[j] - 2.0f * crow[j]));
    }
    acc = simd_sum(acc);
    threadgroup float partial[%(groups)d];
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    if (lane == 0) partial[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float v = lane < %(groups)d ? partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) out[row] = v;
    }
""" % {"threads": _RBF_GEMV_METAL_THREADS, "groups": _RBF_GEMV_METAL_THREADS // 32}


def _build_mlx_predict_fn(dreg_model, sv_chunk=16_384, tf32=False):
    """Apple Silicon counterpart to _build_cupy_predict_fn -- same near-
    verbatim port of DREGModel.predict's chunked RBF dual-sum formula (same
    expanded squared-distance trick, same chunking over support vectors),
    evaluated on an MLX device array (Apple's own array framework, running
    on the machine's Metal GPU) instead of a CuPy one. Same rationale for
    being a from-scratch port of the *formula* rather than a routed-through
    third-party SVM library applies here too -- see that function's and
    this module's docstrings.

    Per sv-chunk, X @ SV.T is a plain MLX matmul (an MLX Metal GEMM) writing
    `cross`, and a single fused mx.fast.metal_kernel (_RBF_GEMV_METAL_SRC)
    then reads `cross` once, applies the RBF formula, multiplies by coefs
    and reduces over support vectors -- the same design as the cupy tier's
    RawKernel. The (query_chunk, sv_chunk) kernel matrix K is never written
    to GPU memory.

    History: this tier used to wrap the elementwise glue (sq_x + sq_sv -
    2*cross, then exp(-gamma*...)) and the final K @ coefs matmul in one
    `mx.compile`d function. mx.compile fuses elementwise ops, but not into a
    matmul, so K was still materialized and read back by a second GEMM. On
    a real M4 (real SVR, 4096-query chunk, sv_chunk 16,384), the fused
    kernel took a call from ~1160-1200 ms to ~910 ms (~1.3x), against an
    ~813 ms floor for the cross GEMMs alone, and max |error| vs the float64
    reference dropped from 1.2e-3 to 7.3e-4 (y_scaled space, synthetic
    inputs), because K is no longer rounded to float32 before a second
    GEMM. See docs/PERF_LOG.md's 2026-11-18 mlx entry.

    Unlike cupy, float32 for the GEMM/kernel isn't a deliberate speed/
    precision tradeoff here -- it's the only option MLX's GPU backend
    supports at all (float64 arrays raise outright on the Metal device, see
    MLX_SMOKE_TEST_ATOL's comment), and Metal Shading Language has no
    double either, so the in-row reduction is float32 too. There is no
    float64 accumulator MLX can hold on-device, so this function pulls each
    chunk's small (query_chunk,)-sized result back to a host NumPy float64
    array and accumulates there instead: one (query_chunk,) vector per
    iteration, not the full kernel matrix, giving the same cross-chunk
    summation insurance cupy's float64 accumulator does. (Batching several
    chunks per host sync was measured and made no difference.)

    sv_chunk's default (16,384, lowered from an earlier, unvalidated
    32,768 once a real `--mlx-sv-chunk` sweep confirmed it) is still the
    fastest value with the fused kernel (8,192: 937 ms, 16,384: 908 ms,
    32,768: 914 ms, 65,536: 944 ms on a real M4). An unchunked call
    (sv_chunk >= n_sv) can crash outright: Metal enforces a hard per-buffer
    allocation ceiling (`mx.device_info()["max_buffer_length"]`, ~9.53GB on
    this M4, independent of total unified memory) that a `(query_chunk=4096,
    sv_chunk=605_187)` float32 `cross` buffer (~9.92GB) exceeds. See
    docs/OPTIMIZATION.md and docs/PERF_LOG.md's 2026-08-22 entry for the
    original sweep.

    tf32: configured first (_configure_mlx_tf32), off by default: MLX turns
    TF32 on by default on M5-class GPUs, and it costs this formula
    measurable fidelity. Done here rather than in build_scorer so every
    caller of this function gets full float32 unless it asks otherwise.
    The returned function's `.tf32` says whether TF32 is in effect."""
    import mlx.core as mx

    tf32_in_effect = _configure_mlx_tf32(tf32)
    SV = mx.array(dreg_model.SV, dtype=mx.float32)
    coefs = mx.array(dreg_model.coefs, dtype=mx.float32)
    sq_sv = mx.sum(SV**2, axis=1)
    gamma = mx.array([dreg_model.gamma], dtype=mx.float32)
    rho = dreg_model.rho
    n_sv = dreg_model.n_sv

    _rbf_gemv = mx.fast.metal_kernel(
        name="pydreg_rbf_gemv",
        input_names=["cross", "sq_x", "sq_sv", "coefs", "gamma"],
        output_names=["out"],
        source=_RBF_GEMV_METAL_SRC,
    )

    def predict(X_scaled):
        n = X_scaled.shape[0]
        y_scaled = np.zeros(n, dtype=np.float64)
        if n == 0:
            return y_scaled
        X = mx.array(X_scaled, dtype=mx.float32)
        sq_x = mx.sum(X**2, axis=1)
        for start in range(0, n_sv, sv_chunk):
            end = min(start + sv_chunk, n_sv)
            (chunk,) = _rbf_gemv(
                inputs=[X @ SV[start:end].T, sq_x, sq_sv[start:end], coefs[start:end], gamma],
                grid=(n * _RBF_GEMV_METAL_THREADS, 1, 1),
                threadgroup=(_RBF_GEMV_METAL_THREADS, 1, 1),
                output_shapes=[(n,)],
                output_dtypes=[mx.float32],
            )
            y_scaled += np.array(chunk, dtype=np.float64)
        y_scaled -= rho
        return y_scaled

    predict.tf32 = tf32_in_effect
    return predict


def build_scorer(dreg_model, backend=None, cupy_sv_chunk=None, mlx_sv_chunk=None, tf32=False):
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
    card, so this is deliberately left tunable rather than hardcoded.

    mlx_sv_chunk: the same lever as cupy_sv_chunk, but for the "mlx" tier
    -- how many support vectors to evaluate per Metal kernel/GEMM call;
    None uses _build_mlx_predict_fn's own default.

    tf32: let the GPU tiers ("cupy", "mlx") run their float32 `X @ SV.T`
    GEMM at TF32 input precision, on hardware that has it (Apple M5,
    NVIDIA compute capability 8.0+). Off by default: end to end it was
    ~1.48x faster on an A100, ~1.25x on an L40S and ~1.57x on an M5, but it
    dropped peak agreement with real dREG from 0.99976 to 0.99620 Jaccard
    on the CUDA cards (0.9997 to 0.9953 on the M5), and it makes results
    depend on the GPU model. When
    TF32 is actually in effect the smoke test uses TF32_SMOKE_TEST_ATOL
    instead of the tier's float32 one. Ignored, with a warning, on the CPU
    tiers. Off also overrides CUPY_TF32 / MLX_ENABLE_TF32 if set."""
    resolved = backend or detect_backend()
    if resolved not in DEFAULT_QUERY_CHUNK:
        raise ValueError(
            f"unknown backend {resolved!r}, expected one of {sorted(DEFAULT_QUERY_CHUNK)}"
        )

    cache_key = (resolved, bool(tf32))
    if cache_key in dreg_model._scorer_cache:
        return dreg_model._scorer_cache[cache_key]

    tf32_in_effect = False
    if tf32 and resolved not in ("cupy", "mlx"):
        logger.warning("--tf32 only affects the cupy and mlx backends; ignored for %s", resolved)

    if resolved == "cupy":
        try:
            import cupy  # noqa: F401
        except ModuleNotFoundError as e:
            raise BackendUnavailable(
                "cupy is not installed (pip install 'pydreg[gpu]')"
            ) from e
        try:
            cupy_kwargs = {} if cupy_sv_chunk is None else {"sv_chunk": cupy_sv_chunk}
            cupy_predict = _build_cupy_predict_fn(dreg_model, tf32=tf32, **cupy_kwargs)
            tf32_in_effect = getattr(cupy_predict, "tf32", False)
        except Exception as e:
            raise BackendUnavailable(
                f"cupy is installed but could not build a GPU predict function: {e}"
            ) from e
        predict_fn = _wrap_sklearn_like(
            dreg_model, cupy_predict, "cupy",
            atol=TF32_SMOKE_TEST_ATOL if tf32_in_effect else CUPY_SMOKE_TEST_ATOL,
        )

    elif resolved == "mlx":
        try:
            import mlx.core  # noqa: F401
        except ModuleNotFoundError as e:
            raise BackendUnavailable(
                "mlx is not installed (pip install 'pydreg[mlx]')"
            ) from e
        try:
            mlx_kwargs = {} if mlx_sv_chunk is None else {"sv_chunk": mlx_sv_chunk}
            mlx_predict = _build_mlx_predict_fn(dreg_model, tf32=tf32, **mlx_kwargs)
            tf32_in_effect = getattr(mlx_predict, "tf32", False)
        except BackendUnavailable:
            raise
        except Exception as e:
            raise BackendUnavailable(
                f"mlx is installed but could not build a GPU predict function: {e}"
            ) from e
        predict_fn = _wrap_sklearn_like(
            dreg_model, mlx_predict, "mlx",
            atol=TF32_SMOKE_TEST_ATOL if tf32_in_effect else MLX_SMOKE_TEST_ATOL,
        )

    elif resolved == "sklearn":
        try:
            sk_model = to_sklearn_svr(dreg_model)
        except ModuleNotFoundError as e:
            raise BackendUnavailable("scikit-learn is not installed") from e
        predict_fn = _wrap_sklearn_like(dreg_model, sk_model.predict, "sklearn")

    else:  # numpy
        predict_fn = dreg_model.predict

    scorer = Scorer(resolved, predict_fn, tf32=tf32_in_effect)
    dreg_model._scorer_cache[cache_key] = scorer
    return scorer

import logging
import os
import re
import subprocess
import sys
import types

import numpy as np
import pytest

from pydreg import backend
from pydreg.models import DREGModel


def test_detect_backend_reports_missing_cupy(monkeypatch, caplog):
    backend.detect_backend.cache_clear()
    monkeypatch.setattr(backend, "_cupy_installed", lambda: False)
    # Isolates the cupy-fallback path from whatever mlx actually is on the
    # machine running this test (e.g. a real Apple Silicon dev box).
    monkeypatch.setattr(backend, "_mlx_installed", lambda: False)

    with caplog.at_level(logging.DEBUG, logger="pydreg.backend"):
        assert backend.detect_backend() == "numpy"

    assert "cupy not installed" in caplog.text


def test_cuda_runtime_available_uses_cupy(monkeypatch):
    class Runtime:
        @staticmethod
        def getDeviceCount():
            return 1

    fake_cupy = types.SimpleNamespace(cuda=types.SimpleNamespace(runtime=Runtime))
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

    assert backend._cuda_runtime_available()


def test_detect_backend_uses_cupy_when_cuda_runtime_available(monkeypatch):
    backend.detect_backend.cache_clear()
    monkeypatch.setattr(backend, "_cupy_installed", lambda: True)
    monkeypatch.setattr(backend, "_cuda_runtime_available", lambda: True)

    assert backend.detect_backend() == "cupy"


def test_detect_backend_reports_no_cuda_without_probe_details(monkeypatch, caplog):
    backend.detect_backend.cache_clear()
    monkeypatch.setattr(backend, "_cupy_installed", lambda: True)
    monkeypatch.setattr(backend, "_cuda_runtime_available", lambda: False)
    # Isolates the cupy-fallback path from whatever mlx actually is on the
    # machine running this test (e.g. a real Apple Silicon dev box).
    monkeypatch.setattr(backend, "_mlx_installed", lambda: False)

    with caplog.at_level(logging.WARNING, logger="pydreg.backend"):
        assert backend.detect_backend() == "numpy"

    assert "cupy installed but no usable CUDA GPU detected at runtime -- falling back to CPU" in caplog.text


def test_detect_backend_reports_missing_mlx(monkeypatch, caplog):
    backend.detect_backend.cache_clear()
    monkeypatch.setattr(backend, "_cupy_installed", lambda: False)
    monkeypatch.setattr(backend, "_mlx_installed", lambda: False)

    with caplog.at_level(logging.DEBUG, logger="pydreg.backend"):
        assert backend.detect_backend() == "numpy"

    assert "mlx not installed" in caplog.text


def test_detect_backend_reports_no_metal_gpu_without_probe_details(monkeypatch, caplog):
    backend.detect_backend.cache_clear()
    monkeypatch.setattr(backend, "_cupy_installed", lambda: False)
    monkeypatch.setattr(backend, "_mlx_installed", lambda: True)
    monkeypatch.setattr(backend, "_mlx_gpu_available", lambda: False)

    with caplog.at_level(logging.WARNING, logger="pydreg.backend"):
        assert backend.detect_backend() == "numpy"

    assert "mlx installed but no usable Metal GPU detected at runtime -- falling back to CPU" in caplog.text


def test_detect_backend_uses_mlx_when_metal_gpu_available(monkeypatch):
    backend.detect_backend.cache_clear()
    monkeypatch.setattr(backend, "_cupy_installed", lambda: False)
    monkeypatch.setattr(backend, "_mlx_installed", lambda: True)
    monkeypatch.setattr(backend, "_mlx_gpu_available", lambda: True)

    assert backend.detect_backend() == "mlx"


def test_detect_backend_prefers_cupy_over_mlx_when_both_available(monkeypatch):
    backend.detect_backend.cache_clear()
    monkeypatch.setattr(backend, "_cupy_installed", lambda: True)
    monkeypatch.setattr(backend, "_cuda_runtime_available", lambda: True)
    monkeypatch.setattr(backend, "_mlx_installed", lambda: True)
    monkeypatch.setattr(backend, "_mlx_gpu_available", lambda: True)

    assert backend.detect_backend() == "cupy"


def _tiny_svr_model():
    """A real DREGModel (not a hand-rolled fake) with a tiny synthetic SVR
    -- built by bypassing __init__'s safetensors loading, since
    _build_cupy_predict_fn/to_sklearn_svr need the real SV/coefs/gamma/rho
    attributes DREGModel.predict itself uses. Reusing the real class (not
    duplicating its predict formula in a fake) means these tests can't
    silently drift out of sync with the reference math."""
    model = object.__new__(DREGModel)
    rng = np.random.default_rng(0)
    model.n_features = 3
    model.n_sv = 5
    model.SV = rng.normal(size=(model.n_sv, model.n_features))
    model.coefs = rng.normal(size=model.n_sv)
    model.x_center = np.zeros(model.n_features)
    model.x_scale = np.ones(model.n_features)
    model.gamma = 0.5
    model.rho = 0.1
    model.y_scale = 2.0
    model.y_center = -0.3
    model._sq_sv = np.sum(model.SV**2, axis=1)
    model._scorer_cache = {}
    return model


class _FakeRawKernel:
    # Stand-in for cupy.RawKernel compiling backend._RBF_GEMV_SRC: applies
    # the same semantics in NumPy (y[row] += sum_j coefs_j *
    # exp(-gamma * (sq_x[row] + sq_sv_j - 2*cross[row, j]))), taking the
    # arguments in the exact order the CUDA signature declares. That makes
    # these CPU tests check _build_cupy_predict_fn's wiring (argument order,
    # chunking, shapes, dtypes of the scalar args); the CUDA C itself is
    # checked against the float64 reference by
    # test_cupy_fused_kernel_matches_reference_on_real_gpu, which needs a GPU.
    def __init__(self, source, name):
        assert name in source

    def __call__(self, grid, block, args):
        cross, sq_x, sq_sv, coefs, gamma, n_cols, y = args
        assert isinstance(gamma, np.float32) and isinstance(n_cols, np.int32)
        assert grid == (cross.shape[0],) and cross.shape[1] == n_cols
        K = np.exp(-gamma * (sq_x[:, None] + sq_sv[None, :] - 2 * cross))
        y += K @ coefs


def _fake_cupy_core(supports_tf32=True, initial_tf32=False):
    """Stands in for cupy._core's float32 compute-type switch: same
    constants and set/get pair, and the same fallback to DEFAULT that real
    CuPy applies on GPUs below compute capability 8.0. initial_tf32 starts
    it at TF32, as CuPy does when CUPY_TF32=1 is set."""
    linalg = types.SimpleNamespace(COMPUTE_TYPE_DEFAULT=1, COMPUTE_TYPE_TF32=7)
    state = {"float32": 7 if initial_tf32 and supports_tf32 else 1}

    def set_compute_type(dtype, compute_type):
        assert np.dtype(dtype) == np.float32
        if compute_type == linalg.COMPUTE_TYPE_TF32 and not supports_tf32:
            compute_type = linalg.COMPUTE_TYPE_DEFAULT
        state["float32"] = compute_type

    return types.SimpleNamespace(
        _routines_linalg=linalg,
        set_compute_type=set_compute_type,
        get_compute_type=lambda dtype: state["float32"],
    )


def _fake_cupy_module():
    # A real GPU isn't available here -- this exercises
    # _build_cupy_predict_fn's own wiring/formula (chunking, kernel math)
    # on NumPy arrays standing in for CuPy ones, since CuPy's array API is
    # a deliberate drop-in match for NumPy's.
    return types.SimpleNamespace(
        asarray=np.asarray,
        ascontiguousarray=np.ascontiguousarray,
        zeros=np.zeros,
        sum=np.sum,
        exp=np.exp,
        asnumpy=np.asarray,
        RawKernel=_FakeRawKernel,
        float32=np.float32,
        float64=np.float64,
        _core=_fake_cupy_core(),
    )


def test_build_cupy_predict_fn_matches_dreg_model_predict(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy_module())
    model = _tiny_svr_model()

    predict = backend._build_cupy_predict_fn(model)
    X_raw = np.random.default_rng(1).normal(size=(4, model.n_features))
    X_scaled = (X_raw - model.x_center) / model.x_scale

    y_scaled = predict(X_scaled)
    reference_y_scaled = (model.predict(X_raw) - model.y_center) / model.y_scale
    # atol loosened from 1e-10: the GEMMs/kernel are now deliberately
    # float32 (see _build_cupy_predict_fn's docstring), so some genuine
    # ~1e-7-relative rounding is expected here, not just formula agreement.
    np.testing.assert_allclose(y_scaled, reference_y_scaled, atol=1e-5)


def test_build_cupy_predict_fn_chunks_over_support_vectors(monkeypatch):
    # sv_chunk smaller than n_sv exercises the multi-iteration accumulation
    # loop, not just the single-chunk fast path.
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy_module())
    model = _tiny_svr_model()

    predict = backend._build_cupy_predict_fn(model, sv_chunk=2)
    X_raw = np.random.default_rng(2).normal(size=(3, model.n_features))
    X_scaled = (X_raw - model.x_center) / model.x_scale

    y_scaled = predict(X_scaled)
    reference_y_scaled = (model.predict(X_raw) - model.y_center) / model.y_scale
    np.testing.assert_allclose(y_scaled, reference_y_scaled, atol=1e-5)


def _real_cuda_available():
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@pytest.mark.skipif(not _real_cuda_available(), reason="needs a real CUDA GPU")
def test_cupy_fused_kernel_matches_reference_on_real_gpu():
    # The real compiled RawKernel, not the NumPy stand-in: n_sv not a
    # multiple of sv_chunk (ragged last chunk) and a query count that isn't
    # a multiple of the block size, against DREGModel.predict's float64
    # formula.
    model = object.__new__(DREGModel)
    rng = np.random.default_rng(4)
    model.n_features = 40
    model.n_sv = 1_003
    model.SV = rng.normal(size=(model.n_sv, model.n_features))
    model.coefs = rng.normal(size=model.n_sv)
    model.x_center = np.zeros(model.n_features)
    model.x_scale = np.ones(model.n_features)
    model.gamma = 0.01
    model.rho = 0.1
    model.y_scale = 1.0
    model.y_center = 0.0
    model._sq_sv = np.sum(model.SV**2, axis=1)
    model._scorer_cache = {}

    predict = backend._build_cupy_predict_fn(model, sv_chunk=256)
    X = rng.normal(size=(301, model.n_features))
    np.testing.assert_allclose(predict(X), model.predict(X), rtol=1e-4, atol=1e-4)
    assert predict(X[:0]).shape == (0,)


@pytest.mark.skipif(not _real_cuda_available(), reason="needs a real CUDA GPU")
def test_cupy_float32_scorer_unaffected_by_a_tf32_scorer_on_real_gpu():
    # The real cuBLAS path: a float32 predict function gives bit-identical
    # output before and after a TF32 one is built and used in the same
    # process. On compute capability 8.0+ the TF32 output must also differ,
    # or this test would pass vacuously.
    model = object.__new__(DREGModel)
    rng = np.random.default_rng(9)
    model.n_features = 360
    model.n_sv = 4_096
    model.SV = rng.normal(size=(model.n_sv, model.n_features))
    model.coefs = rng.normal(size=model.n_sv)
    model.x_center = np.zeros(model.n_features)
    model.x_scale = np.ones(model.n_features)
    model.gamma = 1.0 / model.n_features
    model.rho = 0.1
    model.y_scale = 1.0
    model.y_center = 0.0
    model._sq_sv = np.sum(model.SV**2, axis=1)
    model._scorer_cache = {}
    X = rng.normal(size=(512, model.n_features))

    plain = backend._build_cupy_predict_fn(model)
    before = plain(X)
    asked = backend._build_cupy_predict_fn(model, tf32=True)
    tf32_out = asked(X)
    np.testing.assert_array_equal(plain(X), before)
    if asked.tf32:
        assert np.max(np.abs(tf32_out - before)) > 0


def test_explicit_cupy_build_scorer_builds_a_working_scorer(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy_module())
    model = _tiny_svr_model()

    scorer = backend.build_scorer(model, "cupy")
    X_raw = np.random.default_rng(3).normal(size=(4, model.n_features))

    assert scorer.backend == "cupy"
    np.testing.assert_allclose(scorer.predict(X_raw), model.predict(X_raw), atol=1e-5)


def _cupy_build_with_offset(offset):
    """Wraps the real _build_cupy_predict_fn so its predict_fn returns a
    fixed y_scaled offset -- used to place a smoke-test divergence at a
    precise, known distance from the reference, rather than relying on
    real float32 rounding (which the NumPy-standin fake doesn't actually
    reproduce)."""
    real_build = backend._build_cupy_predict_fn

    def build(dreg_model, **kwargs):
        real_predict = real_build(dreg_model, **kwargs)

        def offset_predict(X_scaled):
            return real_predict(X_scaled) + offset

        offset_predict.tf32 = real_predict.tf32
        return offset_predict

    return build


def test_explicit_cupy_build_scorer_tolerates_a_divergence_between_the_two_tolerances(
    monkeypatch,
):
    # cupy's GEMMs/kernel are deliberately float32 -- CUPY_SMOKE_TEST_ATOL
    # (1e-3) exists specifically so a real divergence in this band (bigger
    # than the default 1e-4, but expected for float32) doesn't raise.
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy_module())
    model = _tiny_svr_model()
    # y_scale=2.0 on this fixture, so a 3e-4 offset in y_scaled space is
    # 6e-4 in the final (unscaled) space the smoke test actually compares.
    monkeypatch.setattr(backend, "_build_cupy_predict_fn", _cupy_build_with_offset(3e-4))

    scorer = backend.build_scorer(model, "cupy")
    X_raw = np.random.default_rng(5).normal(size=(4, model.n_features))
    scorer.predict(X_raw)  # should not raise


def test_explicit_cupy_build_scorer_still_rejects_a_divergence_past_its_looser_tolerance(
    monkeypatch,
):
    # Confirms CUPY_SMOKE_TEST_ATOL loosens the check, it doesn't disable it.
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy_module())
    model = _tiny_svr_model()
    monkeypatch.setattr(backend, "_build_cupy_predict_fn", _cupy_build_with_offset(1.0))

    try:
        backend.build_scorer(model, "cupy").predict(
            np.random.default_rng(6).normal(size=(4, model.n_features))
        )
    except backend.BackendUnavailable as e:
        assert "do not match the NumPy reference" in str(e)
    else:
        raise AssertionError("expected BackendUnavailable")


def test_explicit_cupy_build_scorer_threads_cupy_sv_chunk_through(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy_module())
    model = _tiny_svr_model()
    seen_kwargs = {}
    real_build = backend._build_cupy_predict_fn

    def spying_build(dreg_model, **kwargs):
        seen_kwargs.update(kwargs)
        return real_build(dreg_model, **kwargs)

    monkeypatch.setattr(backend, "_build_cupy_predict_fn", spying_build)

    scorer = backend.build_scorer(model, "cupy", cupy_sv_chunk=2)
    X_raw = np.random.default_rng(4).normal(size=(3, model.n_features))

    assert seen_kwargs == {"sv_chunk": 2, "tf32": False}
    np.testing.assert_allclose(scorer.predict(X_raw), model.predict(X_raw), atol=1e-5)


def _record_gemm_compute_types(fake):
    """Records cupy's float32 compute type each time the fused path's
    `cross = ascontiguousarray(X @ SV.T)` runs, i.e. what the GEMM saw."""
    seen = []
    contiguous = fake.ascontiguousarray

    def recording(a):
        seen.append(fake._core.get_compute_type(np.float32))
        return contiguous(a)

    fake.ascontiguousarray = recording
    return seen


def test_cupy_tf32_defaults_off_and_follows_the_flag(monkeypatch):
    fake = _fake_cupy_module()
    monkeypatch.setitem(sys.modules, "cupy", fake)
    seen = _record_gemm_compute_types(fake)
    model = _tiny_svr_model()
    X_raw = np.random.default_rng(6).normal(size=(4, model.n_features))

    plain = backend.build_scorer(model, "cupy")
    assert plain.tf32 is False
    plain.predict(X_raw)
    assert set(seen) == {1}  # DEFAULT

    seen.clear()
    asked = backend.build_scorer(model, "cupy", tf32=True)
    assert asked.tf32 is True
    asked.predict(X_raw)
    assert set(seen) == {7}  # TF32
    # Neither building nor predicting leaves cupy's own setting changed.
    assert fake._core.get_compute_type(np.float32) == 1


def test_cupy_float32_scorer_stays_float32_after_a_tf32_scorer_is_built(monkeypatch):
    # cupy's compute type is process-wide. Setting it at build time let a
    # float32 scorer silently run at TF32 once a TF32 scorer existed.
    fake = _fake_cupy_module()
    monkeypatch.setitem(sys.modules, "cupy", fake)
    seen = _record_gemm_compute_types(fake)
    model = _tiny_svr_model()
    X_raw = np.random.default_rng(7).normal(size=(4, model.n_features))

    plain = backend.build_scorer(model, "cupy")
    asked = backend.build_scorer(model, "cupy", tf32=True)
    seen.clear()
    plain.predict(X_raw)
    assert set(seen) == {1}
    seen.clear()
    asked.predict(X_raw)
    assert set(seen) == {7}
    seen.clear()
    plain.predict(X_raw)
    assert set(seen) == {1}


def test_cupy_tf32_flag_overrides_cupy_tf32_env_and_restores_it(monkeypatch):
    # CUPY_TF32=1 makes CuPy start at TF32; --tf32 off must still run the
    # GEMM in full float32, and leave CuPy's own setting as it was.
    fake = _fake_cupy_module()
    fake._core = _fake_cupy_core(initial_tf32=True)
    monkeypatch.setitem(sys.modules, "cupy", fake)
    seen = _record_gemm_compute_types(fake)
    model = _tiny_svr_model()

    backend.build_scorer(model, "cupy").predict(
        np.random.default_rng(8).normal(size=(4, model.n_features))
    )
    assert set(seen) == {1}
    assert fake._core.get_compute_type(np.float32) == 7


def test_cupy_tf32_uses_the_looser_smoke_tolerance_only_when_in_effect(monkeypatch):
    # 1.5e-3 in y_scaled space is 3e-3 in score space (y_scale=2): past the
    # float32 tolerance (1e-3), inside TF32_SMOKE_TEST_ATOL (5e-3).
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy_module())
    monkeypatch.setattr(backend, "_build_cupy_predict_fn", _cupy_build_with_offset(1.5e-3))
    X_raw = np.random.default_rng(5).normal(size=(4, 3))

    backend.build_scorer(_tiny_svr_model(), "cupy", tf32=True).predict(X_raw)
    with pytest.raises(backend.BackendUnavailable):
        backend.build_scorer(_tiny_svr_model(), "cupy").predict(X_raw)


def test_cupy_tf32_warns_and_stays_off_below_compute_capability_8(monkeypatch, caplog):
    fake = _fake_cupy_module()
    fake._core = _fake_cupy_core(supports_tf32=False)
    monkeypatch.setitem(sys.modules, "cupy", fake)

    with caplog.at_level(logging.WARNING, logger="pydreg.backend"):
        scorer = backend.build_scorer(_tiny_svr_model(), "cupy", tf32=True)
    assert scorer.tf32 is False
    assert "compute capability 8.0+" in caplog.text


def test_explicit_cupy_build_scorer_raises_when_not_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)
    model = _tiny_svr_model()

    try:
        backend.build_scorer(model, "cupy")
    except backend.BackendUnavailable as e:
        assert "cupy is not installed" in str(e)
    else:
        raise AssertionError("expected BackendUnavailable")


def test_mlx_gpu_available_uses_mlx(monkeypatch):
    fake_core = types.SimpleNamespace(metal=types.SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_core))
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)

    assert backend._mlx_gpu_available()


class _FakeMetalKernel:
    # Stand-in for mx.fast.metal_kernel compiling
    # backend._RBF_GEMV_METAL_SRC: applies the same semantics in NumPy
    # (out[row] = sum_j coefs_j * exp(-gamma * (sq_x[row] + sq_sv_j -
    # 2*cross[row, j]))), taking inputs in the order input_names declares.
    # Like _FakeRawKernel, this checks _build_mlx_predict_fn's wiring
    # (input order, chunking, grid/threadgroup, gamma's dtype); the Metal
    # source itself is checked by
    # test_mlx_fused_kernel_matches_reference_on_real_gpu.
    def __init__(self, name, input_names, output_names, source, **kwargs):
        assert input_names == ["cross", "sq_x", "sq_sv", "coefs", "gamma"]
        assert output_names == ["out"]
        assert all(f"{n}" in source for n in input_names + output_names)

    def __call__(self, inputs, grid, threadgroup, output_shapes, output_dtypes):
        cross, sq_x, sq_sv, coefs, gamma = inputs
        assert gamma.dtype == np.float32 and gamma.shape == (1,)
        n = cross.shape[0]
        assert grid == (n * threadgroup[0], 1, 1) and output_shapes == [(n,)]
        K = np.exp(-gamma[0] * (sq_x[:, None] + sq_sv[None, :] - 2 * cross))
        return [(K @ coefs).astype(output_dtypes[0])]


def _fake_mlx_module():
    # MLX's array API is (like CuPy's) a deliberate drop-in match for
    # NumPy's, so this fake reuses real NumPy functions directly; the one
    # GPU-specific piece, the fused Metal kernel, is _FakeMetalKernel.
    return types.SimpleNamespace(
        array=lambda data, dtype=None: np.asarray(data, dtype=dtype),
        sum=np.sum,
        fast=types.SimpleNamespace(metal_kernel=_FakeMetalKernel),
        float32=np.float32,
        float64=np.float64,
    )


def _install_fake_mlx(monkeypatch):
    # _configure_mlx_tf32 sets MLX_ENABLE_TF32 in os.environ; recording
    # it here makes monkeypatch restore the real environment afterwards.
    monkeypatch.delenv("MLX_ENABLE_TF32", raising=False)
    fake_core = _fake_mlx_module()
    monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_core))
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)
    return fake_core


def test_build_mlx_predict_fn_matches_dreg_model_predict(monkeypatch):
    _install_fake_mlx(monkeypatch)
    model = _tiny_svr_model()

    predict = backend._build_mlx_predict_fn(model)
    X_raw = np.random.default_rng(1).normal(size=(4, model.n_features))
    X_scaled = (X_raw - model.x_center) / model.x_scale

    y_scaled = predict(X_scaled)
    reference_y_scaled = (model.predict(X_raw) - model.y_center) / model.y_scale
    # atol loosened from 1e-10: the GEMMs/kernel are deliberately float32
    # (Metal has no float64 support at all -- see
    # _build_mlx_predict_fn's docstring), so some genuine ~1e-7-relative
    # rounding is expected here, not just formula agreement.
    np.testing.assert_allclose(y_scaled, reference_y_scaled, atol=1e-5)


def test_build_mlx_predict_fn_chunks_over_support_vectors(monkeypatch):
    # sv_chunk smaller than n_sv exercises the multi-iteration accumulation
    # loop, not just the single-chunk fast path.
    _install_fake_mlx(monkeypatch)
    model = _tiny_svr_model()

    predict = backend._build_mlx_predict_fn(model, sv_chunk=2)
    X_raw = np.random.default_rng(2).normal(size=(3, model.n_features))
    X_scaled = (X_raw - model.x_center) / model.x_scale

    y_scaled = predict(X_scaled)
    reference_y_scaled = (model.predict(X_raw) - model.y_center) / model.y_scale
    np.testing.assert_allclose(y_scaled, reference_y_scaled, atol=1e-5)


def _real_metal_available():
    try:
        import mlx.core as mx

        return mx.metal.is_available()
    except Exception:
        return False


@pytest.mark.skipif(not _real_metal_available(), reason="needs a real Metal GPU")
def test_mlx_fused_kernel_matches_reference_on_real_gpu():
    # The real compiled Metal kernel, not the NumPy stand-in: n_sv not a
    # multiple of sv_chunk (ragged last chunk), a row shorter than the
    # threadgroup (last chunk of 235 < 256 threads), and an empty query
    # batch, against DREGModel.predict's float64 formula.
    model = object.__new__(DREGModel)
    rng = np.random.default_rng(4)
    model.n_features = 40
    model.n_sv = 1_003
    model.SV = rng.normal(size=(model.n_sv, model.n_features))
    model.coefs = rng.normal(size=model.n_sv)
    model.x_center = np.zeros(model.n_features)
    model.x_scale = np.ones(model.n_features)
    model.gamma = 0.01
    model.rho = 0.1
    model.y_scale = 1.0
    model.y_center = 0.0
    model._sq_sv = np.sum(model.SV**2, axis=1)
    model._scorer_cache = {}

    predict = backend._build_mlx_predict_fn(model, sv_chunk=256)
    X = rng.normal(size=(301, model.n_features))
    np.testing.assert_allclose(predict(X), model.predict(X), rtol=1e-4, atol=1e-4)
    assert predict(X[:0]).shape == (0,)


def test_explicit_mlx_build_scorer_builds_a_working_scorer(monkeypatch):
    _install_fake_mlx(monkeypatch)
    model = _tiny_svr_model()

    scorer = backend.build_scorer(model, "mlx")
    X_raw = np.random.default_rng(3).normal(size=(4, model.n_features))

    assert scorer.backend == "mlx"
    np.testing.assert_allclose(scorer.predict(X_raw), model.predict(X_raw), atol=1e-5)


def _mlx_build_with_offset(offset):
    """Wraps the real _build_mlx_predict_fn so its predict_fn returns a
    fixed y_scaled offset -- used to place a smoke-test divergence at a
    precise, known distance from the reference, rather than relying on
    real float32 rounding (which the NumPy-standin fake doesn't actually
    reproduce)."""
    real_build = backend._build_mlx_predict_fn

    def build(dreg_model, **kwargs):
        real_predict = real_build(dreg_model, **kwargs)

        def offset_predict(X_scaled):
            return real_predict(X_scaled) + offset

        offset_predict.tf32 = real_predict.tf32
        return offset_predict

    return build


def test_explicit_mlx_build_scorer_tolerates_a_divergence_between_the_two_tolerances(
    monkeypatch,
):
    # mlx's GEMMs/kernel are float32 (Metal has no float64 support at all)
    # -- MLX_SMOKE_TEST_ATOL (1e-3) exists specifically so a real divergence
    # in this band (bigger than the default 1e-4, but expected for float32)
    # doesn't raise.
    _install_fake_mlx(monkeypatch)
    model = _tiny_svr_model()
    # y_scale=2.0 on this fixture, so a 3e-4 offset in y_scaled space is
    # 6e-4 in the final (unscaled) space the smoke test actually compares.
    monkeypatch.setattr(backend, "_build_mlx_predict_fn", _mlx_build_with_offset(3e-4))

    scorer = backend.build_scorer(model, "mlx")
    X_raw = np.random.default_rng(5).normal(size=(4, model.n_features))
    scorer.predict(X_raw)  # should not raise


def test_explicit_mlx_build_scorer_still_rejects_a_divergence_past_its_looser_tolerance(
    monkeypatch,
):
    # Confirms MLX_SMOKE_TEST_ATOL loosens the check, it doesn't disable it.
    _install_fake_mlx(monkeypatch)
    model = _tiny_svr_model()
    monkeypatch.setattr(backend, "_build_mlx_predict_fn", _mlx_build_with_offset(1.0))

    try:
        backend.build_scorer(model, "mlx").predict(
            np.random.default_rng(6).normal(size=(4, model.n_features))
        )
    except backend.BackendUnavailable as e:
        assert "do not match the NumPy reference" in str(e)
    else:
        raise AssertionError("expected BackendUnavailable")


def test_explicit_mlx_build_scorer_threads_mlx_sv_chunk_through(monkeypatch):
    _install_fake_mlx(monkeypatch)
    model = _tiny_svr_model()
    seen_kwargs = {}
    real_build = backend._build_mlx_predict_fn

    def spying_build(dreg_model, **kwargs):
        seen_kwargs.update(kwargs)
        return real_build(dreg_model, **kwargs)

    monkeypatch.setattr(backend, "_build_mlx_predict_fn", spying_build)

    scorer = backend.build_scorer(model, "mlx", mlx_sv_chunk=2)
    X_raw = np.random.default_rng(4).normal(size=(3, model.n_features))

    assert seen_kwargs == {"sv_chunk": 2, "tf32": False}
    np.testing.assert_allclose(scorer.predict(X_raw), model.predict(X_raw), atol=1e-5)


def test_explicit_mlx_build_scorer_raises_when_not_installed(monkeypatch):
    # Simulates mlx being absent regardless of whether it's actually
    # installed in the environment running this test (e.g. a real Apple
    # Silicon dev machine with the `mlx` extra installed) -- assigning None
    # in sys.modules is the standard way to force the next `import` of that
    # name to raise ModuleNotFoundError.
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    model = _tiny_svr_model()

    try:
        backend.build_scorer(model, "mlx")
    except backend.BackendUnavailable as e:
        assert "mlx is not installed" in str(e)
    else:
        raise AssertionError("expected BackendUnavailable")


def test_configure_mlx_tf32_sets_env_from_the_flag_either_way(monkeypatch):
    _install_fake_mlx(monkeypatch)
    assert backend._configure_mlx_tf32(False) is False
    assert os.environ["MLX_ENABLE_TF32"] == "0"

    # The flag wins over a value already in the environment.
    monkeypatch.setenv("MLX_ENABLE_TF32", "1")
    backend._configure_mlx_tf32(False)
    assert os.environ["MLX_ENABLE_TF32"] == "0"

    # Fake matmuls are full precision, i.e. a device without TF32 (an M4):
    # asking for it just reports it isn't in effect.
    assert backend._configure_mlx_tf32(True) is False
    assert os.environ["MLX_ENABLE_TF32"] == "1"


def _round_to_tf32(x):
    """Rounds float32 values to TF32's 10-bit mantissa (truncation)."""
    bits = np.asarray(x, dtype=np.float32).view(np.uint32) & np.uint32(0xFFFFE000)
    return bits.view(np.float32)


def _install_tf32_fake_mlx(monkeypatch):
    # Stands in for MLX doing TF32 matmuls (an M5, or TF32 already locked on
    # earlier in the process): arrays come back TF32-rounded, so the probe's
    # matmul is off.
    fake_core = _install_fake_mlx(monkeypatch)
    fake_core.array = lambda data, dtype=None: _round_to_tf32(np.asarray(data, dtype=np.float32))
    return fake_core


def test_configure_mlx_tf32_raises_when_tf32_is_on_but_not_wanted(monkeypatch):
    _install_tf32_fake_mlx(monkeypatch)

    with pytest.raises(backend.BackendUnavailable, match="MLX_ENABLE_TF32=0"):
        backend._configure_mlx_tf32(False)
    with pytest.raises(backend.BackendUnavailable, match="TF32"):
        backend.build_scorer(_tiny_svr_model(), "mlx")


def test_configure_mlx_tf32_reports_tf32_in_effect_when_wanted(monkeypatch):
    _install_tf32_fake_mlx(monkeypatch)
    assert backend._configure_mlx_tf32(True) is True


def test_build_scorer_caches_tf32_and_float32_scorers_separately(monkeypatch):
    _install_fake_mlx(monkeypatch)
    model = _tiny_svr_model()

    plain = backend.build_scorer(model, "mlx")
    asked = backend.build_scorer(model, "mlx", tf32=True)

    assert plain is not asked
    assert backend.build_scorer(model, "mlx") is plain
    assert plain.tf32 is False
    assert asked.tf32 is False  # fake device has no TF32


def test_tf32_is_ignored_with_a_warning_on_cpu_backends(caplog):
    model = _tiny_svr_model()
    with caplog.at_level(logging.WARNING, logger="pydreg.backend"):
        scorer = backend.build_scorer(model, "numpy", tf32=True)
    assert scorer.tf32 is False
    assert "--tf32 only affects the cupy and mlx backends" in caplog.text


@pytest.mark.skipif(
    not (backend._mlx_installed() and backend._mlx_gpu_available()),
    reason="needs mlx with a real Metal GPU",
)
@pytest.mark.parametrize("enabled", [False, True])
def test_configure_mlx_tf32_matches_real_gemm_precision(enabled):
    # Subprocess: MLX caches MLX_ENABLE_TF32 at the process's first matmul,
    # so each setting needs a process where mlx hasn't run one yet.
    code = (
        "import numpy as np, mlx.core as mx\n"
        "from pydreg import backend\n"
        f"in_effect = backend._configure_mlx_tf32({enabled})\n"
        "rng = np.random.default_rng(0)\n"
        "a = rng.standard_normal((1024, 360)).astype(np.float32)\n"
        "b = rng.standard_normal((2048, 360)).astype(np.float32)\n"
        "ref = a.astype(np.float64) @ b.T.astype(np.float64)\n"
        "got = np.array(mx.array(a) @ mx.array(b).T)\n"
        "print(in_effect, np.abs(got - ref).max())\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "MLX_ENABLE_TF32"}
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    in_effect, err = out.stdout.split()
    # Full float32 is ~1e-4 here; TF32 is ~1e-1 (measured on an M5). An M4
    # has no TF32, so enabled=True legitimately reports False there.
    if in_effect == "True":
        assert enabled and float(err) > 1e-2
    else:
        assert float(err) < 1e-3


class TinyDREGModel:
    def __init__(self):
        self.x_center = np.array([1.0, -1.0])
        self.x_scale = np.array([2.0, 4.0])
        self.y_scale = 3.0
        self.y_center = -0.25

    def predict(self, X):
        X_scaled = (X - self.x_center) / self.x_scale
        return (X_scaled[:, 0] - 2 * X_scaled[:, 1]) * self.y_scale + self.y_center


def test_sklearn_like_wrapper_validates_matching_predictions():
    model = TinyDREGModel()

    def sk_predict(X_scaled):
        return X_scaled[:, 0] - 2 * X_scaled[:, 1]

    predict = backend._wrap_sklearn_like(model, sk_predict, "fake")
    X = np.array([[1.0, -1.0], [3.0, 7.0], [5.0, -5.0]])

    np.testing.assert_allclose(predict(X), model.predict(X))
    # Second call uses the already-validated fast path.
    np.testing.assert_allclose(predict(X), model.predict(X))


def test_sklearn_like_wrapper_rejects_shifted_predictions():
    model = TinyDREGModel()

    def shifted_predict(X_scaled):
        return X_scaled[:, 0] - 2 * X_scaled[:, 1] + 1.0

    predict = backend._wrap_sklearn_like(model, shifted_predict, "fake")
    X = np.array([[1.0, -1.0], [3.0, 7.0]])

    try:
        predict(X)
    except backend.BackendUnavailable as e:
        assert "do not match the NumPy reference" in str(e)
    else:
        raise AssertionError("expected BackendUnavailable")


def test_sklearn_like_wrapper_cross_check_flags_numpy_side_when_sklearn_also_diverges(monkeypatch):
    # If a non-sklearn backend's smoke test fails, and scikit-learn's own
    # libsvm predict on the same sample *also* diverges from the NumPy
    # reference, that implicates DREGModel.predict itself rather than the
    # failing backend's own GPU/library conversion.
    model = TinyDREGModel()

    class FakeSkSVR:
        def predict(self, X_scaled):
            return X_scaled[:, 0] - 2 * X_scaled[:, 1] + 1.0

    monkeypatch.setattr(backend, "to_sklearn_svr", lambda dreg_model: FakeSkSVR())

    def shifted_predict(X_scaled):
        return X_scaled[:, 0] - 2 * X_scaled[:, 1] + 1.0

    predict = backend._wrap_sklearn_like(model, shifted_predict, "fake")
    X = np.array([[1.0, -1.0], [3.0, 7.0]])

    try:
        predict(X)
    except backend.BackendUnavailable as e:
        assert "also diverges from the NumPy reference" in str(e)
        assert "NumPy-reference-side issue" in str(e)
    else:
        raise AssertionError("expected BackendUnavailable")


def test_sklearn_like_wrapper_cross_check_points_at_backend_when_sklearn_agrees(monkeypatch):
    # If scikit-learn's own libsvm predict on the same sample agrees with
    # the NumPy reference, the divergence looks specific to the failing
    # backend, not to DREGModel.predict.
    model = TinyDREGModel()

    class FakeSkSVR:
        def predict(self, X_scaled):
            return X_scaled[:, 0] - 2 * X_scaled[:, 1]

    monkeypatch.setattr(backend, "to_sklearn_svr", lambda dreg_model: FakeSkSVR())

    def shifted_predict(X_scaled):
        return X_scaled[:, 0] - 2 * X_scaled[:, 1] + 1.0

    predict = backend._wrap_sklearn_like(model, shifted_predict, "fake")
    X = np.array([[1.0, -1.0], [3.0, 7.0]])

    try:
        predict(X)
    except backend.BackendUnavailable as e:
        assert "agrees with the NumPy reference" in str(e)
        assert "specific to this backend" in str(e)
    else:
        raise AssertionError("expected BackendUnavailable")


def test_sklearn_like_wrapper_skips_cross_check_for_sklearn_backend_itself(monkeypatch):
    # backend_name="sklearn" IS the cross-check -- don't recurse into it.
    model = TinyDREGModel()
    calls = []
    monkeypatch.setattr(
        backend, "to_sklearn_svr", lambda dreg_model: calls.append(1) or None
    )

    def shifted_predict(X_scaled):
        return X_scaled[:, 0] - 2 * X_scaled[:, 1] + 1.0

    predict = backend._wrap_sklearn_like(model, shifted_predict, "sklearn")
    X = np.array([[1.0, -1.0], [3.0, 7.0]])

    try:
        predict(X)
    except backend.BackendUnavailable as e:
        assert "diverges from the NumPy reference" not in str(e)
        assert "agrees with the NumPy reference" not in str(e)
    else:
        raise AssertionError("expected BackendUnavailable")
    assert calls == []

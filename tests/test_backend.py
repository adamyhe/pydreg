import logging
import re
import sys
import types

import numpy as np

from pydreg import backend
from pydreg.models import DREGModel


def test_detect_backend_reports_missing_cuml(monkeypatch, caplog):
    backend.detect_backend.cache_clear()
    monkeypatch.setattr(backend, "_cuml_installed", lambda: False)

    with caplog.at_level(logging.INFO, logger="pydreg.backend"):
        assert backend.detect_backend() == "numpy"

    assert "cuml not installed" in caplog.text


def test_cuda_runtime_available_uses_cupy(monkeypatch):
    class Runtime:
        @staticmethod
        def getDeviceCount():
            return 1

    fake_cupy = types.SimpleNamespace(cuda=types.SimpleNamespace(runtime=Runtime))
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

    assert backend._cuda_runtime_available()


def test_detect_backend_uses_cuml_when_cuda_runtime_available(monkeypatch):
    backend.detect_backend.cache_clear()
    monkeypatch.setattr(backend, "_cuml_installed", lambda: True)
    monkeypatch.setattr(backend, "_cuda_runtime_available", lambda: True)
    monkeypatch.setattr(backend, "_cuda_compute_capability", lambda: 80)

    assert backend.detect_backend() == "cuml"


def test_cuda_compute_capability_reads_cupy_device(monkeypatch):
    class FakeDevice:
        compute_capability = "61"

    fake_cupy = types.SimpleNamespace(cuda=types.SimpleNamespace(Device=FakeDevice))
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

    assert backend._cuda_compute_capability() == 61


def test_detect_backend_falls_back_to_numpy_on_pascal_compute_capability(monkeypatch, caplog):
    # Real hardware finding: RAPIDS/cuML dropped Pascal (compute capability
    # 6.x) support in 24.02 -- confirmed on a real TITAN X that running
    # cuml there doesn't error, it silently returns wrong predictions. This
    # must be caught before build_scorer(), not left to the smoke test.
    backend.detect_backend.cache_clear()
    monkeypatch.setattr(backend, "_cuml_installed", lambda: True)
    monkeypatch.setattr(backend, "_cuda_runtime_available", lambda: True)
    monkeypatch.setattr(backend, "_cuda_compute_capability", lambda: 61)

    with caplog.at_level(logging.INFO, logger="pydreg.backend"):
        assert backend.detect_backend() == "numpy"

    assert "compute capability 6.1 is below" in caplog.text


def test_detect_backend_reports_no_cuda_without_probe_details(monkeypatch, caplog):
    backend.detect_backend.cache_clear()
    monkeypatch.setattr(backend, "_cuml_installed", lambda: True)
    monkeypatch.setattr(backend, "_cuda_runtime_available", lambda: False)

    with caplog.at_level(logging.INFO, logger="pydreg.backend"):
        assert backend.detect_backend() == "numpy"

    assert "cuml installed but no usable CUDA GPU detected at runtime -- falling back to CPU" in caplog.text


def test_explicit_cuml_build_scorer_raises_when_construction_fails(monkeypatch):
    class BrokenSVR:
        @classmethod
        def from_sklearn(cls, sk_model):
            raise ZeroDivisionError("float division by zero")

    fake_cuml = types.ModuleType("cuml")
    fake_svm = types.ModuleType("cuml.svm")
    fake_svm.SVR = BrokenSVR
    fake_cuml.svm = fake_svm
    monkeypatch.setitem(sys.modules, "cuml", fake_cuml)
    monkeypatch.setitem(sys.modules, "cuml.svm", fake_svm)
    monkeypatch.setattr(backend, "to_sklearn_svr", lambda dreg_model: object())
    monkeypatch.setattr(backend, "_cuda_compute_capability", lambda: 80)

    class FakeModel:
        _scorer_cache = {}

    try:
        backend.build_scorer(FakeModel(), "cuml")
    except backend.BackendUnavailable as e:
        assert "could not build a GPU model" in str(e)
    else:
        raise AssertionError("expected BackendUnavailable")


def test_explicit_cuml_build_scorer_raises_on_pascal_compute_capability(monkeypatch):
    fake_cuml = types.ModuleType("cuml")
    fake_svm = types.ModuleType("cuml.svm")
    fake_cuml.svm = fake_svm
    monkeypatch.setitem(sys.modules, "cuml", fake_cuml)
    monkeypatch.setitem(sys.modules, "cuml.svm", fake_svm)
    monkeypatch.setattr(backend, "_cuda_compute_capability", lambda: 61)

    class FakeModel:
        _scorer_cache = {}

    try:
        backend.build_scorer(FakeModel(), "cuml")
    except backend.BackendUnavailable as e:
        assert "compute capability 6.1 is below" in str(e)
        assert "silently returns wrong predictions" in str(e)
    else:
        raise AssertionError("expected BackendUnavailable")


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


class _FakeElementwiseKernel:
    # Real cupy.ElementwiseKernel compiles `operation` (CUDA C) into a
    # single broadcasting kernel; this stand-in instead `eval`s it as a
    # Python expression against NumPy arrays standing in for CuPy ones.
    # The kernel body is CUDA C, not Python -- expf(...) and float literals
    # with an "f" suffix (e.g. "0.05f") are valid CUDA C but not valid
    # Python syntax, so both get normalized away before eval. This
    # exercises the actual formula/broadcasting/dtype these tests care
    # about, not cupy's own kernel compilation (which needs a real GPU to
    # mean anything).
    def __init__(self, in_params, out_params, operation, name):
        self._in_names = [p.strip().split()[-1] for p in in_params.split(",")]
        rhs = operation.split("=", 1)[1].strip()
        rhs = rhs.replace("expf(", "exp(")
        rhs = re.sub(r"(?<=[0-9.])f\b", "", rhs)
        self._rhs = rhs

    def __call__(self, *args):
        ns = {"exp": np.exp, **dict(zip(self._in_names, args))}
        return eval(self._rhs, ns)  # noqa: S307 -- fixed test-only expression


def _fake_cupy_module():
    # A real GPU isn't available here -- this exercises
    # _build_cupy_predict_fn's own wiring/formula (chunking, kernel math)
    # on NumPy arrays standing in for CuPy ones, since CuPy's array API is
    # a deliberate drop-in match for NumPy's.
    return types.SimpleNamespace(
        asarray=np.asarray,
        zeros=np.zeros,
        sum=np.sum,
        exp=np.exp,
        asnumpy=np.asarray,
        ElementwiseKernel=_FakeElementwiseKernel,
        float32=np.float32,
        float64=np.float64,
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

        return offset_predict

    return build


def test_explicit_cupy_build_scorer_tolerates_a_divergence_between_the_two_tolerances(
    monkeypatch,
):
    # cupy's GEMMs/kernel are deliberately float32 -- CUPY_SMOKE_TEST_ATOL
    # (5e-4) exists specifically so a real divergence in this band (bigger
    # than the default 1e-4, but expected for float32) doesn't raise.
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy_module())
    model = _tiny_svr_model()
    # y_scale=2.0 on this fixture, so a 1.5e-4 offset in y_scaled space is
    # 3e-4 in the final (unscaled) space the smoke test actually compares.
    monkeypatch.setattr(backend, "_build_cupy_predict_fn", _cupy_build_with_offset(1.5e-4))

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

    assert seen_kwargs == {"sv_chunk": 2}
    np.testing.assert_allclose(scorer.predict(X_raw), model.predict(X_raw), atol=1e-5)


def test_explicit_cupy_build_scorer_raises_when_not_installed():
    model = _tiny_svr_model()

    try:
        backend.build_scorer(model, "cupy")
    except backend.BackendUnavailable as e:
        assert "cupy is not installed" in str(e)
    else:
        raise AssertionError("expected BackendUnavailable")


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

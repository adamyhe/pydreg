import logging
import sys
import types

import numpy as np

from pydreg import backend


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

    assert backend.detect_backend() == "cuml"


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

    class FakeModel:
        _scorer_cache = {}

    try:
        backend.build_scorer(FakeModel(), "cuml")
    except backend.BackendUnavailable as e:
        assert "could not build a GPU model" in str(e)
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

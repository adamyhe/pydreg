import logging
import sys
import types

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

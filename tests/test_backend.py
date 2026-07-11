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


def test_detect_backend_uses_cuml_when_probe_succeeds(monkeypatch):
    backend.detect_backend.cache_clear()
    monkeypatch.setattr(backend, "_cuml_installed", lambda: True)

    class WorkingSVR:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return [0.0]

    fake_cuml = types.ModuleType("cuml")
    fake_svm = types.ModuleType("cuml.svm")
    fake_svm.SVR = WorkingSVR
    fake_cuml.svm = fake_svm
    monkeypatch.setitem(sys.modules, "cuml", fake_cuml)
    monkeypatch.setitem(sys.modules, "cuml.svm", fake_svm)

    assert backend.detect_backend() == "cuml"


def test_detect_backend_reports_failed_cuml_probe_without_details(monkeypatch, caplog):
    backend.detect_backend.cache_clear()
    monkeypatch.setattr(backend, "_cuml_installed", lambda: True)

    class BrokenSVR:
        def fit(self, X, y):
            raise ZeroDivisionError("float division by zero")

    fake_cuml = types.ModuleType("cuml")
    fake_svm = types.ModuleType("cuml.svm")
    fake_svm.SVR = BrokenSVR
    fake_cuml.svm = fake_svm
    monkeypatch.setitem(sys.modules, "cuml", fake_cuml)
    monkeypatch.setitem(sys.modules, "cuml.svm", fake_svm)

    with caplog.at_level(logging.INFO, logger="pydreg.backend"):
        assert backend.detect_backend() == "numpy"

    assert "cuml installed but no usable CUDA GPU detected at runtime -- falling back to CPU" in caplog.text
    assert "float division by zero" not in caplog.text

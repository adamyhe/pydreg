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


def test_detect_backend_reports_no_cuda_without_probe_details(monkeypatch, caplog):
    backend.detect_backend.cache_clear()
    monkeypatch.setattr(backend, "_cuml_installed", lambda: True)
    monkeypatch.setattr(backend, "_cuda_runtime_available", lambda: False)

    with caplog.at_level(logging.INFO, logger="pydreg.backend"):
        assert backend.detect_backend() == "numpy"

    assert "cuml installed but no usable CUDA GPU detected at runtime -- falling back to CPU" in caplog.text
    assert "float division by zero" not in caplog.text


def test_detect_backend_hides_cuml_probe_exception_at_info(monkeypatch, caplog):
    backend.detect_backend.cache_clear()
    monkeypatch.setattr(backend, "_cuml_installed", lambda: True)
    monkeypatch.setattr(backend, "_cuda_runtime_available", lambda: True)

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

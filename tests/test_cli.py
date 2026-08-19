from pydreg import cli


def _run_and_capture(monkeypatch, argv):
    calls = []
    monkeypatch.setattr(
        cli.pipeline, "run", lambda *args, **kwargs: calls.append(kwargs)
    )
    cli.main(argv)
    return calls[0]


def test_pmv_laplace_tail_tol_defaults_to_exact(monkeypatch):
    kwargs = _run_and_capture(monkeypatch, ["plus.bw", "minus.bw", "out"])
    assert kwargs["pmv_laplace_tail_tol"] == 0.0


def test_pmv_laplace_fast_sets_recommended_tail_tol(monkeypatch):
    kwargs = _run_and_capture(
        monkeypatch, ["plus.bw", "minus.bw", "out", "--pmv-laplace-fast"]
    )
    assert kwargs["pmv_laplace_tail_tol"] == 1e-6


def test_explicit_pmv_laplace_tail_tol_overrides_fast_default(monkeypatch):
    kwargs = _run_and_capture(
        monkeypatch,
        [
            "plus.bw",
            "minus.bw",
            "out",
            "--pmv-laplace-fast",
            "--pmv-laplace-tail-tol",
            "1e-3",
        ],
    )
    assert kwargs["pmv_laplace_tail_tol"] == 1e-3


def test_explicit_pmv_laplace_tail_tol_without_fast_flag(monkeypatch):
    kwargs = _run_and_capture(
        monkeypatch,
        ["plus.bw", "minus.bw", "out", "--pmv-laplace-tail-tol", "2e-5"],
    )
    assert kwargs["pmv_laplace_tail_tol"] == 2e-5

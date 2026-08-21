from pydreg import cli, stats


def _run_and_capture(monkeypatch, argv):
    calls = []
    monkeypatch.setattr(
        cli.pipeline, "run", lambda *args, **kwargs: calls.append(kwargs)
    )
    cli.main(argv)
    return calls[0]


def test_pmv_laplace_tail_tol_defaults_to_fast(monkeypatch):
    kwargs = _run_and_capture(monkeypatch, ["plus.bw", "minus.bw", "out"])
    assert kwargs["pmv_laplace_tail_tol"] == stats.PMV_LAPLACE_FAST_TAIL_TOL


def test_pmv_laplace_exact_sets_zero_tail_tol(monkeypatch):
    kwargs = _run_and_capture(
        monkeypatch, ["plus.bw", "minus.bw", "out", "--pmv-laplace-exact"]
    )
    assert kwargs["pmv_laplace_tail_tol"] == 0.0


def test_explicit_pmv_laplace_tail_tol_overrides_exact_flag(monkeypatch):
    kwargs = _run_and_capture(
        monkeypatch,
        [
            "plus.bw",
            "minus.bw",
            "out",
            "--pmv-laplace-exact",
            "--pmv-laplace-tail-tol",
            "1e-3",
        ],
    )
    assert kwargs["pmv_laplace_tail_tol"] == 1e-3


def test_explicit_pmv_laplace_tail_tol_without_exact_flag(monkeypatch):
    kwargs = _run_and_capture(
        monkeypatch,
        ["plus.bw", "minus.bw", "out", "--pmv-laplace-tail-tol", "2e-5"],
    )
    assert kwargs["pmv_laplace_tail_tol"] == 2e-5

"""Benchmark pydreg's scoring backends against each other.

Motivation: a manual numpy-vs-sklearn comparison (see docs/PERF_LOG.md,
2026-07-09 entry) found the "sklearn" backend ~15x slower than "numpy" on a
CPU-only Mac (libsvm's predict loop is single-threaded C; DREGModel.predict's
chunked matmul dispatches to a multithreaded BLAS), despite computing
identical math (agrees to ~1e-10) -- which is why backend.detect_backend()
no longer auto-selects "sklearn" on CPU. This script exists so that result
can be reproduced/re-checked on other hardware (e.g. a Linux box with
OpenBLAS/MKL, or with cuml installed) rather than trusting a single machine's
numbers.

Usage:
    uv run python scripts/bench_backends.py
    uv run python scripts/bench_backends.py --model-path _models/dreg_svr/svm.model.safetensors.zst
    uv run python scripts/bench_backends.py --batch-sizes 256 4096 50000 --reps 3

Each backend that's actually usable on this machine (see
pydreg.backend.build_scorer) is benchmarked; unavailable ones (e.g. "cuml"
without a GPU) are skipped with a note, not treated as failures.
"""

import argparse
import time

import numpy as np

from pydreg import backend
from pydreg.models import DREGModel


def make_batch(model, n, rng):
    """Roughly realistic feature scale: centered on x_center with an
    x_scale-sized spread, clipped at 0 since these are read-count-derived
    bin sums (never negative)."""
    X = model.x_center + model.x_scale * rng.standard_normal((n, model.n_features))
    return np.clip(X, 0, None)


def bench(predict_fn, X, reps):
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        predict_fn(X)
        times.append(time.perf_counter() - t0)
    return min(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default=None,
        help="local .safetensors[.zst] path or model directory; defaults to "
        "DREGModel.from_pretrained() (downloads/uses the HF cache)",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["numpy", "sklearn", "cuml"],
        help="backend tiers to attempt (unavailable ones are skipped)",
    )
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=[256, 1024, 4096],
        help="number of query positions per predict() call to time",
    )
    parser.add_argument("--reps", type=int, default=1, help="timed repeats per (backend, batch size); the minimum is reported")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("loading model..." + (f" ({args.model_path})" if args.model_path else " (from_pretrained)"))
    model = DREGModel(args.model_path) if args.model_path else DREGModel.from_pretrained()
    print(f"n_sv={model.n_sv}, n_features={model.n_features}")

    scorers = {}
    for name in args.backends:
        try:
            scorers[name] = backend.build_scorer(model, name)
        except backend.BackendUnavailable as e:
            print(f"skipping {name!r}: {e}")

    rng = np.random.default_rng(args.seed)
    for n in args.batch_sizes:
        X = make_batch(model, n, rng)
        results = {}
        for name, scorer in scorers.items():
            t = bench(scorer.predict, X, args.reps)
            results[name] = t
            print(f"n={n:>7}  {name:>8}={t:8.3f}s  ({n / t:9.0f} pos/s)", flush=True)
        if "numpy" in results:
            for name, t in results.items():
                if name != "numpy":
                    print(f"    ratio({name}/numpy) = {t / results['numpy']:.2f}x")

    # Agreement check -- all backends should compute the same math.
    X = make_batch(model, min(args.batch_sizes), rng)
    preds = {name: scorer.predict(X) for name, scorer in scorers.items()}
    if "numpy" in preds:
        for name, y in preds.items():
            if name != "numpy":
                print(f"max abs diff numpy vs {name}: {np.max(np.abs(y - preds['numpy']))}")


if __name__ == "__main__":
    main()

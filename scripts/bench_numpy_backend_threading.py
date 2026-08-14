"""Diagnoses whether DREGModel.predict()'s GEMM (`X_scaled @ sv_block.T`)
and its numba-fused RBF-kernel-and-reduce step (`_rbf_accumulate`) compete
for cores when both run multi-threaded in the same process, and whether
BLAS parallelizes this GEMM's shape well at all on this machine.

Motivation: on a 10-core Apple Silicon dev machine (Accelerate/vecLib
BLAS), the fused elementwise kernel scaled well in isolation (~5.7x at 10
numba threads) but the GEMM alone only reached ~1.6x regardless of
VECLIB_MAXIMUM_THREADS, capping full predict() at ~3x overall despite the
kernel fix -- see docs/PERF_LOG.md's model-loading/scoring entries. Whether
that GEMM ceiling and any numba/BLAS thread contention is Accelerate-
specific, or shows up on OpenBLAS/MKL (the common case on x86/Linux) too,
can only be answered by actually running this on that hardware -- this
script exists so that can be done directly rather than guessed from one
machine's numbers.

Usage:
    uv run python scripts/bench_numpy_backend_threading.py
    uv run python scripts/bench_numpy_backend_threading.py --model-path _models/dreg_svr/svm.model.safetensors.zst
    uv run python scripts/bench_numpy_backend_threading.py --batch-size 4096 --sv-chunk 20000 --reps 3
"""

import argparse
import time

import numba
import numpy as np
import threadpoolctl

from pydreg.models import DREGModel, _rbf_accumulate


def timed(fn, reps):
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--sv-chunk", type=int, default=20_000)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("threadpoolctl sees:")
    infos = threadpoolctl.threadpool_info()
    if not infos:
        print("  (nothing -- BLAS isn't a threadpoolctl-controllable library on this "
              "machine, e.g. Apple Accelerate)")
    for info in infos:
        print(f"  {info}")
    print()

    print("loading model..." + (f" ({args.model_path})" if args.model_path else " (from_pretrained)"))
    model = DREGModel(args.model_path) if args.model_path else DREGModel.from_pretrained()
    print(f"n_sv={model.n_sv}, n_features={model.n_features}, sv_chunk={args.sv_chunk}\n")

    rng = np.random.default_rng(args.seed)
    X_raw = model.x_center + model.x_scale * rng.standard_normal((args.batch_size, model.n_features))
    X_raw = np.clip(X_raw, 0, None)
    X_scaled = (X_raw - model.x_center) / model.x_scale
    sq_x = np.sum(X_scaled**2, axis=1)
    sv_block = model.SV[: args.sv_chunk]
    sq_sv_block = model._sq_sv[: args.sv_chunk]
    coefs_block = model.coefs[: args.sv_chunk]

    def gemm_only():
        return X_scaled @ sv_block.T

    def kernel_only(cross):
        y = np.zeros(X_scaled.shape[0])
        _rbf_accumulate(cross, sq_x, sq_sv_block, model.gamma, coefs_block, y)
        return y

    def full_predict():
        return model.predict(X_raw, chunk=args.sv_chunk)

    cross = gemm_only()
    kernel_only(cross)  # warm up numba JIT before any timing

    n_cores = numba.config.NUMBA_DEFAULT_NUM_THREADS
    print(f"NUMBA_DEFAULT_NUM_THREADS (~logical cores numba sees) = {n_cores}\n")

    print("--- GEMM alone, BLAS threads swept via threadpoolctl ---")
    for blas_threads in sorted({1, 2, max(1, n_cores // 2), n_cores}):
        with threadpoolctl.threadpool_limits(limits=blas_threads):
            t = timed(gemm_only, args.reps)
        print(f"  blas_threads={blas_threads:2d}  {t*1000:8.1f}ms")
    print()

    print("--- fused kernel alone, numba threads swept (BLAS uncontrolled) ---")
    for numba_threads in sorted({1, 2, max(1, n_cores // 2), n_cores}):
        numba.set_num_threads(numba_threads)
        kernel_only(cross)  # rewarm for this thread count
        t = timed(lambda: kernel_only(cross), args.reps)
        print(f"  numba_threads={numba_threads:2d}  {t*1000:8.1f}ms")
    print()

    print("--- full predict(): every (BLAS threads, numba threads) combination ---")
    print("    (the key question: does BLAS_N + numba_N beat BLAS_1 + numba_N? if")
    print("    not, they're contending for the same cores, not adding capacity.)")
    thread_options = sorted({1, max(1, n_cores // 2), n_cores})
    full_predict()  # warm up
    for blas_threads in thread_options:
        for numba_threads in thread_options:
            numba.set_num_threads(numba_threads)
            with threadpoolctl.threadpool_limits(limits=blas_threads):
                full_predict()  # rewarm
                t = timed(full_predict, args.reps)
            print(
                f"  blas={blas_threads:2d} numba={numba_threads:2d}  "
                f"{t:7.3f}s  ({args.batch_size / t:8.0f} pos/s)"
            )


if __name__ == "__main__":
    main()

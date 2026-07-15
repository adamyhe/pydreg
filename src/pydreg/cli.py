"""Thin CLI entry point mirroring run_dREG.bsh's argument shape (plus/minus
bigWig, output prefix), plus flags for the things this port added: backend
selection and pv/smoothing overrides."""

import argparse
import logging

from . import pipeline


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pydreg", description="dREG peak calling")
    parser.add_argument("plus_bw", help="PRO-seq/GRO-seq plus-strand bigWig")
    parser.add_argument("minus_bw", help="PRO-seq/GRO-seq minus-strand bigWig")
    parser.add_argument("out_prefix", help="output file prefix")
    parser.add_argument(
        "--backend",
        choices=["auto", "cuml", "cupy", "sklearn", "numpy"],
        default="auto",
        help="scoring backend; 'auto' uses cuml when CuPy sees a CUDA device "
        "with compute capability >=7.0, otherwise numpy. 'cupy' is an "
        "experimental GPU tier that works on older GPUs too (not "
        "auto-selected -- see docs/OPTIMIZATION.md). An explicit choice "
        "raises if that backend isn't usable, rather than silently "
        "falling back.",
    )
    parser.add_argument("--smoothwidth", type=int, default=4)
    parser.add_argument("--pv-adjust", default="fdr")
    parser.add_argument("--pv-threshold", type=float, default=0.05)
    parser.add_argument(
        "--query-chunk",
        type=int,
        default=None,
        help="positions scored per batch; defaults to a backend-specific size "
        "(see pydreg.backend.DEFAULT_QUERY_CHUNK)",
    )
    parser.add_argument(
        "-c",
        "--cuml-query-chunk",
        type=int,
        default=2**20,
        help="positions scored per batch for the cuml backend specifically when "
        "--query-chunk is not set; ignored by every other backend (including "
        "cupy, which uses its own DEFAULT_QUERY_CHUNK entry)",
    )
    parser.add_argument(
        "--cupy-sv-chunk",
        type=int,
        default=None,
        help="support vectors (of 605,187) evaluated per GPU kernel/GEMM call "
        "for the cupy backend specifically; defaults to "
        "_build_cupy_predict_fn's own default. The main lever for trading GPU "
        "memory for fewer, larger (better-amortized) kernel launches -- real "
        "headroom varies by card, so this is left tunable rather than hardcoded",
    )
    parser.add_argument(
        "--peak-calling-cores",
        type=int,
        default=1,
        help="worker processes for the final CPU peak-calling stage; legacy "
        "dREG parallelized this stage in 500-peak blocks",
    )
    parser.add_argument(
        "--peak-calling-block-width",
        type=int,
        default=100,
        help="candidate broad peaks per peak-calling worker task; smaller "
        "blocks improve load balancing on uneven broad peaks",
    )
    parser.add_argument(
        "--pmv-laplace-cdf-maxpts",
        type=int,
        default=25000,
        help="maximum integration points for each SciPy multivariate-normal "
        "CDF inside pmv_laplace; 25000 matches R's mvtnorm::pmvnorm()/"
        "GenzBretz() default. Lower values trade fidelity for further speed; "
        "higher values exceed what R's own reference implementation ever computed",
    )
    parser.add_argument(
        "--pmv-laplace-cdf-eps",
        type=float,
        default=1e-3,
        help="absolute/relative tolerance for each SciPy multivariate-normal "
        "CDF inside pmv_laplace; 1e-3 matches R's mvtnorm::pmvnorm()/"
        "GenzBretz() default. Lower values increase precision beyond R's own "
        "reference implementation, at a large speed cost",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable tqdm progress bars (shown by default on a terminal; "
        "auto-hidden anyway when stdout is redirected, e.g. to a log file)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    backend_name = None if args.backend == "auto" else args.backend
    pipeline.run(
        args.plus_bw,
        args.minus_bw,
        args.out_prefix,
        backend_name=backend_name,
        smoothwidth=args.smoothwidth,
        pv_adjust=args.pv_adjust,
        pv_threshold=args.pv_threshold,
        query_chunk=args.query_chunk,
        cuml_query_chunk=args.cuml_query_chunk,
        cupy_sv_chunk=args.cupy_sv_chunk,
        peak_calling_cores=args.peak_calling_cores,
        peak_calling_block_width=args.peak_calling_block_width,
        pmv_laplace_cdf_maxpts=args.pmv_laplace_cdf_maxpts,
        pmv_laplace_cdf_eps=args.pmv_laplace_cdf_eps,
        progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()

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
        choices=["auto", "cuml", "sklearn", "numpy"],
        default="auto",
        help="scoring backend; 'auto' tries cuml, then sklearn, then numpy. An "
        "explicit choice raises if that backend isn't usable, rather than "
        "silently falling back.",
    )
    parser.add_argument("--smoothwidth", type=int, default=4)
    parser.add_argument("--pv-adjust", default="fdr")
    parser.add_argument("--pv-threshold", type=float, default=0.05)
    parser.add_argument(
        "--query-chunk", type=int, default=None,
        help="positions scored per batch; defaults to a backend-specific size "
        "(see pydreg.backend.DEFAULT_QUERY_CHUNK)",
    )
    parser.add_argument(
        "--cuml-query-chunk", type=int, default=800_000,
        help="positions scored per batch for the cuml backend when --query-chunk "
        "is not set; ignored by CPU backends",
    )
    parser.add_argument(
        "--peak-calling-cores", type=int, default=1,
        help="worker processes for the final CPU peak-calling stage; legacy "
        "dREG parallelized this stage in 500-peak blocks",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--no-progress", action="store_true",
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
        peak_calling_cores=args.peak_calling_cores,
        progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()

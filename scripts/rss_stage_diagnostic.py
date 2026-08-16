"""One-shot diagnostic: runs pipeline.run() with RSS sampled at every stage
boundary (same boundaries as the normal -v log's _timed() calls), so a
single real run shows exactly which stage's memory footprint is actually
growing -- instead of inferring it from before/after peak-RSS deltas.

Usage (same positional args as the `pydreg` CLI):
    uv run python scripts/rss_stage_diagnostic.py \\
        data/K562_groseq.pl.bw data/K562_groseq.mn.bw pred/K562_groseq_diag \\
        --cores 16

Prints one line per stage boundary: `[RSS so far] START/END stage (elapsed)`.
Leaves the normal -v log output intact (this only adds RSS numbers to the
existing stage boundaries, it doesn't replace them) -- run with -v too if
you want both.

Also runs against pre-0.2.7 checkouts (e.g. v0.2.6) for a real-hardware
before/after comparison, even though this script itself is new: `cores`
was called `peak_calling_cores` before the --cores unification, so this
falls back to that name if the current checkout's pipeline.run() doesn't
accept `cores`. This file lives outside src/pydreg/, so `git checkout
v0.2.6 -- .` won't remove it -- copy it somewhere outside the repo first
if checking out an older tag some other way (e.g. `git worktree`)."""

import argparse
import logging
import os
import resource
import time
from contextlib import contextmanager

from pydreg import pipeline


def rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1024 / 1024 if os.uname().sysname == "Darwin" else r / 1024


@contextmanager
def _timed_with_rss(name):
    print(f"[RSS {rss_mb():8.0f} MB]  START  {name}", flush=True)
    t0 = time.perf_counter()
    yield
    print(f"[RSS {rss_mb():8.0f} MB]  END    {name}  ({time.perf_counter() - t0:.1f}s)", flush=True)


pipeline._timed = _timed_with_rss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plus_bw")
    parser.add_argument("minus_bw")
    parser.add_argument("out_prefix")
    parser.add_argument("--backend", default=None)
    parser.add_argument("--cores", "-p", type=int, default=1)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    print(f"[RSS {rss_mb():8.0f} MB]  baseline (process start)", flush=True)
    run_kwargs = dict(
        backend_name=args.backend,
        progress=args.verbose,
    )
    try:
        result = pipeline.run(args.plus_bw, args.minus_bw, args.out_prefix, cores=args.cores, **run_kwargs)
    except TypeError:
        # pre-0.2.7 checkouts (e.g. v0.2.6): --cores unification hadn't
        # happened yet, and this was the peak-calling-only knob.
        result = pipeline.run(
            args.plus_bw, args.minus_bw, args.out_prefix, peak_calling_cores=args.cores, **run_kwargs
        )
    print(f"[RSS {rss_mb():8.0f} MB]  FINAL", flush=True)
    print(
        f"dense_infp rows: {len(result['dense_infp'])}, "
        f"peaks: {0 if result['peak_bed'] is None else len(result['peak_bed'])}"
    )


if __name__ == "__main__":
    main()

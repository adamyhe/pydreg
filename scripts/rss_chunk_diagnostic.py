"""Finer-grained version of rss_stage_diagnostic.py: also samples RSS every
N calls to features.extract_features_batch (one call per scoring chunk --
~1000 calls for a few million positions), not just at each pipeline stage
boundary. Distinguishes a one-time warm-up cost (RSS jumps once on an early
chunk, then flat) from something accumulating per chunk (RSS climbs
steadily across chunks) -- the two point at completely different causes
(e.g. numba JIT/thread-pool warm-up vs. ThreadPoolExecutor creation churn
inside extract_features_batch's multi-threaded path) and a single
before/after number for the whole stage can't tell them apart.

Usage: identical to rss_stage_diagnostic.py, plus --sample-every:
    uv run python scripts/rss_chunk_diagnostic.py \\
        data/K562_groseq.pl.bw data/K562_groseq.mn.bw pred/K562_groseq_diag \\
        --cores 16 --sample-every 25

Also works against pre-0.2.7 checkouts (falls back to peak_calling_cores=),
though extract_features_batch's `extra_readers` kwarg didn't exist there --
handled below.
"""

import argparse
import inspect
import logging
import os
import resource
import time
from contextlib import contextmanager

from pydreg import features, pipeline


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

_call_count = 0
_sample_every = 25
_accepts_extra_readers = "extra_readers" in inspect.signature(features.extract_features_batch).parameters
_orig_extract_features_batch = features.extract_features_batch


def _counting_extract_features_batch(*args, **kwargs):
    global _call_count
    if not _accepts_extra_readers:
        kwargs.pop("extra_readers", None)
    result = _orig_extract_features_batch(*args, **kwargs)
    _call_count += 1
    if _call_count == 1 or _call_count % _sample_every == 0:
        print(f"    [RSS {rss_mb():8.0f} MB]  extract_features_batch call #{_call_count}", flush=True)
    return result


features.extract_features_batch = _counting_extract_features_batch
pipeline.features.extract_features_batch = _counting_extract_features_batch


def main():
    global _sample_every
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plus_bw")
    parser.add_argument("minus_bw")
    parser.add_argument("out_prefix")
    parser.add_argument("--backend", default=None)
    parser.add_argument("--cores", "-p", type=int, default=1)
    parser.add_argument("--sample-every", type=int, default=25)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    _sample_every = args.sample_every

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    print(f"[RSS {rss_mb():8.0f} MB]  baseline (process start)", flush=True)
    run_kwargs = dict(backend_name=args.backend, progress=args.verbose)
    try:
        result = pipeline.run(args.plus_bw, args.minus_bw, args.out_prefix, cores=args.cores, **run_kwargs)
    except TypeError:
        result = pipeline.run(
            args.plus_bw, args.minus_bw, args.out_prefix, peak_calling_cores=args.cores, **run_kwargs
        )
    print(f"[RSS {rss_mb():8.0f} MB]  FINAL  ({_call_count} total extract_features_batch calls)", flush=True)
    print(
        f"dense_infp rows: {len(result['dense_infp'])}, "
        f"peaks: {0 if result['peak_bed'] is None else len(result['peak_bed'])}"
    )


if __name__ == "__main__":
    main()

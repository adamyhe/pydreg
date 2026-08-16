"""Instruments BOTH extract_features_batch and Scorer.predict together (on
top of the existing per-stage boundaries), interleaved in call order, since
pipeline._score_positions overlaps each chunk's extraction with the
*previous* chunk's scorer.predict() call running concurrently on another
thread -- isolated tests of either half alone (rss_chunk_diagnostic.py,
rss_scorer_only_diagnostic.py) each plateau immediately on this real
hardware, but the real combined run climbs gradually across ~1000 calls.
This should pin down whether the growth tracks extraction calls, predict
calls, or genuinely only shows up when both run concurrently on real
(not fixed/repeated) data.

Usage: identical to rss_stage_diagnostic.py / rss_chunk_diagnostic.py:
    uv run python scripts/rss_combined_diagnostic.py \\
        data/K562_groseq.pl.bw data/K562_groseq.mn.bw pred/K562_groseq_diag \\
        --cores 16 --sample-every 25
"""

import argparse
import inspect
import logging
import os
import resource
import time
from contextlib import contextmanager

from pydreg import backend, features, pipeline


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

_sample_every = 25
_extract_count = 0
_predict_count = 0
_accepts_extra_readers = "extra_readers" in inspect.signature(features.extract_features_batch).parameters
_orig_extract_features_batch = features.extract_features_batch
_orig_scorer_predict = backend.Scorer.predict


def _counting_extract_features_batch(*args, **kwargs):
    global _extract_count
    if not _accepts_extra_readers:
        kwargs.pop("extra_readers", None)
    result = _orig_extract_features_batch(*args, **kwargs)
    _extract_count += 1
    if _extract_count == 1 or _extract_count % _sample_every == 0:
        print(f"    [RSS {rss_mb():8.0f} MB]  E extract_features_batch #{_extract_count} shape={result.shape}", flush=True)
    return result


def _counting_predict(self, X):
    global _predict_count
    result = _orig_scorer_predict(self, X)
    _predict_count += 1
    if _predict_count == 1 or _predict_count % _sample_every == 0:
        print(f"    [RSS {rss_mb():8.0f} MB]  P scorer.predict      #{_predict_count} shape={X.shape}", flush=True)
    return result


features.extract_features_batch = _counting_extract_features_batch
pipeline.features.extract_features_batch = _counting_extract_features_batch
backend.Scorer.predict = _counting_predict


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
    print(
        f"[RSS {rss_mb():8.0f} MB]  FINAL  "
        f"({_extract_count} extract calls, {_predict_count} predict calls)",
        flush=True,
    )
    print(
        f"dense_infp rows: {len(result['dense_infp'])}, "
        f"peaks: {0 if result['peak_bed'] is None else len(result['peak_bed'])}"
    )


if __name__ == "__main__":
    main()

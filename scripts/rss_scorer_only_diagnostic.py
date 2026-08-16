"""Isolates the scoring backend's own memory behavior from extraction's, by
calling scorer.predict() repeatedly on a fixed input -- no bigWig I/O, no
extraction, nothing else running concurrently. Motivation: rss_chunk_
diagnostic.py samples RSS right after extract_features_batch returns, but
pipeline._score_positions overlaps each chunk's extraction with the
*previous* chunk's scorer.predict() call running concurrently on another
thread -- so those samples can't help but also reflect whatever the scoring
backend is doing at the same time. Repeating extract_features_batch alone
(no scoring at all) 1000x on a real dataset did not reproduce the gradual
RSS growth seen in a real run's "scoring informative positions" step --
this script tests the other half directly.

Usage:
    uv run python scripts/rss_scorer_only_diagnostic.py --backend cupy --calls 1200

Prints RSS every N calls (--sample-every, default 25). Uses the real
pretrained model and a fixed random query batch (same array reused every
call, so any growth can't be blamed on varying input data).
"""

import argparse
import os
import resource

import numpy as np

from pydreg import backend
from pydreg.models import DREGModel


def rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1024 / 1024 if os.uname().sysname == "Darwin" else r / 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--calls", type=int, default=1200)
    parser.add_argument("--sample-every", type=int, default=25)
    parser.add_argument("--vary-last-chunk", action="store_true",
                         help="every 10th call uses a smaller batch (mimics a "
                              "chromosome's ragged last chunk) -- tests whether "
                              "varying array shapes specifically drives growth")
    args = parser.parse_args()

    print(f"[RSS {rss_mb():8.0f} MB]  baseline")
    model = DREGModel.from_pretrained()
    print(f"[RSS {rss_mb():8.0f} MB]  after loading model")

    scorer = backend.build_scorer(model, args.backend)
    print(f"using backend: {scorer.backend}")

    rng = np.random.default_rng(0)
    X_full = model.x_center + model.x_scale * rng.standard_normal((args.batch_size, model.n_features))
    X_full = np.clip(X_full, 0, None)
    X_small = X_full[: args.batch_size // 3]

    for i in range(1, args.calls + 1):
        X = X_small if (args.vary_last_chunk and i % 10 == 0) else X_full
        scorer.predict(X)
        if i == 1 or i % args.sample_every == 0:
            print(f"  call #{i:5d}: RSS = {rss_mb():.0f} MB")

    print(f"[RSS {rss_mb():8.0f} MB]  FINAL ({args.calls} calls)")


if __name__ == "__main__":
    main()

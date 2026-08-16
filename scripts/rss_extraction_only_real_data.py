"""Isolates extraction alone against REAL bigWig files -- no model loading,
no scoring backend, nothing else. Every synthetic-data reproduction of the
"scoring informative positions" RSS growth (rss_chunk_diagnostic.py-style
repeated extract_features_batch calls, at small and large synthetic-genome
scale, sparse and dense position patterns, with and without a concurrent
mock scorer) has come back completely flat -- the one variable that's never
been controlled for is real bigWig file structure itself (real PRO-seq
data is 3'-mapped point-mode read counts, likely far more numerous/smaller
intervals than the smooth synthetic data used elsewhere), and the growth
appeared with BOTH the cupy and numpy scoring backends, which share almost
no code -- ruling out the backend and pointing back at whatever's common
to both: extraction on real data.

Usage:
    uv run python scripts/rss_extraction_only_real_data.py \\
        data/K562_groseq.pl.bw data/K562_groseq.mn.bw --cores 16

Scans real informative positions once (same as a real run), then repeatedly
calls extract_features_batch over real chunks of them -- no scoring, no
model loading beyond the window_sizes/half_n_windows shape needed to call
extract_features_batch identically to a real run.
"""

import argparse
import os
import resource

import numpy as np
import pybigtools

from pydreg import features, infp


def rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1024 / 1024 if os.uname().sysname == "Darwin" else r / 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plus_bw")
    parser.add_argument("minus_bw")
    parser.add_argument("--cores", "-p", type=int, default=1)
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--sample-every", type=int, default=25)
    parser.add_argument("--max-calls", type=int, default=None)
    args = parser.parse_args()

    import numba

    numba.set_num_threads(args.cores)

    print(f"[RSS {rss_mb():8.0f} MB]  baseline")
    bw_plus = pybigtools.open(args.plus_bw)
    bw_minus = pybigtools.open(args.minus_bw)

    print("scanning real informative positions...")
    infp_bed = infp.get_informative_positions(bw_plus, bw_minus)
    print(f"[RSS {rss_mb():8.0f} MB]  after scanning ({len(infp_bed)} positions)")

    # same pretrained model's zoom config, hardcoded here so this doesn't
    # need to download/load the SVR at all -- extraction only cares about
    # window_sizes/half_n_windows shape, not the model weights themselves
    window_sizes = np.array([10, 25, 50, 500, 5000])
    half_n_windows = np.array([10, 10, 30, 20, 20])

    extra_readers = [
        (pybigtools.open(args.plus_bw), pybigtools.open(args.minus_bw))
        for _ in range(max(0, args.cores - 1))
    ]

    max_dist = features.max_dist_from_center(window_sizes, half_n_windows)
    n_reader_pairs = 1 + len(extra_readers)
    cluster_call_count = 0
    multi_worker_call_count = 0

    call_count = 0
    for chrom, group in infp_bed.groupby("chrom", sort=False):
        centers = group["start"].to_numpy()
        for start in range(0, centers.shape[0], args.chunk):
            chunk_centers = centers[start : start + args.chunk]
            sorted_centers = np.sort(chunk_centers)
            clusters = features._build_clusters(sorted_centers, max_dist)
            n_workers = features._cap_workers_for_memory(
                clusters, sorted_centers, max_dist, min(n_reader_pairs, len(clusters))
            )
            if len(clusters) > 1:
                cluster_call_count += 1
            if n_workers > 1:
                multi_worker_call_count += 1

            features.extract_features_batch(
                bw_plus, bw_minus, chrom, chunk_centers, window_sizes, half_n_windows,
                extra_readers=extra_readers,
            )
            call_count += 1
            if call_count == 1 or call_count % args.sample_every == 0:
                print(
                    f"  call #{call_count:5d} ({chrom}): RSS = {rss_mb():.0f} MB  "
                    f"clusters={len(clusters)} n_workers={n_workers}"
                )
            if args.max_calls and call_count >= args.max_calls:
                print(f"[RSS {rss_mb():8.0f} MB]  stopping early at --max-calls={args.max_calls}")
                break
        else:
            continue
        break

    print(
        f"calls with >1 cluster: {cluster_call_count}/{call_count}  "
        f"calls with n_workers>1: {multi_worker_call_count}/{call_count}"
    )

    print(f"[RSS {rss_mb():8.0f} MB]  FINAL ({call_count} extract_features_batch calls)")


if __name__ == "__main__":
    main()

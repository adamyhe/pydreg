#!/usr/bin/env python3
"""Checks figures/timing_scripts_cuda.sh's outputs against the published
benchmark results and summarizes timings, per GPU and precision mode.

Reads every $OUT_DIR/cuda/<gpu-label>/<mode>/ run directory (or the ones
named with --gpus/--modes). Per library, it compares against the published
results in adamyhe/pydreg-supporting-data, which came from the cupy
backend on an NVIDIA Titan Xp (float32, unseeded):

  - scores: max |local - published| over positions in both *.dREG.infp.bw
    tracks, at most --score-atol (default 1e-3);
  - positions: the fraction of positions in only one track, at most
    --position-tol (default 1e-4);
  - peaks: `bedtools jaccard` against real dREG's published peaks, at
    least the published pydreg run's own Jaccard minus --jaccard-tol
    (default 0.001). Peak calling is stochastic and the published run was
    unseeded, so this asks "agrees with dREG as well as before".

These are the same checks as compare_apple_backends.py. One caveat: the
published run predates 0.4.0's fused cupy kernel, which itself moved
scores by up to ~1e-3 (float32 noise, closer to the float64 reference),
so a current-code run can land near or just over --score-atol and still be
correct. Jaccard against dREG is the check that matters most; if only
scores/positions fail by a hair, compare against a pre-0.4.0 run before
treating it as a regression. They gate fp32
runs only: tf32 runs are expected to move scores by a few 1e-3 and peaks
by a few 1e-3 Jaccard, so for them the numbers are reported without
pass/fail.

Timings: wall time and peak RSS from `/usr/bin/time -v`, the
"scoring informative positions" and "calling peaks" stages from pydreg's
-v log, and the ratios to the published dREG run's wall time and RSS. Those
ratios are only a like-for-like comparison on the machine dREG was
benchmarked on (the Titan Xp node); on any other card they compare
different hardware, and are marked so. With both modes, the tf32 speedup
over fp32 is reported too.

The per-run medians and ranges printed at the end are the numbers the
README's Performance section quotes.

Requires `bedtools` on PATH. Downloads each library's published
infp.bw/peak.prob.bed.gz/time.log (pydreg and dREG) on first use, cached
under ~/.cache/huggingface.

Writes $OUT_DIR/cuda/summary.csv. Exits 1 if any fp32 check fails.

Usage:
    uv run python3 figures/compare_cuda_runs.py [--gpus nvidia-titan-xp] [--libs G1 ...]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from _common import LIBRARIES, fetch, parse_time_log
from compare_apple_backends import compare_scores, jaccard, parse_stage_seconds

REPO_ROOT = Path(__file__).resolve().parent.parent
MODES = ("fp32", "tf32")
# The GPU label timing_scripts_cuda.sh derives for the card the published
# dREG and pydreg runs used; only there are the dREG ratios like-for-like.
REFERENCE_GPU_LABELS = {"nvidia-titan-xp"}


def summarize(values):
    values = [v for v in values if v is not None]
    if not values:
        return "-"
    return f"median {np.median(values):.6g} (range {min(values):.6g}-{max(values):.6g}, n={len(values)})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(os.environ.get("OUT_DIR", REPO_ROOT / "figures" / "output")),
        help="timing_scripts_cuda.sh's OUT_DIR (default: figures/output)",
    )
    parser.add_argument("--gpus", nargs="+", default=None, help="GPU labels (default: all)")
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--libs", nargs="+", default=LIBRARIES)
    parser.add_argument("--score-atol", type=float, default=1e-3)
    parser.add_argument("--position-tol", type=float, default=1e-4)
    parser.add_argument("--jaccard-tol", type=float, default=0.001)
    args = parser.parse_args()

    cuda_dir = args.out_dir / "cuda"
    if args.gpus:
        gpus = args.gpus
    elif cuda_dir.exists():
        gpus = sorted(p.name for p in cuda_dir.iterdir() if p.is_dir())
    else:
        gpus = []
    runs = [(g, m) for g in gpus for m in args.modes if (cuda_dir / g / m).is_dir()]
    if not runs:
        print(f"no run directories found under {cuda_dir}")
        return 1

    published = {}

    def reference(lib):
        if lib not in published:
            print(f"{lib}: fetching published results (cached after the first run)...", flush=True)
            pub_peaks = fetch("pydreg", lib, "peak.prob.bed.gz")
            dreg_peaks = fetch("dreg", lib, "peak.prob.bed.gz")
            published[lib] = {
                "infp": fetch("pydreg", lib, "infp.bw"),
                "peaks": pub_peaks,
                "dreg_peaks": dreg_peaks,
                "jaccard": jaccard(dreg_peaks, pub_peaks),
                "dreg_time": parse_time_log(fetch("dreg", lib, "time.log")),
            }
        return published[lib]

    rows, failures = [], []
    for gpu, mode in runs:
        run_dir = cuda_dir / gpu / mode
        for lib in args.libs:
            prefix = run_dir / lib
            if not Path(f"{prefix}.done").exists():
                continue
            ref = reference(lib)
            scores = compare_scores(Path(f"{prefix}.dREG.infp.bw"), ref["infp"])
            local_peaks = Path(f"{prefix}.dREG.peak.prob.bed.gz")
            j_dreg = jaccard(ref["dreg_peaks"], local_peaks)
            timing = parse_time_log(Path(f"{prefix}.time.log")) or {}
            stages = parse_stage_seconds(Path(f"{prefix}.log"))
            dreg_time = ref["dreg_time"] or {}

            checks = {
                "scores": scores["max_abs_diff"] <= args.score_atol,
                "positions": scores["only_frac"] <= args.position_tol,
                "peaks": j_dreg >= ref["jaccard"] - args.jaccard_tol,
            }
            failed = [name for name, ok in checks.items() if not ok]
            gated = mode == "fp32"
            if gated:
                failures += [f"{gpu}/{mode}/{lib}: {name}" for name in failed]

            wall = timing.get("wall_seconds")
            rss_gib = timing["rss_kb"] / 2**20 if "rss_kb" in timing else None
            dreg_wall = dreg_time.get("wall_seconds")
            dreg_rss = dreg_time["rss_kb"] / 2**20 if "rss_kb" in dreg_time else None
            rows.append({
                "gpu": gpu,
                "mode": mode,
                "library": lib,
                "pass": ("yes" if not failed else "NO") if gated else "n/a",
                "max_abs_score_diff": scores["max_abs_diff"],
                "only_in_one_frac": scores["only_frac"],
                "jaccard_vs_dreg": j_dreg,
                "published_jaccard_vs_dreg": ref["jaccard"],
                "jaccard_vs_published_pydreg": jaccard(ref["peaks"], local_peaks),
                "wall_s": wall,
                "scoring_s": stages.get("scoring informative positions"),
                "peak_calling_s": stages.get("calling peaks"),
                "peak_rss_gib": rss_gib,
                "dreg_wall_s": dreg_wall,
                "dreg_peak_rss_gib": dreg_rss,
                "speedup_vs_dreg": dreg_wall / wall if wall and dreg_wall else None,
                "rss_reduction_vs_dreg": dreg_rss / rss_gib if rss_gib and dreg_rss else None,
                "dreg_same_hardware": gpu in REFERENCE_GPU_LABELS,
            })

    if not rows:
        print("no completed runs found")
        return 1

    def fmt(value, spec):
        return "-" if value is None else format(value, spec)

    print(
        f"\n{'gpu/mode':<34}{'library':<18}{'pass':<6}{'max|dscore|':>12}{'only-frac':>11}"
        f"{'J(dREG)':>10}{'J(pub)':>10}{'wall s':>9}{'score s':>9}{'peaks s':>9}"
        f"{'RSS GiB':>9}{'vs dREG':>9}"
    )
    for r in rows:
        print(
            f"{r['gpu'] + '/' + r['mode']:<34}{r['library']:<18}{r['pass']:<6}"
            f"{r['max_abs_score_diff']:>12.2e}{r['only_in_one_frac']:>11.2e}"
            f"{r['jaccard_vs_dreg']:>10.6f}{r['published_jaccard_vs_dreg']:>10.6f}"
            f"{fmt(r['wall_s'], '.1f'):>9}{fmt(r['scoring_s'], '.1f'):>9}"
            f"{fmt(r['peak_calling_s'], '.1f'):>9}{fmt(r['peak_rss_gib'], '.2f'):>9}"
            f"{fmt(r['speedup_vs_dreg'], '.2f') + 'x':>9}"
        )

    print("\nper run (the README's Performance numbers come from the reference GPU's fp32 run):")
    for gpu, mode in runs:
        sel = [r for r in rows if r["gpu"] == gpu and r["mode"] == mode]
        if not sel:
            continue
        same = sel[0]["dreg_same_hardware"]
        print(f"  {gpu}/{mode} ({len(sel)} libraries)")
        print(f"    Jaccard vs dREG:        {summarize([r['jaccard_vs_dreg'] for r in sel])}")
        print(f"    wall-time speedup:      {summarize([r['speedup_vs_dreg'] for r in sel])}"
              + ("" if same else "  [dREG ran on different hardware]"))
        print(f"    peak RSS reduction:     {summarize([r['rss_reduction_vs_dreg'] for r in sel])}")
        print(f"    wall time (s):          {summarize([r['wall_s'] for r in sel])}")

    for gpu in gpus:
        pairs = {
            r["library"]: r for r in rows if r["gpu"] == gpu and r["mode"] == "fp32"
        }
        tf32 = []
        for r in rows:
            fp32 = pairs.get(r["library"])
            if r["gpu"] == gpu and r["mode"] == "tf32" and fp32 and r["wall_s"] and fp32["wall_s"]:
                tf32.append((r["library"], fp32["wall_s"] / r["wall_s"],
                             r["jaccard_vs_dreg"], fp32["jaccard_vs_dreg"]))
        if tf32:
            print(f"\n{gpu}: --tf32 vs fp32")
            print(f"  {'library':<18}{'speedup':>9}{'J(dREG) tf32':>14}{'J(dREG) fp32':>14}")
            for lib, speedup, j_tf32, j_fp32 in tf32:
                print(f"  {lib:<18}{speedup:>8.2f}x{j_tf32:>14.6f}{j_fp32:>14.6f}")

    summary = cuda_dir / "summary.csv"
    with summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {summary}")

    if failures:
        print("\nFAILED:\n  " + "\n  ".join(failures))
        return 1
    print("\nall fp32 equivalence checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

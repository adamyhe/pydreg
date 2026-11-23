#!/usr/bin/env python3
"""Checks figures/timing_scripts_apple.sh's outputs against the published
benchmark results and summarizes mlx vs numpy (CPU) timings.

The published pydreg results (adamyhe/pydreg-supporting-data) came from
the cupy backend on an NVIDIA Titan Xp, i.e. float32 scoring, unseeded. So
"equivalent" here means, per library and backend:

  - scores: max |local - published| over positions present in both
    *.dREG.infp.bw tracks is at most --score-atol (default 1e-3, the same
    tolerance pydreg.backend's own cupy/mlx smoke tests use against the
    float64 reference). numpy scores in float64 and mlx in float32, so
    neither matches the float32 cupy scores bit for bit.
  - positions: the fraction of positions present in only one of the two
    tracks is at most --position-tol (default 1e-4). Not required to be 0:
    the track includes gap-filled and densified positions, which are chosen
    by score thresholds, so score differences at the 1e-4 level can flip a
    handful.
  - peaks: `bedtools jaccard` of the local *.dREG.peak.prob.bed.gz against
    real dREG's published peaks is at least the published pydreg run's own
    Jaccard against dREG minus --jaccard-tol (default 0.001). Peak calling
    is stochastic (QMC) and the published run was unseeded, so this asks
    "agrees with dREG as well as the published run did" rather than for
    identical peaks.

The checks gate numpy and mlx runs only. mlx-tf32 runs (mlx with --tf32)
are expected to move scores by a few 1e-3 and peaks by a few 1e-3
Jaccard on M5-class GPUs, so for them the numbers are reported without
pass/fail, along with their speedup over plain mlx.

Also reported, without pass/fail: Jaccard against the published pydreg
peaks, and numpy-vs-mlx score/peak agreement when both ran.

Timing columns: wall time and peak RSS from `/usr/bin/time -l`, and the
"scoring informative positions" stage (where the backends differ) from
pydreg's -v log. speedup = numpy time / mlx time.

Requires `bedtools` on PATH. Downloads each library's published
infp.bw/peak.prob.bed.gz (pydreg and dREG) on first use, cached under
~/.cache/huggingface.

Writes $OUT_DIR/apple/summary.csv. Exits 1 if any check fails.

Usage:
    uv run python3 figures/compare_apple_backends.py [--libs G1 G3 ...]
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from _common import LIBRARIES, fetch
from plot_peak_agreement import sorted_bed

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKENDS = ("numpy", "mlx", "mlx-tf32")
# Backends the equivalence checks gate; mlx-tf32 is reported, not gated.
GATED = ("numpy", "mlx")

# Per-chromosome reads go in windows of this many bp, bounding memory at a
# few hundred MB however large the chromosome.
_READ_WINDOW_BP = 20_000_000


def jaccard(a: Path, b: Path) -> float:
    a_bed, b_bed = sorted_bed(a), sorted_bed(b)
    try:
        result = subprocess.run(
            ["bedtools", "jaccard", "-a", a_bed, "-b", b_bed],
            capture_output=True,
            text=True,
            check=True,
        )
    finally:
        Path(a_bed).unlink()
        Path(b_bed).unlink()
    header, values = result.stdout.strip().split("\n")
    return float(dict(zip(header.split("\t"), values.split("\t")))["jaccard"])


def compare_scores(a: Path, b: Path) -> dict:
    """Max abs score difference over positions defined in both bigWigs, and
    how many positions are defined in only one."""
    from figwig import BigWigReader

    bw_a, bw_b = BigWigReader(str(a)), BigWigReader(str(b))
    max_diff, n_shared, n_only = 0.0, 0, 0
    for chrom in sorted(set(bw_a.chrom_sizes) | set(bw_b.chrom_sizes)):
        if chrom not in bw_a.chrom_sizes or chrom not in bw_b.chrom_sizes:
            other = bw_a if chrom in bw_a.chrom_sizes else bw_b
            size = other.chrom_sizes[chrom]
            for start in range(0, size, _READ_WINDOW_BP):
                width = min(_READ_WINDOW_BP, size - start)
                v = other.read([chrom], [start], width=width, missing=np.nan)[0]
                n_only += int(np.isfinite(v).sum())
            continue
        size = min(bw_a.chrom_sizes[chrom], bw_b.chrom_sizes[chrom])
        for start in range(0, size, _READ_WINDOW_BP):
            width = min(_READ_WINDOW_BP, size - start)
            va = bw_a.read([chrom], [start], width=width, missing=np.nan)[0]
            vb = bw_b.read([chrom], [start], width=width, missing=np.nan)[0]
            fa, fb = np.isfinite(va), np.isfinite(vb)
            both = fa & fb
            n_shared += int(both.sum())
            n_only += int((fa ^ fb).sum())
            if both.any():
                diff = np.abs(va[both].astype(np.float64) - vb[both].astype(np.float64))
                max_diff = max(max_diff, float(diff.max()))
    return {
        "max_abs_diff": max_diff,
        "n_shared": n_shared,
        "only_frac": n_only / max(n_shared + n_only, 1),
    }


def parse_time_log(path: Path) -> dict | None:
    """Wall seconds and peak RSS (GiB) from a macOS `/usr/bin/time -l` log."""
    if not path.exists():
        return None
    text = path.read_text()
    real = re.search(r"([\d.]+) real", text)
    rss = re.search(r"(\d+)\s+maximum resident set size", text)
    if not (real and rss):
        return None
    return {"wall_s": float(real.group(1)), "rss_gib": int(rss.group(1)) / 2**30}


def parse_stage_seconds(path: Path) -> dict:
    """{stage name: seconds} from pydreg's -v log ("<stage> done in <s>s")."""
    if not path.exists():
        return {}
    return {
        m.group(1): float(m.group(2))
        for m in re.finditer(r"INFO (.+?) done in ([\d.]+)s", path.read_text())
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(os.environ.get("OUT_DIR", REPO_ROOT / "figures" / "output")),
        help="timing_scripts_apple.sh's OUT_DIR (default: figures/output)",
    )
    parser.add_argument("--libs", nargs="+", default=LIBRARIES)
    parser.add_argument("--score-atol", type=float, default=1e-3)
    parser.add_argument("--position-tol", type=float, default=1e-4)
    parser.add_argument("--jaccard-tol", type=float, default=0.001)
    args = parser.parse_args()

    apple_dir = args.out_dir / "apple"
    rows, failures = [], []
    for lib in args.libs:
        done = [b for b in BACKENDS if (apple_dir / b / f"{lib}.done").exists()]
        if not done:
            print(f"{lib}: no completed runs in {apple_dir}, skipping")
            continue

        print(f"{lib}: fetching published results (cached after the first run)...", flush=True)
        published_infp = fetch("pydreg", lib, "infp.bw")
        published_peaks = fetch("pydreg", lib, "peak.prob.bed.gz")
        dreg_peaks = fetch("dreg", lib, "peak.prob.bed.gz")
        reference_j = jaccard(dreg_peaks, published_peaks)

        for backend in done:
            prefix = apple_dir / backend / lib
            scores = compare_scores(Path(f"{prefix}.dREG.infp.bw"), published_infp)
            local_peaks = Path(f"{prefix}.dREG.peak.prob.bed.gz")
            j_dreg = jaccard(dreg_peaks, local_peaks)
            timing = parse_time_log(Path(f"{prefix}.time.log")) or {}
            stages = parse_stage_seconds(Path(f"{prefix}.log"))

            checks = {
                "scores": scores["max_abs_diff"] <= args.score_atol,
                "positions": scores["only_frac"] <= args.position_tol,
                "peaks": j_dreg >= reference_j - args.jaccard_tol,
            }
            failed = [name for name, ok in checks.items() if not ok]
            gated = backend in GATED
            if gated:
                failures += [f"{lib} ({backend}): {name}" for name in failed]
            rows.append({
                "library": lib,
                "backend": backend,
                "pass": ("yes" if not failed else "NO") if gated else "n/a",
                "max_abs_score_diff": scores["max_abs_diff"],
                "only_in_one_frac": scores["only_frac"],
                "jaccard_vs_dreg": j_dreg,
                "published_jaccard_vs_dreg": reference_j,
                "jaccard_vs_published_pydreg": jaccard(published_peaks, local_peaks),
                "wall_s": timing.get("wall_s"),
                "scoring_s": stages.get("scoring informative positions"),
                "peak_rss_gib": timing.get("rss_gib"),
            })

        if set(GATED) <= set(done):
            pair = compare_scores(
                apple_dir / "numpy" / f"{lib}.dREG.infp.bw",
                apple_dir / "mlx" / f"{lib}.dREG.infp.bw",
            )
            pair_j = jaccard(
                apple_dir / "numpy" / f"{lib}.dREG.peak.prob.bed.gz",
                apple_dir / "mlx" / f"{lib}.dREG.peak.prob.bed.gz",
            )
            print(
                f"{lib}: numpy vs mlx max |score diff| {pair['max_abs_diff']:.2e}, "
                f"positions in only one {pair['only_frac']:.2e}, peak Jaccard {pair_j:.6f}"
            )

    if not rows:
        print(f"no completed runs found under {apple_dir}")
        return 1

    def fmt(value, spec):
        return "-" if value is None else format(value, spec)

    print(
        f"\n{'library':<18}{'backend':<10}{'pass':<6}{'max|dscore|':>12}{'only-frac':>11}"
        f"{'J(dREG)':>10}{'J(pub)':>10}{'wall s':>9}{'score s':>9}{'RSS GiB':>9}"
    )
    for r in rows:
        print(
            f"{r['library']:<18}{r['backend']:<10}{r['pass']:<6}"
            f"{r['max_abs_score_diff']:>12.2e}{r['only_in_one_frac']:>11.2e}"
            f"{r['jaccard_vs_dreg']:>10.6f}{r['published_jaccard_vs_dreg']:>10.6f}"
            f"{fmt(r['wall_s'], '.1f'):>9}{fmt(r['scoring_s'], '.1f'):>9}"
            f"{fmt(r['peak_rss_gib'], '.2f'):>9}"
        )

    by_lib = {}
    for r in rows:
        by_lib.setdefault(r["library"], {})[r["backend"]] = r
    speedups = [
        (lib, b["numpy"]["wall_s"] / b["mlx"]["wall_s"],
         b["numpy"]["scoring_s"] / b["mlx"]["scoring_s"])
        for lib, b in by_lib.items()
        if set(GATED) <= set(b)
        and all(b[k]["wall_s"] and b[k]["scoring_s"] for k in GATED)
    ]
    if speedups:
        print(f"\n{'library':<18}{'wall speedup':>14}{'scoring speedup':>17}")
        for lib, wall, scoring in speedups:
            print(f"{lib:<18}{wall:>13.2f}x{scoring:>16.2f}x")
        print(
            f"{'median':<18}{np.median([s[1] for s in speedups]):>13.2f}x"
            f"{np.median([s[2] for s in speedups]):>16.2f}x"
        )

    tf32 = [
        (lib, b["mlx"]["wall_s"] / b["mlx-tf32"]["wall_s"],
         b["mlx"]["scoring_s"] / b["mlx-tf32"]["scoring_s"],
         b["mlx-tf32"]["jaccard_vs_dreg"], b["mlx"]["jaccard_vs_dreg"])
        for lib, b in by_lib.items()
        if {"mlx", "mlx-tf32"} <= set(b)
        and all(b[k]["wall_s"] and b[k]["scoring_s"] for k in ("mlx", "mlx-tf32"))
    ]
    if tf32:
        print(
            f"\n--tf32 vs mlx\n{'library':<18}{'wall speedup':>14}{'scoring speedup':>17}"
            f"{'J(dREG) tf32':>14}{'J(dREG) mlx':>13}"
        )
        for lib, wall, scoring, j_tf32, j_mlx in tf32:
            print(f"{lib:<18}{wall:>13.2f}x{scoring:>16.2f}x{j_tf32:>14.6f}{j_mlx:>13.6f}")

    summary = apple_dir / "summary.csv"
    with summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {summary}")

    if failures:
        print("\nFAILED:\n  " + "\n  ".join(failures))
        return 1
    print("\nall equivalence checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

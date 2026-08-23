#!/usr/bin/env python3
"""Panel (e): per-library agreement between pydreg's and real dREG's called
peaks, as `bedtools jaccard` on *.dREG.peak.prob.bed.gz -- the same file and
tool this project's own historical benchmark used (see
figures/timing_scripts_compare.sh).

Requires `bedtools` on PATH.

Usage:
    uv run python3 figures/plot_peak_agreement.py
"""

from __future__ import annotations

import gzip
import statistics
import subprocess
import tempfile
from pathlib import Path

from _common import LIBRARIES, PLOTS_DIR, SVG_STYLE, escape, fetch, nice_ticks


def sorted_bed(path: Path) -> str:
    """bedtools jaccard requires both inputs sorted the same way; decompress
    and sort once into a plain temp file. `subprocess`'s stdin= needs a real
    file descriptor, which would hand bedtools the still-gzipped bytes
    directly if given a GzipFile -- decompress into memory first instead."""
    data = gzip.open(path, "rb").read()
    tmp = tempfile.NamedTemporaryFile(suffix=".bed", delete=False)
    subprocess.run(["bedtools", "sort", "-i", "-"], input=data, stdout=tmp, check=True)
    tmp.close()
    return tmp.name


def jaccard(lib: str) -> tuple[float, int]:
    dreg_bed = sorted_bed(fetch("dreg", lib, "peak.prob.bed.gz"))
    pydreg_bed = sorted_bed(fetch("pydreg", lib, "peak.prob.bed.gz"))
    result = subprocess.run(
        ["bedtools", "jaccard", "-a", dreg_bed, "-b", pydreg_bed],
        capture_output=True,
        text=True,
        check=True,
    )
    Path(dreg_bed).unlink()
    Path(pydreg_bed).unlink()
    header, values = result.stdout.strip().split("\n")
    row = dict(zip(header.split("\t"), values.split("\t")))
    return float(row["jaccard"]), int(row["n_intersections"])


def box_whisker_svg(rows: list[dict]) -> str:
    """Single vertical box+whisker summarizing the per-library Jaccard
    values (Tukey fences: whiskers extend to the most extreme value within
    1.5*IQR of the box, anything further out is drawn as its own point),
    with every library's actual value overlaid as a dot -- n=12 is too few
    to summarize as just a box without also showing the real points. Same
    height (and margins) as the other panels so it stacks cleanly into one
    figure, but a narrower canvas -- a single box has no use for the same
    440px-wide plot area a scatter needs, and stretching it that wide just
    leaves dead space on both sides."""
    values = sorted(r["jaccard"] for r in rows)
    q1, med, q3 = statistics.quantiles(values, n=4, method="inclusive")
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    in_range = [v for v in values if lower_fence <= v <= upper_fence]
    whisker_lo = min(in_range) if in_range else min(values)
    whisker_hi = max(in_range) if in_range else max(values)

    # Transpose of the original horizontal panel's 560x280 canvas. margin_left
    # has to be wide enough for BOTH the rotated y-axis label AND ~6-digit
    # tick text at the shared 24px tick/axis font -- narrower and they
    # collide.
    width, height = 280, 620
    margin_left, margin_top, margin_right, margin_bottom = 150, 75, 20, 100
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    right = margin_left + plot_w
    bottom = margin_top + plot_h

    hi = 1.0
    span = hi - min(values)
    pad = max(span * 0.2, 0.0005)
    lo = max(0.0, min(values) - pad)

    def sy(v):
        return margin_top + (1 - (v - lo) / (hi - lo)) * plot_h

    cx = margin_left + plot_w / 2
    box_w = 80
    cap_w = 40

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        "<title>pydreg vs dREG called-peak agreement (Jaccard index)</title>",
        SVG_STYLE,
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        '<text x="10" y="34" class="title">Agreement</text>',
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_w}" height="{plot_h}" class="plot"/>',
    ]

    for tick in nice_ticks(lo, hi, 5):
        ty = sy(tick)
        parts.append(
            f'<line x1="{margin_left}" y1="{ty:.1f}" x2="{right}" y2="{ty:.1f}" class="grid"/>'
        )
        parts.append(
            f'<text x="{margin_left - 10}" y="{ty + 8:.1f}" class="tick" text-anchor="end">{tick:.4f}</text>'
        )

    ref_y = sy(1.0)
    parts.append(
        f'<line x1="{margin_left}" y1="{ref_y:.1f}" x2="{right}" y2="{ref_y:.1f}" class="ref"/>'
    )

    y_lo_w, y_hi_w = sy(whisker_lo), sy(whisker_hi)
    y_q1, y_q3, y_med = sy(q1), sy(q3), sy(med)

    parts.extend(
        [
            f'<line x1="{cx:.1f}" y1="{y_lo_w:.1f}" x2="{cx:.1f}" y2="{y_q1:.1f}" class="stem"/>',
            f'<line x1="{cx:.1f}" y1="{y_q3:.1f}" x2="{cx:.1f}" y2="{y_hi_w:.1f}" class="stem"/>',
            f'<line x1="{cx - cap_w / 2:.1f}" y1="{y_lo_w:.1f}" x2="{cx + cap_w / 2:.1f}" y2="{y_lo_w:.1f}" class="stem"/>',
            f'<line x1="{cx - cap_w / 2:.1f}" y1="{y_hi_w:.1f}" x2="{cx + cap_w / 2:.1f}" y2="{y_hi_w:.1f}" class="stem"/>',
            f'<rect x="{cx - box_w / 2:.1f}" y="{min(y_q1, y_q3):.1f}" width="{box_w:.1f}" height="{abs(y_q3 - y_q1):.1f}" '
            'fill="#dce9f8" stroke="#174a6e" stroke-width="1.5"/>',
            f'<line x1="{cx - box_w / 2:.1f}" y1="{y_med:.1f}" x2="{cx + box_w / 2:.1f}" y2="{y_med:.1f}" '
            'stroke="#174a6e" stroke-width="2"/>',
        ]
    )

    for row in rows:
        y = sy(row["jaccard"])
        # Deterministic (not random) small horizontal spread so overlapping
        # points stay distinguishable -- reruns must produce byte-identical
        # output, so this is a hash-free, stable function of the library
        # name rather than `random`.
        jitter = (sum(ord(c) for c in row["library"]) % 5 - 2) * 12
        x = cx + jitter
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" class="mark">'
            f"<title>{escape(row['library'])}: {row['jaccard']:.6f}</title></circle>"
        )

    parts.extend(
        [
            f'<text x="30" y="{margin_top + plot_h / 2:.1f}" class="axis" text-anchor="middle" '
            f'transform="rotate(-90 30 {margin_top + plot_h / 2:.1f})">'
            "Jaccard index</text>",
            f'<text x="{cx:.1f}" y="{bottom + 55}" class="axis" text-anchor="middle">pydreg/dREG</text>',
            f'<text x="10" y="{margin_top - 15}" class="note" text-anchor="start">'
            f"n={len(values)} median={med:.4f}</text>",
            "</svg>",
        ]
    )
    return "\n".join(parts)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for lib in LIBRARIES:
        j, n_intersections = jaccard(lib)
        print(f"{lib}: jaccard={j:.6f} n_intersections={n_intersections}")
        rows.append({"library": lib, "jaccard": j})

    rows.sort(key=lambda r: r["jaccard"])
    svg = box_whisker_svg(rows)
    out = PLOTS_DIR / "peak_agreement.svg"
    out.write_text(svg + "\n")
    print(f"Wrote {out}")

    values = [r["jaccard"] for r in rows]
    print(
        f"n={len(values)} libraries, median jaccard {statistics.median(values):.6f}, "
        f"min {min(values):.6f}, max {max(values):.6f}"
    )


if __name__ == "__main__":
    main()

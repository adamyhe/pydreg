#!/usr/bin/env python3
"""Panel: per-operation efficiency -- positions processed per GPU kernel
launch and per Host-to-Device memcpy call, dREG vs pydreg, one row per
library across the full 12-library benchmark. This is the direct
"inefficiency" chart the other two panels don't make explicit:
plot_gpu_utilization.py shows the GPU sitting idle (a scheduling
symptom), plot_gpu_time_breakdown.py shows totals and raw counts side by
side, but neither computes the number that actually says "each individual
dREG operation does far less useful work than pydreg's" -- this does.

Dot plots on a log-scale axis, not bar charts: the two tools differ by
one-to-three orders of magnitude, and bar *length* under a log scale no
longer represents proportional magnitude the way position along a log
axis does -- the same reason figures/_common.py's scatter panels use
log-scale point marks rather than log-scale bars.

Each library's position count is read from its own runs (the analyzer
parses it out of both tools' logs and cross-checks the two agree), so the
per-operation numbers are per-library rather than divided by one
hardcoded constant.

Reads gpu_out/summary_dreg_<LIB>_vs_pydreg_<LIB>.json, as produced by
analyze_gpu_profile.py -- run that first.

Usage:
    python3 figures/gpu_profiling/plot_gpu_efficiency.py
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _gpu_common import (  # noqa: E402
    COLOR_DREG,
    COLOR_PYDREG,
    SVG_STYLE,
    add_common_args,
    escape,
    resolve_pairs,
    write_svg,
)

ROW_H = 26
PANEL_GAP = 78

PANELS = [
    ("Positions per kernel launch", "kernel", "positions/launch"),
    ("Positions per H2D memcpy call", "memcpy", "positions/call"),
]


def log_x(value, low, high, x0, w):
    frac = (math.log(value) - math.log(low)) / (math.log(high) - math.log(low))
    return x0 + frac * w


def load_metrics(pair):
    """Per-operation efficiency for one library's two runs.

    The dominant kernel (rows are pre-sorted by total time descending, so
    index 0) and the Host-to-Device memcpy specifically -- matching exactly
    the statistic already vetted into docs/PERF_LOG.md and README.md, not a
    sum across every kernel/memcpy type, which would be a different and
    un-vetted number that silently disagreed with the prose.
    """
    by_label = pair.load_summary()
    positions = pair.positions(by_label)
    if positions is None:
        print(
            f"note: [{pair.library}] neither log states an informative-position "
            "count -- skipping (per-operation efficiency needs it)"
        )
        return None

    out = {"positions": positions}
    for tool, label in (("dreg", pair.dreg_label), ("pydreg", pair.pydreg_label)):
        s = by_label[label]
        if not s.get("nsys_cuda_gpu_kern_sum"):
            print(
                f"note: [{pair.library}] {label} has no nsys kernel data "
                "(captured without nsys on PATH?) -- skipping library"
            )
            return None
        kern = s["nsys_cuda_gpu_kern_sum"][0]
        h2d = next(
            (r for r in s["nsys_cuda_gpu_mem_time_sum"] if "Host-to-Device" in r["Name"]),
            None,
        )
        if h2d is None:
            print(f"note: [{pair.library}] {label} has no H2D memcpy rows -- skipping")
            return None
        out[tool] = {
            "kernel": positions / kern["Instances"],
            "kernel_ns": kern["Avg (ns)"],
            "kernel_name": kern["Name"],
            "memcpy": positions / h2d["Instances"],
            "memcpy_ns": h2d["Avg (ns)"],
        }
    return out


def axis_bounds(values):
    """Decade-aligned log-axis bounds enclosing every plotted value, with
    a half-decade of breathing room so no dot lands on the frame."""
    low = 10 ** math.floor(math.log10(min(values)) - 0.15)
    high = 10 ** math.ceil(math.log10(max(values)) + 0.15)
    return low, high


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()

    pairs = resolve_pairs(args, require="summary")
    rows = [(p.library, m) for p in pairs if (m := load_metrics(p))]
    if not rows:
        raise SystemExit("no library had usable nsys kernel/memcpy data to plot")

    keys = [key for _, key, _ in PANELS]
    low, high = axis_bounds(
        [m[tool][key] for _, m in rows for tool in ("dreg", "pydreg") for key in keys]
    )
    ticks = [10 ** e for e in range(int(math.log10(low)), int(math.log10(high)) + 1)]

    width = 960
    margin_left, margin_right = 196, 122
    top_margin, bottom_margin = 136, 56
    legend_y = 86  # below the subtitle (y=58), above the first panel title
    plot_w = width - margin_left - margin_right
    panel_h = len(rows) * ROW_H
    height = top_margin + 2 * panel_h + PANEL_GAP + bottom_margin

    median_gaps = {
        key: statistics.median(m["pydreg"][key] / m["dreg"][key] for _, m in rows)
        for _, key, _ in PANELS
    }

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        "<title>Positions processed per GPU operation: dREG vs pydreg, per library, "
        "log scale</title>",
        SVG_STYLE,
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        f'<text x="{margin_left}" y="34" class="title">Work done per GPU operation</text>',
        f'<text x="{margin_left}" y="58" class="subtitle">'
        f"{len(rows)} librar{'y' if len(rows) == 1 else 'ies'}, log scale, higher = "
        f"more efficient -- median gap {median_gaps['kernel']:,.0f}x per kernel launch, "
        f"{median_gaps['memcpy']:,.0f}x per memcpy call</text>",
    ]

    for i, (color, label) in enumerate([(COLOR_DREG, "dREG"), (COLOR_PYDREG, "pydreg")]):
        lx = margin_left + i * 104
        parts.append(f'<circle cx="{lx}" cy="{legend_y}" r="7" fill="{color}"/>')
        parts.append(
            f'<text x="{lx + 14}" y="{legend_y + 5}" class="legend">{escape(label)}</text>'
        )

    for panel_i, (panel_title, key, unit) in enumerate(PANELS):
        y_top = top_margin + panel_i * (panel_h + PANEL_GAP)
        parts.append(
            f'<text x="{margin_left}" y="{y_top - 12}" class="colhead">'
            f"{escape(panel_title)}</text>"
        )
        for tick in ticks:
            x = log_x(tick, low, high, margin_left, plot_w)
            parts.append(
                f'<line x1="{x:.1f}" y1="{y_top}" x2="{x:.1f}" y2="{y_top + panel_h}" '
                'class="grid"/>'
            )
            parts.append(
                f'<text x="{x:.1f}" y="{y_top + panel_h + 22}" class="tick" '
                f'text-anchor="middle">{tick:,g}</text>'
            )
        parts.append(
            f'<text x="{margin_left + plot_w / 2:.1f}" y="{y_top + panel_h + 44}" '
            f'class="axis" text-anchor="middle">{escape(unit)} (log scale)</text>'
        )

        for row_i, (library, m) in enumerate(rows):
            y = y_top + (row_i + 0.5) * ROW_H
            parts.append(
                f'<text x="{margin_left - 18}" y="{y + 5:.1f}" class="rowlabel" '
                f'text-anchor="end">{escape(library)}</text>'
            )
            xd = log_x(m["dreg"][key], low, high, margin_left, plot_w)
            xp = log_x(m["pydreg"][key], low, high, margin_left, plot_w)
            parts.append(
                f'<line x1="{xd:.1f}" y1="{y:.1f}" x2="{xp:.1f}" y2="{y:.1f}" '
                'class="connector"/>'
            )
            for x, tool, color, name in (
                (xd, "dreg", COLOR_DREG, "dREG"),
                (xp, "pydreg", COLOR_PYDREG, "pydreg"),
            ):
                val = m[tool][key]
                ns = m[tool][f"{key}_ns"]
                parts.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{color}" '
                    f'stroke="#fff" stroke-width="1.5">'
                    f"<title>{escape(library)} {escape(name)}: {val:,.1f} {unit}, "
                    f"{ns / 1e3:,.2f}us avg per op, {m['positions']:,} positions"
                    "</title></circle>"
                )
            gap = m["pydreg"][key] / m["dreg"][key]
            parts.append(
                f'<text x="{margin_left + plot_w + 16}" y="{y + 5:.1f}" class="gap">'
                f"{gap:,.0f}x</text>"
            )

    parts.append("</svg>")
    write_svg(parts, "gpu_efficiency.svg")

    for _, key, unit in PANELS:
        print(f"\n{unit}:")
        for library, m in rows:
            print(
                f"  {library}: dREG {m['dreg'][key]:,.1f} "
                f"({m['dreg'][f'{key}_ns'] / 1e3:,.2f}us/op), "
                f"pydreg {m['pydreg'][key]:,.1f} "
                f"({m['pydreg'][f'{key}_ns'] / 1e3:,.2f}us/op), "
                f"gap {m['pydreg'][key] / m['dreg'][key]:,.1f}x"
            )


if __name__ == "__main__":
    main()

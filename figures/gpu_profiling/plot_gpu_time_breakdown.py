#!/usr/bin/env python3
"""Panel: where dREG's and pydreg's GPU time actually goes -- scheduling
idle time vs. kernel compute time vs. memcpy time -- for the identical
informative-positions scoring phase, one library per row across the full
12-library benchmark. Both sides are phase-exact: dREG's run_predict.bsh
is already exactly this one phase, and pydreg's numbers use its
nsys-windowed data (see docs/PERF_LOG.md's 2026-08-18 (cont.) entries),
not its whole three-phase run.

Horizontal bars, unlike the single-library version this replaces: with 12
libraries the category axis carries names like "Jurkat_ChROseq_1", which
read straight across as row labels but would need rotating (or truncating)
as vertical-bar x-labels. Bar *length* is time in seconds on one shared
axis across every library, so both the within-library dREG-vs-pydreg gap
and the between-library size differences are directly comparable.

Kernel-launch and memcpy *call counts* aren't drawn here -- memcpy's total
time is a sliver of either bar (a fraction of a percent), so it can't
carry a count in its geometry, and 24 bars' worth of count annotations
would swamp the figure. plot_gpu_efficiency.py is the panel that turns
those counts into a per-operation claim; here they're in the tooltips.

Reads gpu_out/summary_dreg_<LIB>_vs_pydreg_<LIB>.json, as produced by
analyze_gpu_profile.py -- run that first. No plotting library dependency,
matching the other figures/plot_*.py scripts -- hand-built SVG.

Usage:
    python3 figures/gpu_profiling/plot_gpu_time_breakdown.py
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _gpu_common import (  # noqa: E402
    COLOR_IDLE,
    COLOR_KERNEL,
    COLOR_MEMCPY,
    SVG_STYLE,
    add_common_args,
    escape,
    resolve_pairs,
    write_svg,
)

BAR_H = 17
BAR_GAP = 5
GROUP_GAP = 24

SEGMENTS = (
    ("idle", COLOR_IDLE, "scheduling idle"),
    ("kernel", COLOR_KERNEL, "kernel compute"),
    ("memcpy", COLOR_MEMCPY, "memcpy"),
)


def load_group(summary):
    idle_s = summary["nsys_idle_gaps"]["idle_ns"] / 1e9
    kernel_s = sum(r["Total Time (ns)"] for r in summary["nsys_cuda_gpu_kern_sum"]) / 1e9
    memcpy_s = (
        sum(r["Total Time (ns)"] for r in summary["nsys_cuda_gpu_mem_time_sum"]) / 1e9
    )
    # Instance counts: the *dominant* kernel (rows are pre-sorted by total
    # time descending, so index 0) and the Host-to-Device memcpy
    # specifically -- matching exactly the numbers already vetted into
    # docs/PERF_LOG.md/README.md, not a sum across every kernel/memcpy type
    # (a different, larger, un-vetted statistic that would silently
    # disagree with the prose).
    kernel_n = summary["nsys_cuda_gpu_kern_sum"][0]["Instances"]
    h2d_n = next(
        r["Instances"]
        for r in summary["nsys_cuda_gpu_mem_time_sum"]
        if "Host-to-Device" in r["Name"]
    )
    return {
        "idle": idle_s,
        "kernel": kernel_s,
        "memcpy": memcpy_s,
        "total": idle_s + kernel_s + memcpy_s,
        "kernel_n": kernel_n,
        "memcpy_n": h2d_n,
        "windowed": summary.get("nsys_windowed"),
    }


def load_rows(pairs):
    rows = []
    for pair in pairs:
        by_label = pair.load_summary()
        try:
            dreg = load_group(by_label[pair.dreg_label])
            pydreg = load_group(by_label[pair.pydreg_label])
        except (KeyError, IndexError, StopIteration):
            print(
                f"note: [{pair.library}] incomplete nsys kernel/memcpy/idle data "
                "-- skipping (captured without nsys on PATH?)"
            )
            continue
        rows.append((pair.library, dreg, pydreg, pair.positions(by_label)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()

    rows = load_rows(resolve_pairs(args, require="summary"))
    if not rows:
        raise SystemExit("no library had usable nsys data to plot")

    max_total = max(max(d["total"], p["total"]) for _, d, p, _ in rows)
    group_h = 2 * BAR_H + BAR_GAP

    width = 1040
    margin_left, margin_right = 190, 250
    top_margin, bottom_margin = 116, 66
    plot_w = width - margin_left - margin_right
    plot_h = len(rows) * (group_h + GROUP_GAP)
    height = top_margin + plot_h + bottom_margin

    def x_for(seconds):
        return margin_left + (seconds / max_total) * plot_w

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        "<title>Where dREG's and pydreg's GPU time goes, per library, "
        "informative-positions scoring phase</title>",
        SVG_STYLE,
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        f'<text x="{margin_left}" y="34" class="title">Where the GPU time actually '
        "goes</text>",
        f'<text x="{margin_left}" y="58" class="subtitle">'
        f"{len(rows)} librar{'y' if len(rows) == 1 else 'ies'}, both sides phase-exact "
        "-- shared time axis, dREG on top of each pair</text>",
    ]

    # memcpy's share of GPU time is a fraction of a percent on both tools,
    # so its segment is sub-pixel at any sane figure width -- say that in
    # the legend instead of leaving a swatch the reader can't find in the
    # bars. plot_gpu_efficiency.py is where memcpy *counts* get their due.
    max_memcpy_pct = max(
        100 * g["memcpy"] / g["total"] for _, d, p_, _ in rows for g in (d, p_)
    )
    memcpy_label = (
        "Memcpy (<1% of every bar)" if max_memcpy_pct < 1 else "Memcpy"
    )
    for i, (color, label) in enumerate(
        [(COLOR_IDLE, "Scheduling idle"), (COLOR_KERNEL, "Kernel compute"), (COLOR_MEMCPY, memcpy_label)]
    ):
        lx = margin_left + i * 172
        parts.append(f'<rect x="{lx}" y="{top_margin - 50}" width="16" height="16" fill="{color}"/>')
        parts.append(
            f'<text x="{lx + 24}" y="{top_margin - 37}" class="legend">{escape(label)}</text>'
        )

    # A decade-ish tick step that lands on a round number for any sweep.
    step = next(
        s for s in (50, 100, 250, 500, 1000, 2500, 5000, 10000) if max_total / s <= 6
    )
    tick = 0
    while tick <= max_total:
        x = x_for(tick)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top_margin}" x2="{x:.1f}" '
            f'y2="{top_margin + plot_h}" class="grid"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{top_margin + plot_h + 24}" class="tick" '
            f'text-anchor="middle">{tick:,g}s</text>'
        )
        tick += step

    for row_i, (library, dreg, pydreg, positions) in enumerate(rows):
        y_group = top_margin + row_i * (group_h + GROUP_GAP)
        parts.append(
            f'<text x="{margin_left - 16}" y="{y_group + group_h / 2 + 5:.1f}" '
            f'class="rowlabel" text-anchor="end">{escape(library)}</text>'
        )
        for bar_i, (name, g) in enumerate((("dREG", dreg), ("pydreg", pydreg))):
            y = y_group + bar_i * (BAR_H + BAR_GAP)
            cum = 0.0
            for key, color, seg_label in SEGMENTS:
                val = g[key]
                x0, x1 = x_for(cum), x_for(cum + val)
                parts.append(
                    f'<rect x="{x0:.1f}" y="{y}" width="{max(x1 - x0, 0.4):.1f}" '
                    f'height="{BAR_H}" fill="{color}">'
                    f"<title>{escape(library)} {escape(name)} {escape(seg_label)}: "
                    f"{val:,.1f}s ({100 * val / g['total']:.1f}% of {g['total']:,.0f}s); "
                    f"{g['kernel_n']:,} kernel launches, {g['memcpy_n']:,} H2D memcpy "
                    f"calls, {positions:,} positions</title></rect>"
                )
                cum += val
            parts.append(
                f'<text x="{x_for(g["total"]) + 8:.1f}" y="{y + BAR_H - 4}" '
                f'class="panelstat">{name} {g["total"]:,.0f}s</text>'
            )

        ratio = dreg["total"] / pydreg["total"] if pydreg["total"] else float("nan")
        parts.append(
            f'<text x="{width - 16}" y="{y_group + group_h / 2 + 5:.1f}" class="gap" '
            f'text-anchor="end">{ratio:.1f}x</text>'
        )

    parts.append(
        f'<text x="{margin_left + plot_w / 2:.1f}" y="{height - 22}" class="axis" '
        'text-anchor="middle">GPU time during the informative-positions scoring '
        "phase (seconds)</text>"
    )
    parts.append("</svg>")
    write_svg(parts, "gpu_time_breakdown.svg")

    total_ratios = [d["total"] / p["total"] for _, d, p, _ in rows if p["total"]]
    busy_ratios = [
        (d["kernel"] + d["memcpy"]) / (p["kernel"] + p["memcpy"])
        for _, d, p, _ in rows
        if p["kernel"] + p["memcpy"]
    ]
    for library, dreg, pydreg, _ in rows:
        print(
            f"{library}: dREG idle={dreg['idle']:,.1f}s kernel={dreg['kernel']:,.1f}s "
            f"memcpy={dreg['memcpy']:,.1f}s (windowed={dreg['windowed']}) | "
            f"pydreg idle={pydreg['idle']:,.1f}s kernel={pydreg['kernel']:,.1f}s "
            f"memcpy={pydreg['memcpy']:,.1f}s (windowed={pydreg['windowed']})"
        )
    print(
        f"\nmedian total ratio {statistics.median(total_ratios):.2f}x, "
        f"median busy-only ratio {statistics.median(busy_ratios):.2f}x "
        f"across {len(rows)} librar{'y' if len(rows) == 1 else 'ies'}"
    )


if __name__ == "__main__":
    main()

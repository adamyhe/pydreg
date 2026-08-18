#!/usr/bin/env python3
"""Panel: where dREG's and pydreg's GPU time actually goes, for the
identical 5,617,218-position informative-positions scoring phase --
scheduling idle time vs. kernel compute time vs. memcpy time. Both sides
are phase-exact: dREG's run_predict.bsh is already exactly this one
phase, and pydreg's numbers use its nsys-windowed data (see
docs/PERF_LOG.md's 2026-08-18 (cont.) entry), not its whole three-phase
run.

Reads gpu_out/summary_dreg_vs_pydreg2.json (produced by
analyze_gpu_profile.py against real profiling data) -- run that first if
this file doesn't exist. No plotting library dependency, matching the
other figures/plot_*.py scripts -- hand-built SVG.

Usage:
    python3 figures/gpu_profiling/plot_gpu_time_breakdown.py
"""

from __future__ import annotations

import html
import json
from pathlib import Path

HERE = Path(__file__).parent
PLOTS_DIR = HERE.parent / "plots"
SUMMARY_PATH = HERE / "gpu_out" / "summary_dreg_vs_pydreg2.json"

# Deliberately distinct from plot_gpu_utilization.py's dREG/pydreg tool
# colors (orange/blue) -- these encode a different dimension (time TYPE,
# not which tool), so reusing the tool colors here would make the same
# color mean two different things across the two panels. Validated via
# the dataviz skill's validate_palette.js (categorical slots 3 and 7):
# CVD/normal-vision separation both pass; aqua's contrast vs. the white
# surface WARNs below 3:1, which is why every segment gets a direct
# value label rather than relying on the fill color alone.
COLOR_IDLE = "#c7c6bd"
COLOR_KERNEL = "#1baf7a"
COLOR_MEMCPY = "#4a3aa7"

SVG_STYLE = """
<style>
text { font-family: DejaVu Sans, Helvetica, Arial, sans-serif; fill: #202124; }
.title { font-size: 26px; font-weight: 600; }
.subtitle { font-size: 17px; fill: #5f6368; }
.tick { font-size: 17px; fill: #5f6368; }
.barlabel { font-size: 17px; font-weight: 600; }
.grouplabel { font-size: 21px; font-weight: 600; }
.legend { font-size: 17px; }
.callout { font-size: 17px; fill: #202124; }
.grid { stroke: #e3e6e8; stroke-width: 1; }
</style>
""".strip()


def escape(value):
    return html.escape(str(value))


def load_group(summary_by_label, label):
    s = summary_by_label[label]
    idle_s = s["nsys_idle_gaps"]["idle_ns"] / 1e9
    kernel_s = sum(r["Total Time (ns)"] for r in s["nsys_cuda_gpu_kern_sum"]) / 1e9
    memcpy_s = sum(r["Total Time (ns)"] for r in s["nsys_cuda_gpu_mem_time_sum"]) / 1e9
    # Instance counts: the *dominant* kernel (rows are pre-sorted by total
    # time descending, so index 0) and the Host-to-Device memcpy
    # specifically -- matching exactly the numbers already vetted and
    # written into docs/PERF_LOG.md/README.md, not a sum across every
    # kernel/memcpy type (a different, larger, un-vetted statistic that
    # would silently disagree with the prose).
    kernel_n = s["nsys_cuda_gpu_kern_sum"][0]["Instances"]
    h2d_n = next(
        r["Instances"] for r in s["nsys_cuda_gpu_mem_time_sum"] if "Host-to-Device" in r["Name"]
    )
    return {
        "idle": idle_s,
        "kernel": kernel_s,
        "memcpy": memcpy_s,
        "kernel_n": kernel_n,
        "memcpy_n": h2d_n,
        "windowed": s.get("nsys_windowed"),
    }


def main():
    summary = json.loads(SUMMARY_PATH.read_text())
    by_label = {s["label"]: s for s in summary}
    dreg = load_group(by_label, "dreg")
    pydreg = load_group(by_label, "pydreg2")

    groups = [("dREG", dreg), ("pydreg", pydreg)]
    totals = {name: g["idle"] + g["kernel"] + g["memcpy"] for name, g in groups}
    max_total = max(totals.values())

    # memcpy's total *time* is a tiny sliver of either bar (0.7-3.9s out of
    # hundreds-to-thousands of seconds) -- invisible if encoded by height,
    # even though its *call count* (972x more for dREG) was one of the two
    # headline findings. Bar height stays time-based (that's the actual
    # "where does the time go" story); launch/call counts are added as
    # direct text annotations instead of forcing them into geometry they
    # can't represent at this scale.
    width, height = 720, 660
    margin_left, margin_top, margin_right, margin_bottom = 90, 90, 240, 130
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    bar_w = 140
    gap = 120

    def y_for(value):
        return margin_top + plot_h - (value / max_total) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        "<title>Where dREG's and pydreg's GPU time goes, same 5,617,218-position "
        "scoring phase</title>",
        SVG_STYLE,
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        f'<text x="{margin_left}" y="34" class="title">Where the GPU time actually goes</text>',
        f'<text x="{margin_left}" y="58" class="subtitle">Same 5,617,218-position '
        "phase, both sides phase-exact</text>",
    ]

    tick_step = 500 if max_total > 1500 else 100
    tick = 0
    while tick <= max_total:
        y = y_for(tick)
        parts.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{margin_left + plot_w}" '
            f'y2="{y:.1f}" class="grid"/>'
        )
        parts.append(
            f'<text x="{margin_left - 12}" y="{y + 6:.1f}" class="tick" '
            f'text-anchor="end">{tick:g}s</text>'
        )
        tick += tick_step

    x = margin_left + (plot_w - (2 * bar_w + gap)) / 2
    for name, g in groups:
        cum = 0.0
        for key, color, seg_label in (
            ("idle", COLOR_IDLE, "scheduling idle"),
            ("kernel", COLOR_KERNEL, "kernel compute"),
            ("memcpy", COLOR_MEMCPY, "memcpy"),
        ):
            val = g[key]
            y_bottom = y_for(cum)
            y_top = y_for(cum + val)
            seg_h = y_bottom - y_top
            parts.append(
                f'<rect x="{x:.1f}" y="{y_top:.1f}" width="{bar_w}" height="{seg_h:.1f}" '
                f'fill="{color}" stroke="#fff" stroke-width="2">'
                f"<title>{escape(name)} {escape(seg_label)}: {val:.1f}s "
                f"({100 * val / totals[name]:.1f}%)</title></rect>"
            )
            if seg_h > 18:
                parts.append(
                    f'<text x="{x + bar_w / 2:.1f}" y="{(y_top + y_bottom) / 2 + 6:.1f}" '
                    f'class="barlabel" text-anchor="middle" fill="#fff">{val:.0f}s</text>'
                )
            cum += val
        parts.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{margin_top + plot_h + 34}" '
            f'class="grouplabel" text-anchor="middle">{escape(name)}</text>'
        )
        parts.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{margin_top + plot_h + 58}" '
            f'class="tick" text-anchor="middle">{totals[name]:.0f}s total</text>'
        )
        parts.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{margin_top + plot_h + 82}" '
            f'class="tick" text-anchor="middle">{g["kernel_n"]:,} kernel launches</text>'
        )
        parts.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{margin_top + plot_h + 104}" '
            f'class="tick" text-anchor="middle">{g["memcpy_n"]:,} H2D memcpy calls</text>'
        )
        x += bar_w + gap

    legend_x = margin_left + plot_w + 30
    legend_y = margin_top + 20
    for i, (color, label) in enumerate(
        [
            (COLOR_IDLE, "Scheduling idle"),
            (COLOR_KERNEL, "Kernel compute"),
            (COLOR_MEMCPY, "Memcpy"),
        ]
    ):
        ly = legend_y + i * 32
        parts.append(f'<rect x="{legend_x}" y="{ly}" width="18" height="18" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 26}" y="{ly + 14}" class="legend">{escape(label)}</text>')

    total_ratio = totals["dREG"] / totals["pydreg"]
    busy_ratio = (dreg["kernel"] + dreg["memcpy"]) / (pydreg["kernel"] + pydreg["memcpy"])
    kern_n_ratio = dreg["kernel_n"] / pydreg["kernel_n"]
    memcpy_n_ratio = dreg["memcpy_n"] / pydreg["memcpy_n"]
    callout_y = legend_y + 3 * 32 + 24
    for i, line in enumerate(
        [
            f"Total time: {total_ratio:.1f}x",
            f"Busy-only time: {busy_ratio:.2f}x",
            f"Kernel launches: {kern_n_ratio:.0f}x",
            f"H2D memcpy calls: {memcpy_n_ratio:.0f}x",
        ]
    ):
        parts.append(f'<text x="{legend_x}" y="{callout_y + i * 24}" class="callout">{line}</text>')

    parts.append("</svg>")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / "gpu_time_breakdown.svg"
    out.write_text("\n".join(parts) + "\n")
    print(f"Wrote {out}")
    print(
        f"dREG:   idle={dreg['idle']:.1f}s kernel={dreg['kernel']:.1f}s "
        f"memcpy={dreg['memcpy']:.1f}s (nsys_windowed={dreg['windowed']})"
    )
    print(
        f"pydreg: idle={pydreg['idle']:.1f}s kernel={pydreg['kernel']:.1f}s "
        f"memcpy={pydreg['memcpy']:.1f}s (nsys_windowed={pydreg['windowed']})"
    )
    print(f"total ratio {total_ratio:.2f}x, busy-only ratio {busy_ratio:.2f}x")


if __name__ == "__main__":
    main()

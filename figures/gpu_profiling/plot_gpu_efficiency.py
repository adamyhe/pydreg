#!/usr/bin/env python3
"""Panel: per-operation efficiency -- positions processed per GPU kernel
launch and per Host-to-Device memcpy call, dREG vs pydreg, same
5,617,218-position phase. This is the direct "inefficiency" chart the
other two panels don't make explicit: plot_gpu_utilization.py shows the
GPU sitting idle (a scheduling symptom), plot_gpu_time_breakdown.py shows
totals and raw counts side by side, but neither computes the number that
actually says "each individual dREG operation does far less useful work
than pydreg's" -- this does.

A dot plot on a log-scale axis, not a bar chart: the two tools differ by
~25x (kernel) and ~972x (memcpy), and bar *length* under a log scale no
longer represents proportional magnitude the way position along a log
axis does -- the same reason figures/_common.py's scatter panels use
log-scale point marks rather than log-scale bars.

Reads gpu_out/summary_dreg_vs_pydreg2.json (produced by
analyze_gpu_profile.py against real profiling data) -- run that first if
this file doesn't exist.

Usage:
    python3 figures/gpu_profiling/plot_gpu_efficiency.py
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

HERE = Path(__file__).parent
PLOTS_DIR = HERE.parent / "plots"
SUMMARY_PATH = HERE / "gpu_out" / "summary_dreg_vs_pydreg2.json"
POSITIONS = 5_617_218  # confirmed identical both sides -- see docs/PERF_LOG.md

# Same tool-identity colors as plot_gpu_utilization.py (this panel is
# also a dREG-vs-pydreg comparison, not a time-type breakdown like
# plot_gpu_time_breakdown.py's gray/aqua/violet) -- validated together
# via the dataviz skill's validate_palette.js.
COLOR_DREG = "#eb6834"
COLOR_PYDREG = "#2a78d6"

SVG_STYLE = """
<style>
text { font-family: DejaVu Sans, Helvetica, Arial, sans-serif; fill: #202124; }
.title { font-size: 26px; font-weight: 600; }
.subtitle { font-size: 17px; fill: #5f6368; }
.rowlabel { font-size: 19px; font-weight: 600; }
.tick { font-size: 15px; fill: #5f6368; }
.value { font-size: 16px; font-weight: 600; }
.gap { font-size: 16px; fill: #5f6368; }
.legend { font-size: 17px; }
.grid { stroke: #e3e6e8; stroke-width: 1; }
.connector { stroke: #c7c6bd; stroke-width: 2; }
</style>
""".strip()


def escape(value):
    return html.escape(str(value))


def log_x(value, low, high, x0, w):
    frac = (math.log(value) - math.log(low)) / (math.log(high) - math.log(low))
    return x0 + frac * w


def load_metrics(summary):
    by_label = {s["label"]: s for s in summary}
    out = {}
    for label in ("dreg", "pydreg2"):
        s = by_label[label]
        kern = s["nsys_cuda_gpu_kern_sum"][0]
        h2d = next(r for r in s["nsys_cuda_gpu_mem_time_sum"] if "Host-to-Device" in r["Name"])
        out[label] = {
            "kernel_pos_per_op": POSITIONS / kern["Instances"],
            "kernel_ns_per_op": kern["Avg (ns)"],
            "memcpy_pos_per_op": POSITIONS / h2d["Instances"],
            "memcpy_ns_per_op": h2d["Avg (ns)"],
        }
    return out


def main():
    summary = json.loads(SUMMARY_PATH.read_text())
    m = load_metrics(summary)
    dreg, pydreg = m["dreg"], m["pydreg2"]

    rows = [
        ("Kernel launch", "kernel_pos_per_op", "kernel_ns_per_op", "positions/launch"),
        ("H2D memcpy call", "memcpy_pos_per_op", "memcpy_ns_per_op", "positions/call"),
    ]

    width, height = 720, 380
    margin_left, margin_top, margin_right, margin_bottom = 210, 110, 60, 60
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    row_h = plot_h / len(rows)

    low, high = 1, 10000
    ticks = [1, 10, 100, 1000, 10000]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        "<title>Positions processed per GPU operation: dREG vs pydreg, log scale</title>",
        SVG_STYLE,
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        f'<text x="{margin_left}" y="34" class="title">Work done per GPU operation</text>',
        f'<text x="{margin_left}" y="58" class="subtitle">Same 5,617,218-position phase '
        "-- log scale, higher = more efficient</text>",
    ]

    for tick in ticks:
        x = log_x(tick, low, high, margin_left, plot_w)
        parts.append(
            f'<line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{margin_top + plot_h}" '
            'class="grid"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{margin_top + plot_h + 24}" class="tick" '
            f'text-anchor="middle">{tick:g}</text>'
        )

    for i, (row_label, pos_key, ns_key, unit) in enumerate(rows):
        y = margin_top + (i + 0.5) * row_h
        parts.append(f'<text x="{margin_left - 20}" y="{y - 10:.1f}" class="rowlabel" text-anchor="end">{escape(row_label)}</text>')

        x_dreg = log_x(dreg[pos_key], low, high, margin_left, plot_w)
        x_pydreg = log_x(pydreg[pos_key], low, high, margin_left, plot_w)
        parts.append(f'<line x1="{x_dreg:.1f}" y1="{y:.1f}" x2="{x_pydreg:.1f}" y2="{y:.1f}" class="connector"/>')

        for x, val, color, name, ns_val in (
            (x_dreg, dreg[pos_key], COLOR_DREG, "dREG", dreg[ns_key]),
            (x_pydreg, pydreg[pos_key], COLOR_PYDREG, "pydreg", pydreg[ns_key]),
        ):
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="{color}" stroke="#fff" stroke-width="2">'
                f"<title>{escape(name)}: {val:.1f} {unit}, {ns_val / 1e3:.2f}us avg per op</title></circle>"
            )
            label_above = val < math.sqrt(dreg[pos_key] * pydreg[pos_key])
            ly = y - 16 if label_above else y + 26
            parts.append(
                f'<text x="{x:.1f}" y="{ly:.1f}" class="value" text-anchor="middle" fill="{color}">'
                f"{val:,.1f}</text>"
            )

        gap = pydreg[pos_key] / dreg[pos_key]
        mid_x = (x_dreg + x_pydreg) / 2
        parts.append(
            f'<text x="{mid_x:.1f}" y="{y - 30:.1f}" class="gap" text-anchor="middle">{gap:,.0f}x</text>'
        )

    parts.append(
        f'<text x="{margin_left + plot_w / 2}" y="{height - 8}" class="tick" text-anchor="middle">'
        "Positions processed per operation (log scale)</text>"
    )

    legend_x = margin_left
    legend_y = margin_top - 34
    for i, (color, label) in enumerate([(COLOR_DREG, "dREG"), (COLOR_PYDREG, "pydreg")]):
        lx = legend_x + i * 100
        parts.append(f'<circle cx="{lx}" cy="{legend_y}" r="7" fill="{color}"/>')
        parts.append(f'<text x="{lx + 14}" y="{legend_y + 5}" class="legend">{escape(label)}</text>')

    parts.append("</svg>")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / "gpu_efficiency.svg"
    out.write_text("\n".join(parts) + "\n")
    print(f"Wrote {out}")
    for row_label, pos_key, ns_key, unit in rows:
        print(
            f"{row_label}: dREG {dreg[pos_key]:.2f} {unit} ({dreg[ns_key]/1e3:.2f}us/op), "
            f"pydreg {pydreg[pos_key]:.2f} {unit} ({pydreg[ns_key]/1e3:.2f}us/op), "
            f"gap {pydreg[pos_key]/dreg[pos_key]:.1f}x"
        )


if __name__ == "__main__":
    main()

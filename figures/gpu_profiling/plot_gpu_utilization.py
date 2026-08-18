#!/usr/bin/env python3
"""Panel: GPU utilization (Volatile GPU-Util / dmon's `sm` column) over
time, dREG vs pydreg, both restricted to the identical informative-
positions scoring phase. The qualitative counterpart to
plot_gpu_time_breakdown.py's quantitative one -- this is what "dREG has
regular spikes up and down, pydreg is a steady plateau" actually looks
like, real data, not a description of it.

Each sub-panel uses its OWN x-axis (seconds since that phase started, not
a shared/normalized scale) so the sawtooth-vs-plateau *shape* is legible
in both -- dREG's phase is ~6.5x longer in wall-clock terms, and cramming
both onto one shared axis would compress pydreg's trace into a sliver a
few pixels wide. Each sub-panel's title states its actual duration, so
the magnitude difference isn't lost, just not encoded in x-position.

Reads gpu_out/{dreg,pydreg2}.dmon.csv + .log directly (not the analyzer's
JSON summary, which only has aggregate stats, not the full time series)
via analyze_gpu_profile.py's own parsing functions -- run profile_gpu.sh
first if these don't exist.

Usage:
    python3 figures/gpu_profiling/plot_gpu_utilization.py
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

HERE = Path(__file__).parent
PLOTS_DIR = HERE.parent / "plots"
GPU_OUT = HERE / "gpu_out"

sys.path.insert(0, str(HERE))
from analyze_gpu_profile import (  # noqa: E402
    DEFAULT_PHASE_END_RE,
    DEFAULT_PHASE_START_RE,
    parse_dmon,
    parse_log_window,
    select_active_gpu,
    window_df,
)

# Matches the dREG/pydreg identity coloring used consistently across this
# investigation's figures (kept distinct from plot_gpu_time_breakdown.py's
# idle/kernel/memcpy colors, which encode a different dimension). Both
# pairs validated together via the dataviz skill's validate_palette.js
# (categorical slots 1 and 2): CVD ΔE 24.7, normal-vision ΔE 33.6, both
# well clear of the pass floors.
COLOR_DREG = "#eb6834"
COLOR_PYDREG = "#2a78d6"

SVG_STYLE = """
<style>
text { font-family: DejaVu Sans, Helvetica, Arial, sans-serif; fill: #202124; }
.title { font-size: 26px; font-weight: 600; }
.subtitle { font-size: 17px; fill: #5f6368; }
.paneltitle { font-size: 19px; font-weight: 600; }
.panelstat { font-size: 16px; fill: #5f6368; }
.tick { font-size: 15px; fill: #5f6368; }
.axis { font-size: 16px; fill: #5f6368; }
.grid { stroke: #e3e6e8; stroke-width: 1; }
.plotbg { fill: #fbfbfa; stroke: #e3e6e8; }
</style>
""".strip()


def escape(value):
    return html.escape(str(value))


def load_series(dmon_path, gpu_index, log_path=None, whole_trace=False):
    df = select_active_gpu(parse_dmon(dmon_path), dmon_path.stem, gpu_index)
    window = None if whole_trace else parse_log_window(log_path, DEFAULT_PHASE_START_RE, DEFAULT_PHASE_END_RE)
    df = window_df(df, window).sort_values("timestamp")
    t0 = df["timestamp"].min()
    seconds = (df["timestamp"] - t0).dt.total_seconds().tolist()
    sm = df["sm"].tolist()
    return seconds, sm


def subpanel_svg(parts, x0, y0, w, h, seconds, sm, color, title, gpu_index):
    """Draws one utilization-over-time sub-panel into `parts` (a plot
    background, gridlines at 0/25/50/75/100%, the sm% line + filled area,
    sparse hover points, axes, and a title/stat line) at the given screen
    rectangle. Returns nothing -- appends SVG fragments in place."""
    duration = seconds[-1] if seconds else 0
    mean_sm = sum(sm) / len(sm) if sm else 0

    parts.append(
        f'<text x="{x0}" y="{y0 - 12}" class="paneltitle">{escape(title)} '
        f"(GPU {gpu_index}, {duration:.0f}s total, mean {mean_sm:.0f}% util)</text>"
    )
    parts.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" class="plotbg"/>')

    def xf(t):
        return x0 + (t / duration) * w if duration else x0

    def yf(pct):
        return y0 + h - (pct / 100) * h

    for pct in (0, 25, 50, 75, 100):
        y = yf(pct)
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + w}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{x0 - 8}" y="{y + 5:.1f}" class="tick" text-anchor="end">{pct}%</text>')

    tick_step = 500 if duration > 1500 else (60 if duration > 200 else 30)
    t = 0
    while t <= duration:
        x = xf(t)
        parts.append(f'<text x="{x:.1f}" y="{y0 + h + 20}" class="tick" text-anchor="middle">{t:g}s</text>')
        t += tick_step

    if seconds:
        line_pts = " ".join(f"{xf(t):.1f},{yf(v):.1f}" for t, v in zip(seconds, sm))
        area_pts = f"{xf(seconds[0]):.1f},{yf(0):.1f} {line_pts} {xf(seconds[-1]):.1f},{yf(0):.1f}"
        parts.append(f'<polygon points="{area_pts}" fill="{color}" fill-opacity="0.15" stroke="none"/>')
        parts.append(f'<polyline points="{line_pts}" fill="none" stroke="{color}" stroke-width="1.6"/>')

        # Sparse hover points (~40 across the trace) for basic
        # inspectability, matching the <title>-tooltip convention the
        # other figures/plot_*.py scatter panels already use.
        stride = max(1, len(seconds) // 40)
        for t, v in list(zip(seconds, sm))[::stride]:
            parts.append(
                f'<circle cx="{xf(t):.1f}" cy="{yf(v):.1f}" r="6" fill="{color}" fill-opacity="0">'
                f"<title>t={t:.0f}s, {v:.0f}% util</title></circle>"
            )

    parts.append(
        f'<text x="{x0 + w / 2}" y="{y0 + h + 42}" class="axis" text-anchor="middle">'
        "Seconds since phase started</text>"
    )


def main():
    dreg_seconds, dreg_sm = load_series(
        GPU_OUT / "dreg.dmon.csv", gpu_index=1, whole_trace=True
    )
    pydreg_seconds, pydreg_sm = load_series(
        GPU_OUT / "pydreg2.dmon.csv", gpu_index=1, log_path=GPU_OUT / "pydreg2.log"
    )

    width = 780
    margin_left, margin_right = 70, 30
    panel_w = width - margin_left - margin_right
    panel_h = 230
    top_margin = 90
    between = 90
    bottom_margin = 95
    height = top_margin + panel_h + between + panel_h + bottom_margin

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        "<title>GPU utilization over time: dREG vs pydreg, same informative-positions "
        "scoring phase</title>",
        SVG_STYLE,
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        f'<text x="{margin_left}" y="34" class="title">GPU utilization over time</text>',
        f'<text x="{margin_left}" y="58" class="subtitle">Same 5,617,218-position phase '
        "-- own x-axis per panel, see title for duration</text>",
    ]

    y_dreg = top_margin
    subpanel_svg(parts, margin_left, y_dreg, panel_w, panel_h, dreg_seconds, dreg_sm, COLOR_DREG, "dREG", 1)

    y_pydreg = y_dreg + panel_h + between
    subpanel_svg(
        parts, margin_left, y_pydreg, panel_w, panel_h, pydreg_seconds, pydreg_sm, COLOR_PYDREG, "pydreg", 1
    )

    speedup = dreg_seconds[-1] / pydreg_seconds[-1] if pydreg_seconds and pydreg_seconds[-1] else None
    if speedup:
        parts.append(
            f'<text x="{margin_left}" y="{height - 18}" class="panelstat">'
            f"dREG took {speedup:.1f}x as long as pydreg (note the different x-axis scales)</text>"
        )

    parts.append("</svg>")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / "gpu_utilization.svg"
    out.write_text("\n".join(parts) + "\n")
    print(f"Wrote {out}")
    print(f"dREG:   {len(dreg_seconds)} samples, {dreg_seconds[-1]:.0f}s, mean sm {sum(dreg_sm)/len(dreg_sm):.1f}%")
    print(
        f"pydreg: {len(pydreg_seconds)} samples, {pydreg_seconds[-1]:.0f}s, "
        f"mean sm {sum(pydreg_sm)/len(pydreg_sm):.1f}%"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Panel: GPU utilization (Volatile GPU-Util / dmon's `sm` column) over
time, dREG vs pydreg, one separate track per library across the full
12-library benchmark. The qualitative counterpart to
plot_gpu_time_breakdown.py's quantitative one -- this is what "dREG has
regular spikes up and down, pydreg is a steady plateau" actually looks
like, real data, not a description of it, and repeated across every
library so the pattern reads as a property of the two implementations
rather than of one capture.

Every track keeps its OWN x-axis (seconds since that run's scoring phase
started, not a shared or normalized scale). dREG's phase runs several
times longer than pydreg's on the same library, and libraries differ from
each other again on top of that; forcing all 24 tracks onto one shared
axis would compress the short ones into slivers a few pixels wide and
destroy the sawtooth-vs-plateau *shape*, which is the entire point of this
figure. Each track states its own duration and mean utilization instead,
so the magnitude difference is reported rather than encoded in x-position
-- plot_gpu_time_breakdown.py is where durations are compared to scale.

Reads gpu_out/<label>.dmon.csv + .log directly (not the analyzer's JSON
summary, which keeps only aggregate stats, not the full time series) via
analyze_gpu_profile.py's own parsing functions.

Usage:
    python3 figures/gpu_profiling/plot_gpu_utilization.py
    python3 figures/gpu_profiling/plot_gpu_utilization.py \
        --gpu-index dreg_G1=1,pydreg_G1=0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _gpu_common import (  # noqa: E402
    COLOR_DREG,
    COLOR_PYDREG,
    SVG_STYLE,
    add_common_args,
    escape,
    load_series,
    resolve_pairs,
    write_svg,
)
from analyze_gpu_profile import parse_gpu_index  # noqa: E402

ROW_H = 74
ROW_GAP = 26
COL_GAP = 54


def track_svg(parts, x0, y0, w, h, seconds, sm, color):
    """Draws one run's utilization-over-time track into `parts` at the
    given screen rectangle: background, 0/50/100% gridlines, the sm% line
    plus filled area, and sparse hover points. Returns nothing."""
    duration = seconds[-1] if seconds else 0
    parts.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" class="plotbg"/>')

    def xf(t):
        return x0 + (t / duration) * w if duration else x0

    def yf(pct):
        return y0 + h - (pct / 100) * h

    for pct in (0, 50, 100):
        y = yf(pct)
        parts.append(
            f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + w}" y2="{y:.1f}" class="grid"/>'
        )

    if not seconds:
        return

    line_pts = " ".join(f"{xf(t):.1f},{yf(v):.1f}" for t, v in zip(seconds, sm))
    area_pts = (
        f"{xf(seconds[0]):.1f},{yf(0):.1f} {line_pts} {xf(seconds[-1]):.1f},{yf(0):.1f}"
    )
    parts.append(
        f'<polygon points="{area_pts}" fill="{color}" fill-opacity="0.15" stroke="none"/>'
    )
    parts.append(
        f'<polyline points="{line_pts}" fill="none" stroke="{color}" stroke-width="1.4"/>'
    )

    # Sparse hover points (~25 per track) for basic inspectability,
    # matching the <title>-tooltip convention the other figures/plot_*.py
    # panels already use. Fewer per track than the old single-library
    # version used, since there are now 24 tracks rather than 2.
    stride = max(1, len(seconds) // 25)
    for t, v in list(zip(seconds, sm))[::stride]:
        parts.append(
            f'<circle cx="{xf(t):.1f}" cy="{yf(v):.1f}" r="5" fill="{color}" '
            f'fill-opacity="0"><title>t={t:.0f}s, {v:.0f}% util</title></circle>'
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument(
        "--gpu-index",
        default=None,
        help="which nvidia-smi GPU index each label's dmon data is on -- a "
        "bare int for every label, or comma-separated label=index pairs. "
        "Same format and the same hard-fail-on-typo validation as "
        "analyze_gpu_profile.py's; see README.md on why CUDA's device "
        "index is not nvidia-smi's.",
    )
    args = parser.parse_args()

    pairs = resolve_pairs(args, require="dmon")
    all_labels = [lb for p in pairs for lb in p.labels]
    gpu_by_label, gpu_default = parse_gpu_index(args.gpu_index, all_labels)

    rows = []
    for pair in pairs:
        # dREG's run_predict.bsh has no separate phases to slice out -- the
        # whole process already is the informative-positions scoring phase
        # (README.md) -- so it's never windowed; pydreg's full pipeline is.
        dreg = load_series(
            pair, pair.dreg_label, gpu_by_label.get(pair.dreg_label, gpu_default), True
        )
        pydreg = load_series(
            pair,
            pair.pydreg_label,
            gpu_by_label.get(pair.pydreg_label, gpu_default),
            False,
        )
        rows.append((pair.library, dreg, pydreg))

    width = 1020
    margin_left, margin_right = 168, 30
    top_margin, bottom_margin = 122, 60
    panel_w = (width - margin_left - margin_right - COL_GAP) / 2
    height = top_margin + len(rows) * (ROW_H + ROW_GAP) + bottom_margin

    x_dreg = margin_left
    x_pydreg = margin_left + panel_w + COL_GAP

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        "<title>GPU utilization over time: dREG vs pydreg, separate track per "
        "library, informative-positions scoring phase</title>",
        SVG_STYLE,
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        f'<text x="{margin_left}" y="34" class="title">GPU utilization over time, '
        "per library</text>",
        f'<text x="{margin_left}" y="58" class="subtitle">Informative-positions '
        f"scoring phase, {len(rows)} librar"
        f'{"y" if len(rows) == 1 else "ies"} -- every track has its own x-axis; '
        "see each track's stated duration</text>",
        f'<text x="{x_dreg}" y="{top_margin - 16}" class="colhead" fill="{COLOR_DREG}">'
        "dREG</text>",
        f'<text x="{x_pydreg}" y="{top_margin - 16}" class="colhead" '
        f'fill="{COLOR_PYDREG}">pydreg</text>',
    ]

    for i, (library, dreg, pydreg) in enumerate(rows):
        y = top_margin + i * (ROW_H + ROW_GAP)
        parts.append(
            f'<text x="{margin_left - 16}" y="{y + ROW_H / 2 + 5:.1f}" class="rowlabel" '
            f'text-anchor="end">{escape(library)}</text>'
        )
        # y-axis ticks on the leftmost track of each row only -- the scale
        # is identical in all tracks, so repeating it in each is noise.
        for pct, dy in ((100, 10), (0, ROW_H)):
            parts.append(
                f'<text x="{margin_left - 16}" y="{y + dy}" class="tick" '
                f'text-anchor="end">{pct}%</text>'
            )

        for x0, (seconds, sm), color in (
            (x_dreg, dreg, COLOR_DREG),
            (x_pydreg, pydreg, COLOR_PYDREG),
        ):
            track_svg(parts, x0, y, panel_w, ROW_H, seconds, sm, color)
            duration = seconds[-1] if seconds else 0
            mean_sm = sum(sm) / len(sm) if sm else 0
            parts.append(
                f'<text x="{x0 + panel_w:.1f}" y="{y + ROW_H + 16}" class="panelstat" '
                f'text-anchor="end">{duration:,.0f}s, mean {mean_sm:.0f}% util</text>'
            )
            parts.append(
                f'<text x="{x0}" y="{y + ROW_H + 16}" class="panelstat">0s</text>'
            )

    parts.append(
        f'<text x="{margin_left + (width - margin_left - margin_right) / 2:.1f}" '
        f'y="{height - 20}" class="axis" text-anchor="middle">'
        "Seconds since that run's scoring phase started (independent scale per track)"
        "</text>"
    )
    parts.append("</svg>")

    write_svg(parts, "gpu_utilization.svg")
    for library, (ds, dsm), (ps, psm) in rows:
        ratio = ds[-1] / ps[-1] if ps and ps[-1] else float("nan")
        print(
            f"{library}: dREG {ds[-1]:,.0f}s mean {sum(dsm)/len(dsm):.1f}% | "
            f"pydreg {ps[-1]:,.0f}s mean {sum(psm)/len(psm):.1f}% | {ratio:.1f}x longer"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate a compact, conceptual schematic of the pydreg/dREG workflow.

The figure is data-free -- it illustrates what each pipeline stage
transforms (bigWig -> informative positions -> features -> SVR score ->
RF-assisted peak calling -> FDR-filtered peaks), not a real genomic run.
Pure stdlib (hand-built SVG), matching this repo's existing
figures/legacy/plot_timing_comparison.py convention of not pulling in a
plotting library for a one-off figure.

Usage:
    python3 figures/plot_pipeline_schematic.py
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

FIGURE_WIDTH_PT = 640.0
MARGIN = 10.0
TITLE_HEIGHT = 20.0
STRIP_HEIGHT = 80.0

DARK = "#222222"
BLUE = "#0057b8"
RED = "#d62728"
GREEN = "#009e73"
ORANGE = "#e69f00"
LIGHT_BLUE = "#dce9f8"
LIGHT_GREEN = "#ddf1e9"
LIGHT_ORANGE = "#f7ecd3"
LIGHT_RED = "#f8dedb"
LIGHT_GRAY = "#ededed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "plots")
    parser.add_argument("--formats", nargs="+", default=["svg", "pdf"], choices=["svg", "pdf"])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    svg = build_figure()
    svg_path = args.output_dir / "pipeline_schematic.svg"
    svg_path.write_text(svg)
    print(f"Wrote {svg_path}")
    if "pdf" in args.formats:
        write_pdf(svg_path, args.output_dir / "pipeline_schematic.pdf")


def build_figure() -> str:
    width = FIGURE_WIDTH_PT
    height = TITLE_HEIGHT + STRIP_HEIGHT + 2 * MARGIN
    parts = [
        svg_header(width, height),
        style_block(),
        title(width),
        overview_strip(MARGIN, MARGIN + TITLE_HEIGHT, width - 2 * MARGIN, STRIP_HEIGHT),
        "</svg>\n",
    ]
    return "\n".join(parts)


def svg_header(width: float, height: float) -> str:
    return (
        '<?xml version="1.0" encoding="utf-8" standalone="no"?>\n'
        f'<svg width="{width:g}pt" height="{height:g}pt" viewBox="0 0 {width:g} {height:g}" '
        'xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1">'
    )


def style_block() -> str:
    return """
<defs>
  <style type="text/css">
    text { font-family: DejaVu Sans, Arial, sans-serif; fill: #222222; }
    .fig-title { font-size: 12px; font-weight: 700; }
    .step-title { font-size: 10px; font-weight: 700; }
    .step-subtitle { font-size: 8.6px; }
    .step-box { stroke: #222222; stroke-width: 1; rx: 5; ry: 5; }
    .mini-line { stroke: #222222; stroke-width: 1.1; fill: none; stroke-linecap: round; }
    .mini-thin { stroke: #222222; stroke-width: 0.7; fill: none; }
    .mini-dashed { stroke: #222222; stroke-width: 0.8; fill: none; stroke-dasharray: 2.5,2; }
    .mini-red { stroke: #d62728; stroke-width: 0.9; fill: none; }
    .step-arrow { stroke: #222222; stroke-width: 1.2; fill: none; marker-end: url(#arrowhead); }
  </style>
  <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto" markerUnits="strokeWidth">
    <path d="M 0 0 L 8 3 L 0 6 z" fill="#222222"/>
  </marker>
</defs>
"""


def title(width: float) -> str:
    return (
        f'<text class="fig-title" x="{width / 2:g}" y="{MARGIN + 10:g}" text-anchor="middle">'
        "pydreg / dREG workflow: PRO-seq/GRO-seq bigWig → called regulatory elements</text>"
    )


def overview_strip(x: float, y: float, width: float, height: float) -> str:
    steps = [
        ("Step 1", "informative positions", LIGHT_BLUE, mini_infp),
        ("Step 2", "multi-scale features", LIGHT_GREEN, mini_features),
        ("Step 3", "SVR scoring", LIGHT_ORANGE, mini_svr),
        ("Step 4", "RF peak calling", LIGHT_RED, mini_rf_split),
        ("Step 5", "FDR filter + output", LIGHT_GRAY, mini_output),
    ]
    box_y = y + 3
    box_h = height - 6
    gap = 22.0
    box_w = (width - gap * (len(steps) - 1)) / len(steps)
    parts = []
    for i, (title_text, subtitle, fill, draw_mini) in enumerate(steps):
        bx = x + i * (box_w + gap)
        parts.append(
            f'<rect class="step-box" x="{bx:g}" y="{box_y:g}" width="{box_w:g}" height="{box_h:g}" fill="{fill}"/>'
        )
        parts.append(
            f'<text class="step-title" x="{bx + box_w / 2:g}" y="{box_y + 12:g}" text-anchor="middle">{escape(title_text)}</text>'
        )
        parts.append(
            f'<text class="step-subtitle" x="{bx + box_w / 2:g}" y="{box_y + box_h - 5:g}" text-anchor="middle">{escape(subtitle)}</text>'
        )
        parts.append(draw_mini(bx + 8, box_y + 15, box_w - 16, box_h - 29))
        if i < len(steps) - 1:
            y0 = box_y + box_h / 2
            parts.append(
                f'<path class="step-arrow" d="M {bx + box_w + 4:g} {y0:g} L {bx + box_w + gap - 4:g} {y0:g}"/>'
            )
    return "\n".join(parts)


def mini_infp(x: float, y: float, width: float, height: float) -> str:
    """A sparse plus/minus strand read pileup scanned window by window: a
    window is kept (green, marked with the informative position it yields)
    only if it clears the read-depth filter, and dropped (gray, hatched --
    the same kept/dropped convention as mini_output) otherwise, the same
    way get_informative_positions actually decides."""
    plus_y = y + height * 0.20
    minus_y = y + height * 0.76
    tick = height * 0.14
    windows = [
        (0.03, 0.28, [0.10, 0.19], [0.14]),
        (0.37, 0.62, [], []),
        (0.71, 0.96, [0.80], [0.85, 0.91]),
    ]
    parts = [
        f'<path class="mini-line" d="M {x:g} {plus_y:g} L {x + width:g} {plus_y:g}"/>',
        f'<path class="mini-line" d="M {x:g} {minus_y:g} L {x + width:g} {minus_y:g}"/>',
    ]
    band_y0 = plus_y - tick * 1.3
    band_h = (minus_y + tick * 1.3) - band_y0
    for start, end, plus_reads, minus_reads in windows:
        bx = x + width * start
        bw = width * (end - start)
        kept = bool(plus_reads) and bool(minus_reads)
        fill = GREEN if kept else LIGHT_GRAY
        opacity = "0.22" if kept else "0.55"
        parts.append(
            f'<rect x="{bx:g}" y="{band_y0:g}" width="{bw:g}" height="{band_h:g}" fill="{fill}" '
            f'opacity="{opacity}" stroke="{GREEN if kept else DARK}" stroke-width="0.8" '
            f'stroke-dasharray="{"none" if kept else "2,1.5"}"/>'
        )
        if not kept:
            parts.append(
                f'<path class="mini-thin" d="M {bx:g} {band_y0:g} L {bx + bw:g} {band_y0 + band_h:g} '
                f'M {bx:g} {band_y0 + band_h:g} L {bx + bw:g} {band_y0:g}"/>'
            )
        for frac in plus_reads:
            rx = x + width * frac
            parts.append(f'<path class="mini-thin" d="M {rx:g} {plus_y - tick:g} L {rx:g} {plus_y:g}"/>')
        for frac in minus_reads:
            rx = x + width * frac
            parts.append(f'<path class="mini-thin" d="M {rx:g} {minus_y:g} L {rx:g} {minus_y + tick:g}"/>')
        if kept:
            parts.append(
                f'<circle cx="{bx + bw / 2:g}" cy="{(plus_y + minus_y) / 2:g}" r="{height * 0.06:g}" fill="{RED}"/>'
            )
    return "\n".join(parts)


def draw_feature_vector(x: float, y: float, width: float, height: float) -> str:
    """The multi-scale feature vector: three window-scale groups (colored
    to match mini_features' nested-window ticks), each with a plus-strand
    and minus-strand bin. Drawn identically wherever the vector appears
    (mini_features' output, mini_svr's input) since it's the same object;
    all bars grow upward from one shared baseline -- feature values are
    non-negative read counts, so unlike mini_infp's plus/minus tracks there
    is no below-axis half to mirror them into."""
    colors = [BLUE, GREEN, ORANGE]
    plus_hs = [0.55, 0.85, 0.65]
    minus_hs = [0.40, 0.70, 0.50]
    group_w = width / 3
    bar_w = group_w * 0.36
    bar_gap = bar_w * 0.3
    max_bar = height * 0.72
    baseline = y + height * 0.5 + max_bar / 2
    parts = [f'<path class="mini-thin" d="M {x:g} {baseline:g} L {x + width:g} {baseline:g}"/>']
    for i, color in enumerate(colors):
        gx = x + i * group_w + (group_w - (2 * bar_w + bar_gap)) / 2
        ph = max_bar * plus_hs[i]
        mh = max_bar * minus_hs[i]
        parts.append(f'<rect x="{gx:g}" y="{baseline - ph:g}" width="{bar_w:g}" height="{ph:g}" fill="{color}"/>')
        parts.append(
            f'<rect x="{gx + bar_w + bar_gap:g}" y="{baseline - mh:g}" width="{bar_w:g}" height="{mh:g}" '
            f'fill="{color}" opacity="0.55"/>'
        )
    return "\n".join(parts)


def mini_features(x: float, y: float, width: float, height: float) -> str:
    """Nested windows ("zoom levels") around a candidate position, colored
    to match the plus/minus-strand bin pair each contributes to the
    fixed-length multi-scale feature vector on the right."""
    mid_y = y + height * 0.5
    left_w = width * 0.40
    gap = width * 0.20
    right_x = x + left_w + gap
    right_w = width - left_w - gap
    center_x = x + left_w / 2

    colors = [BLUE, GREEN, ORANGE]
    half_fracs = [0.30, 0.55, 0.80]
    box_h = height * 0.78
    parts = [
        f'<rect x="{x:g}" y="{mid_y - box_h / 2:g}" width="{left_w:g}" height="{box_h:g}" '
        f'fill="{LIGHT_GRAY}" opacity="0.4" stroke="{DARK}" stroke-width="0.6" stroke-dasharray="2,1.5"/>',
        f'<path class="mini-thin" d="M {x:g} {mid_y:g} L {x + left_w:g} {mid_y:g}"/>',
        f'<circle cx="{center_x:g}" cy="{mid_y:g}" r="{height * 0.05:g}" fill="{RED}"/>',
    ]
    tick_h = height * 0.30
    for i, (frac, color) in enumerate(zip(half_fracs, colors)):
        half_w = (left_w / 2) * frac
        th = tick_h * (0.55 + 0.22 * i)
        for sign in (-1, 1):
            tx = center_x + sign * half_w
            parts.append(
                f'<path d="M {tx:g} {mid_y - th / 2:g} L {tx:g} {mid_y + th / 2:g}" '
                f'stroke="{color}" stroke-width="1.4" fill="none"/>'
            )

    parts.append(f'<path class="step-arrow" d="M {x + left_w + 3:g} {mid_y:g} L {right_x - 3:g} {mid_y:g}"/>')
    parts.append(draw_feature_vector(right_x, y, right_w, height))
    return "\n".join(parts)


def mini_svr(x: float, y: float, width: float, height: float) -> str:
    """Feature vector in, smoothed genome-wide dREG score curve out, with
    the significance threshold used downstream for peak calling."""
    mid_y = y + height * 0.5
    vec_x = x
    vec_w = width * 0.40
    parts = [draw_feature_vector(vec_x, y, vec_w, height)]

    plot_x = x + width * 0.58
    plot_w = x + width * 0.98 - plot_x
    plot_y = y + height * 0.15
    plot_h = height * 0.70

    parts.append(f'<path class="step-arrow" d="M {vec_x + vec_w + 3:g} {mid_y:g} L {plot_x - 3:g} {mid_y:g}"/>')
    parts.append(
        f'<rect x="{plot_x:g}" y="{plot_y:g}" width="{plot_w:g}" height="{plot_h:g}" '
        f'fill="#ffffff" opacity="0.6" stroke="{DARK}" stroke-width="0.6"/>'
    )
    pts = [(0.0, 0.55), (0.20, 0.62), (0.40, 0.50), (0.55, 0.15), (0.70, 0.55), (0.85, 0.45), (1.0, 0.50)]
    d = " ".join(
        ("M" if i == 0 else "L") + f" {plot_x + plot_w * px:g} {plot_y + plot_h * py:g}"
        for i, (px, py) in enumerate(pts)
    )
    parts.append(f'<path d="{d}" fill="none" stroke="{BLUE}" stroke-width="1.3"/>')
    thresh_y = plot_y + plot_h * 0.28
    parts.append(f'<path class="mini-dashed" d="M {plot_x:g} {thresh_y:g} L {plot_x + plot_w:g} {thresh_y:g}"/>')
    peak_x = plot_x + plot_w * 0.55
    peak_y = plot_y + plot_h * 0.10
    parts.append(f'<circle cx="{peak_x:g}" cy="{peak_y:g}" r="{height * 0.06:g}" fill="{RED}"/>')
    return "\n".join(parts)


def mini_rf_split(x: float, y: float, width: float, height: float) -> str:
    """One broad candidate peak refined into narrower peaks by the
    random-forest boundary splitter."""
    mid_y = y + height * 0.5
    box_h = height * 0.55
    before_w = width * 0.32
    before_x = x
    before_y = mid_y - box_h / 2
    parts = [
        f'<rect x="{before_x:g}" y="{before_y:g}" width="{before_w:g}" height="{box_h:g}" '
        f'fill="{LIGHT_ORANGE}" stroke="{DARK}" stroke-width="0.9"/>',
        f'<text class="step-subtitle" x="{before_x + before_w / 2:g}" y="{before_y + box_h + height * 0.18:g}" text-anchor="middle">broad</text>',
    ]
    parts.append(
        f'<path class="step-arrow" d="M {before_x + before_w + 4:g} {mid_y:g} '
        f'L {before_x + before_w + width * 0.19:g} {mid_y:g}"/>'
    )
    after_x = before_x + before_w + width * 0.22
    after_w = width - (after_x - x)
    split = after_x + after_w * 0.42
    parts.append(
        f'<rect x="{after_x:g}" y="{before_y:g}" width="{split - after_x:g}" height="{box_h:g}" '
        f'fill="{LIGHT_GREEN}" stroke="{DARK}" stroke-width="0.9"/>'
    )
    parts.append(
        f'<rect x="{split:g}" y="{before_y:g}" width="{after_x + after_w - split:g}" height="{box_h:g}" '
        f'fill="{LIGHT_BLUE}" stroke="{DARK}" stroke-width="0.9"/>'
    )
    parts.append(f'<path class="mini-red" d="M {split:g} {before_y - 3:g} L {split:g} {before_y + box_h + 3:g}"/>')
    parts.append(
        f'<text class="step-subtitle" x="{after_x + after_w / 2:g}" y="{before_y + box_h + height * 0.18:g}" text-anchor="middle">split</text>'
    )
    return "\n".join(parts)


def mini_output(x: float, y: float, width: float, height: float) -> str:
    """Scored peaks filtered by FDR threshold, written out as BED/bigWig."""
    mid_y = y + height * 0.5
    axis_x0 = x
    axis_x1 = x + width * 0.56
    block_h = height * 0.30
    parts = [f'<path class="mini-line" d="M {axis_x0:g} {mid_y:g} L {axis_x1:g} {mid_y:g}"/>']
    blocks = [(0.02, 0.16, True), (0.24, 0.34, False), (0.42, 0.54, True)]
    for start, end, kept in blocks:
        bx = axis_x0 + (axis_x1 - axis_x0) * start
        bw = (axis_x1 - axis_x0) * (end - start)
        fill = GREEN if kept else LIGHT_GRAY
        opacity = "1" if kept else "0.6"
        parts.append(
            f'<rect x="{bx:g}" y="{mid_y - block_h / 2:g}" width="{bw:g}" height="{block_h:g}" fill="{fill}" '
            f'opacity="{opacity}" stroke="{DARK}" stroke-width="0.6"/>'
        )
        if not kept:
            parts.append(
                f'<path class="mini-thin" d="M {bx:g} {mid_y - block_h / 2:g} L {bx + bw:g} {mid_y + block_h / 2:g} '
                f'M {bx:g} {mid_y + block_h / 2:g} L {bx + bw:g} {mid_y - block_h / 2:g}"/>'
            )
    parts.append(f'<path class="step-arrow" d="M {axis_x1 + 3:g} {mid_y:g} L {x + width * 0.76:g} {mid_y:g}"/>')
    doc_x = x + width * 0.80
    doc_w = width * 0.18
    doc_h = height * 0.85
    doc_y = mid_y - doc_h / 2
    parts.append(
        f'<rect x="{doc_x:g}" y="{doc_y:g}" width="{doc_w:g}" height="{doc_h:g}" '
        f'fill="#ffffff" stroke="{DARK}" stroke-width="0.9"/>'
    )
    for i in range(3):
        ly = doc_y + doc_h * (0.3 + 0.25 * i)
        parts.append(f'<path class="mini-thin" d="M {doc_x + 2:g} {ly:g} L {doc_x + doc_w - 2:g} {ly:g}"/>')
    return "\n".join(parts)


def write_pdf(svg_path: Path, pdf_path: Path) -> None:
    try:
        subprocess.run(["rsvg-convert", "-f", "pdf", "-o", str(pdf_path), str(svg_path)], check=True)
    except FileNotFoundError:
        print("Skipped PDF: rsvg-convert is not installed")
        return
    print(f"Wrote {pdf_path}")


def escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


if __name__ == "__main__":
    main()

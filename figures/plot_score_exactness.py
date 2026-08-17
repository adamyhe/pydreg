#!/usr/bin/env python3
"""Panel (d): exactness of pydreg's raw SVR scores against real dREG's, at
every position where both tools called an informative position.

Downloads each library's paired *.dREG.infp.bw (the raw, pre-peak-calling
SVR score track -- dREG's eval_svm.R output, ported by pydreg.models /
pydreg.backend) from the adamyhe/pydreg-supporting-data HF dataset, reads
every chromosome as a dense per-bp array (~a few seconds per genome once
the file itself is local -- pybigtools' remote HTTP range-request path is
far slower than a bulk download, confirmed empirically during development),
and pairs up positions where BOTH tools reported a defined (non-NaN) score.
Pools all 13 benchmark libraries into one comparison, since the claim being
validated ("pydreg's raw scores match dREG's") isn't library-specific.

Writes two alternative views for comparison:
  - score_exactness_scatter.svg: dREG score vs pydreg score, joint density
    heatmap against the equality diagonal.
  - score_exactness_residual.svg: dREG score vs (pydreg - dREG), a
    Bland-Altman-style residual plot. A joint scatter of two nearly-
    identical variables squeezes all the interesting signal (how far off
    is pydreg) into a sliver of pixels perpendicular to the diagonal --
    thinner than one heatmap bin at this data's scale, which is what makes
    the scatter view look like a staircase rather than a smooth band. The
    residual view puts that signal on its own axis instead.

This downloads ~2.5GB across 26 bigWig files once; huggingface_hub caches
them under ~/.cache/huggingface after that, so re-runs are fast.

Usage:
    uv run python3 figures/plot_score_exactness.py
"""

from __future__ import annotations

import math

import numpy as np

from _common import LIBRARIES, PLOTS_DIR, escape, fetch, nice_ticks, SVG_STYLE


def paired_scores(lib: str) -> tuple[np.ndarray, np.ndarray]:
    import pybigtools

    dreg_bw = pybigtools.open(str(fetch("dreg", lib, "infp.bw")))
    pydreg_bw = pybigtools.open(str(fetch("pydreg", lib, "infp.bw")))
    dreg_chroms = dreg_bw.chroms()
    pydreg_chroms = pydreg_bw.chroms()
    shared = sorted(set(dreg_chroms) & set(pydreg_chroms))

    dreg_vals, pydreg_vals = [], []
    for chrom in shared:
        size = min(dreg_chroms[chrom], pydreg_chroms[chrom])
        d = dreg_bw.values(chrom, 0, size, fillna=None)
        p = pydreg_bw.values(chrom, 0, size, fillna=None)
        mask = np.isfinite(d) & np.isfinite(p)
        if mask.any():
            dreg_vals.append(d[mask].astype(np.float32))
            pydreg_vals.append(p[mask].astype(np.float32))
        del d, p, mask

    if not dreg_vals:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    return np.concatenate(dreg_vals), np.concatenate(pydreg_vals)


def blue_shade(t: float) -> str:
    light = (222, 235, 247)
    dark = (8, 48, 107)
    r = round(light[0] + (dark[0] - light[0]) * t)
    g = round(light[1] + (dark[1] - light[1]) * t)
    b = round(light[2] + (dark[2] - light[2]) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def density_heatmap_svg(
    x: np.ndarray,
    y: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    *,
    title: str,
    x_label: str,
    y_label: str,
    note: str,
    ref_line: tuple[tuple[float, float], tuple[float, float]] | None,
    bins: int = 140,
    x_tick_fmt: str = "{:.2f}",
    y_tick_fmt: str = "{:.2f}",
) -> str:
    """Shared 2D-histogram-as-SVG-rects renderer for both the joint scatter
    and the residual view below -- only the data, axis ranges/labels, and
    optional reference line (a straight line in data space, e.g. the y=x
    diagonal or a y=0 horizontal) differ between the two."""
    x_lo, x_hi = x_range
    y_lo, y_hi = y_range

    counts, _, _ = np.histogram2d(x, y, bins=bins, range=[[x_lo, x_hi], [y_lo, y_hi]])
    max_count = counts.max()

    width, height = 560, 560
    # margin_left has to fit BOTH the rotated y-axis label AND up to
    # 7-character tick text (e.g. "-0.0271") at the shared 24px tick/axis
    # font -- narrower and they collide (confirmed on real output).
    margin_left, margin_top, margin_right, margin_bottom = 160, 55, 30, 90
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    bottom = margin_top + plot_h

    def sx(v):
        return margin_left + (v - x_lo) / (x_hi - x_lo) * plot_w

    def sy(v):
        return margin_top + (1 - (v - y_lo) / (y_hi - y_lo)) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        f"<title>{escape(title)}</title>",
        SVG_STYLE,
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        f'<text x="{margin_left}" y="30" class="title">{escape(title)}</text>',
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_w}" height="{plot_h}" class="plot"/>',
    ]

    cell_w = plot_w / bins
    cell_h = plot_h / bins
    for i in range(bins):
        col = counts[i]
        for j in range(bins):
            c = col[j]
            if c <= 0:
                continue
            intensity = math.log1p(c) / math.log1p(max_count)
            cx = margin_left + i * cell_w
            cy_top = margin_top + (bins - 1 - j) * cell_h
            parts.append(
                f'<rect x="{cx:.2f}" y="{cy_top:.2f}" width="{cell_w:.2f}" height="{cell_h:.2f}" '
                f'fill="{blue_shade(intensity)}"/>'
            )

    if ref_line is not None:
        (rx1, ry1), (rx2, ry2) = ref_line
        parts.append(
            f'<line x1="{sx(rx1):.1f}" y1="{sy(ry1):.1f}" x2="{sx(rx2):.1f}" y2="{sy(ry2):.1f}" class="diagonal"/>'
        )

    for tick in nice_ticks(x_lo, x_hi, 5):
        tx = sx(tick)
        parts.append(f'<text x="{tx:.1f}" y="{bottom + 22}" class="tick" text-anchor="middle">{x_tick_fmt.format(tick)}</text>')
    for tick in nice_ticks(y_lo, y_hi, 5):
        ty = sy(tick)
        parts.append(f'<text x="{margin_left - 10}" y="{ty + 4:.1f}" class="tick" text-anchor="end">{y_tick_fmt.format(tick)}</text>')

    parts.extend(
        [
            f'<text x="{margin_left + plot_w / 2}" y="{bottom + 55}" class="axis" text-anchor="middle">{escape(x_label)}</text>',
            f'<text x="25" y="{margin_top + plot_h / 2}" class="axis" text-anchor="middle" '
            f'transform="rotate(-90 25 {margin_top + plot_h / 2})">{escape(y_label)}</text>',
            f'<text x="{margin_left + 8}" y="{margin_top + 20}" class="note" text-anchor="start">{escape(note)}</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts)


def scatter_view(x: np.ndarray, y: np.ndarray, n: int, r: float, max_abs_diff: float, bins: int = 400) -> str:
    """Joint density scatter, dREG score vs pydreg score, against the y=x
    diagonal. At fine bin resolution to minimize (not eliminate) the
    staircase artifact described in this module's docstring."""
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    pad = (hi - lo) * 0.03
    lo, hi = lo - pad, hi + pad
    note = f"r={r:.4f}"
    return density_heatmap_svg(
        x,
        y,
        (lo, hi),
        (lo, hi),
        title="SVR scatter",
        x_label="dREG raw SVR score",
        y_label="pydreg raw SVR score",
        note=note,
        ref_line=((lo, lo), (hi, hi)),
        bins=bins,
    )


def residual_view(x: np.ndarray, diff: np.ndarray, n: int, r: float, max_abs_diff: float, bins: int = 200) -> str:
    """Bland-Altman-style residual plot: dREG score vs (pydreg - dREG).
    Zoomed to the true max|diff| (not a tight percentile) so the plot shows
    the actual shape of the distribution -- dense near zero, tapering
    toward the extremes -- rather than an artificially cropped view where
    "99.5% of points fit in frame" just means the frame looks solid."""
    x_lo = float(x.min())
    x_hi = float(x.max())
    x_pad = (x_hi - x_lo) * 0.03
    x_lo, x_hi = x_lo - x_pad, x_hi + x_pad

    y_bound = max(max_abs_diff * 1.05, 1e-6)
    note = f"max|diff|={max_abs_diff:.3g}"
    return density_heatmap_svg(
        x,
        diff,
        (x_lo, x_hi),
        (-y_bound, y_bound),
        title="SVR residuals",
        x_label="dREG raw SVR score",
        y_label="pydreg - dREG (delta)",
        note=note,
        ref_line=((x_lo, 0.0), (x_hi, 0.0)),
        bins=bins,
        y_tick_fmt="{:.4f}",
    )


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    all_dreg, all_pydreg = [], []
    for lib in LIBRARIES:
        print(f"Fetching {lib}...")
        d, p = paired_scores(lib)
        print(f"  {len(d):,} matched informative positions")
        all_dreg.append(d)
        all_pydreg.append(p)

    dreg_scores = np.concatenate(all_dreg)
    pydreg_scores = np.concatenate(all_pydreg)
    n = len(dreg_scores)
    if n == 0:
        raise SystemExit("No matched informative positions found across any library")

    diff = pydreg_scores - dreg_scores
    max_abs_diff = float(np.max(np.abs(diff)))
    median_abs_diff = float(np.median(np.abs(diff)))
    r = float(np.corrcoef(dreg_scores, pydreg_scores)[0, 1])
    print(f"n={n:,} pooled positions across {len(LIBRARIES)} libraries")
    print(f"Pearson r={r:.6f}, max|diff|={max_abs_diff:.6g}, median|diff|={median_abs_diff:.6g}")

    scatter_out = PLOTS_DIR / "score_exactness_scatter.svg"
    scatter_out.write_text(scatter_view(dreg_scores, pydreg_scores, n, r, max_abs_diff) + "\n")
    print(f"Wrote {scatter_out}")

    residual_out = PLOTS_DIR / "score_exactness_residual.svg"
    residual_out.write_text(residual_view(dreg_scores, diff, n, r, max_abs_diff) + "\n")
    print(f"Wrote {residual_out}")


if __name__ == "__main__":
    main()

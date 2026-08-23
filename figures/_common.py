"""Shared helpers for the pydreg-vs-dREG benchmark figure scripts
(plot_walltime.py, plot_memory.py, plot_score_exactness.py,
plot_peak_agreement.py): fetching the adamyhe/pydreg-supporting-data HF
dataset (cached via huggingface_hub's own on-disk cache -- never copied
into this repo) plus the log-log scatter SVG style first established in
figures/legacy/plot_timing_comparison.py.

Not a package module -- each plot_*.py script imports this as a sibling
file (`from _common import ...`), relying on Python's default of adding a
directly-run script's own directory to sys.path.
"""

from __future__ import annotations

import html
import math
import re
from pathlib import Path

HF_REPO = "adamyhe/pydreg-supporting-data"

# Local override: a benchmark_data/<tool>/<name> file, if present, is used
# in place of the HF dataset -- lets a fresh local benchmark run (e.g. one
# not yet uploaded to HF) feed the figures without touching the network.
BENCHMARK_DATA_DIR = Path(__file__).parent / "benchmark_data"

# The 12 unique libraries in the finalized 0.2.7 benchmark (see
# figures/timing_scripts_pydreg_only.sh). G2 was a duplicate upload of
# K562_groseq (same underlying library, uploaded under two names by
# mistake) and has been removed from both the HF dataset and this list --
# don't re-add it without re-checking that's still true.
LIBRARIES = [
    "G1",
    "G3",
    "G5",
    "G6",
    "G7",
    "GM12878_groseq",
    "K562_groseq",
    "Jurkat_PROseq",
    "Jurkat_ChROseq_1",
    "Jurkat_ChROseq_2",
    "Jurkat_ChROseq",
    "Jurkat_leChROseq",
]

PLOTS_DIR = Path(__file__).parent / "plots"

SVG_STYLE = """
<style>
text { font-family: DejaVu Sans, Helvetica, Arial, sans-serif; fill: #202124; }
.title { font-size: 28px; font-weight: 600; }
.axis { font-size: 24px; }
.tick, .note, .value { font-size: 24px; fill: #5f6368; }
.lib { font-size: 24px; }
.plot { fill: #fff; stroke: #9aa0a6; }
.grid { stroke: #e3e6e8; stroke-width: 1; }
.diagonal { stroke: #5f6368; stroke-width: 2; stroke-dasharray: 7 6; }
.ref { stroke: #d62728; stroke-width: 1.2; stroke-dasharray: 5 4; }
.mark { fill: #2878b5; stroke: #174a6e; stroke-width: 1.5; }
.stem { stroke: #9aa0a6; stroke-width: 1.4; }
</style>
""".strip()


def fetch(tool: str, lib: str, suffix: str) -> Path:
    """Downloads (or returns the already-cached copy of) a file from
    adamyhe/pydreg-supporting-data, via huggingface_hub's own
    content-addressed cache (~/.cache/huggingface by default) -- never
    copied into this repo. `suffix` is e.g. "infp.bw" or
    "peak.prob.bed.gz" for the ".dREG."-infixed outputs, or the literal
    "time.log" for the benchmark logs (which aren't dREG-infixed)."""
    name = f"{lib}.time.log" if suffix == "time.log" else f"{lib}.dREG.{suffix}"

    local = BENCHMARK_DATA_DIR / tool / name
    if local.exists():
        return local

    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=HF_REPO, repo_type="dataset", filename=f"{tool}/{name}"))


def parse_elapsed(value: str) -> float:
    parts = [float(p) for p in value.split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return parts[0] * 60 + parts[1]


def parse_time_log(path: Path) -> dict | None:
    """Parses a `/usr/bin/time -v` log; returns None for a run that didn't
    complete (e.g. a run interrupted by SIGINT) so callers can skip it
    rather than plot garbage."""
    text = path.read_text()
    if re.search(r"^Command (?:exited with non-zero status|terminated by signal)", text, re.MULTILINE):
        return None
    status = re.search(r"Exit status:\s*(\d+)", text)
    elapsed = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\S+)", text)
    rss = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    if not (status and status.group(1) == "0" and elapsed and rss):
        return None
    return {"wall_seconds": parse_elapsed(elapsed.group(1)), "rss_kb": int(rss.group(1))}


def escape(value: str) -> str:
    return html.escape(str(value))


def log_point(value, low, high, start, length, invert=False):
    """Maps `value` on a log scale over [low, high] to [start, start+length]
    (or the mirrored range if invert, for SVG's flipped y-axis)."""
    fraction = (math.log(value) - math.log(low)) / (math.log(high) - math.log(low))
    if invert:
        fraction = 1 - fraction
    return start + fraction * length


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    step = (hi - lo) / (n - 1)
    return [lo + i * step for i in range(n)]


def scatter_panel_svg(rows, x_key, y_key, bounds, ticks, x_label, y_label, title, note, tooltip_fmt, width=560, height=560):
    """One self-contained log-log scatter SVG: an equality diagonal plus one
    point per row, styled identically to the original combined
    dreg-vs-pydreg comparison (figures/legacy/plot_timing_comparison.py),
    just split out to one panel per file."""
    margin_left, margin_top, margin_right, margin_bottom = 90, 55, 30, 90
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    right = margin_left + plot_w
    bottom = margin_top + plot_h
    low, high = bounds

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        f"<title>{escape(title)}</title>",
        SVG_STYLE,
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        f'<text x="{margin_left}" y="30" class="title">{escape(title)}</text>',
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_w}" height="{plot_h}" class="plot"/>',
    ]

    for tick in ticks:
        x = log_point(tick, low, high, margin_left, plot_w)
        y = log_point(tick, low, high, margin_top, plot_h, invert=True)
        parts.extend(
            [
                f'<line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{bottom}" class="grid"/>',
                f'<line x1="{margin_left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" class="grid"/>',
                f'<text x="{x:.1f}" y="{bottom + 22}" class="tick" text-anchor="middle">{tick:g}</text>',
                f'<text x="{margin_left - 10}" y="{y + 4:.1f}" class="tick" text-anchor="end">{tick:g}</text>',
            ]
        )

    x1 = log_point(low, low, high, margin_left, plot_w)
    y1 = log_point(low, low, high, margin_top, plot_h, invert=True)
    x2 = log_point(high, low, high, margin_left, plot_w)
    y2 = log_point(high, low, high, margin_top, plot_h, invert=True)
    parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="diagonal"/>')

    for row in rows:
        x = log_point(row[x_key], low, high, margin_left, plot_w)
        y = log_point(row[y_key], low, high, margin_top, plot_h, invert=True)
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" class="mark">'
            f"<title>{escape(tooltip_fmt(row))}</title></circle>"
        )

    parts.extend(
        [
            f'<text x="{margin_left + plot_w / 2}" y="{bottom + 55}" class="axis" text-anchor="middle">{escape(x_label)}</text>',
            f'<text x="{margin_left - 62}" y="{margin_top + plot_h / 2}" class="axis" text-anchor="middle" '
            f'transform="rotate(-90 {margin_left - 62} {margin_top + plot_h / 2})">{escape(y_label)}</text>',
            f'<text x="{margin_left + 8}" y="{margin_top + 20}" class="note" text-anchor="start">{escape(note)}</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts)

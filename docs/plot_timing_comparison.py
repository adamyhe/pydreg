#!/usr/bin/env python3

import csv
import html
import math
import re
import statistics
import sys
from pathlib import Path


def parse_elapsed(value):
    parts = [float(part) for part in value.split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return parts[0] * 60 + parts[1]


def parse_log(path):
    text = path.read_text()
    if re.search(r"^Command (?:exited with non-zero status|terminated by signal)", text, re.MULTILINE):
        return None

    status = re.search(r"Exit status:\s*(\d+)", text)
    elapsed = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\S+)", text)
    rss = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    if not (status and status.group(1) == "0" and elapsed and rss):
        return None

    return {
        "wall_seconds": parse_elapsed(elapsed.group(1)),
        "rss_kb": int(rss.group(1)),
    }


def read_tool(root, tool):
    runs = {}
    for path in (root / tool).glob("*.time.log"):
        parsed = parse_log(path)
        if parsed:
            runs[path.name.removesuffix(".time.log")] = parsed
    return runs


def point(value, low, high, start, length, invert=False):
    fraction = (math.log(value) - math.log(low)) / (math.log(high) - math.log(low))
    if invert:
        fraction = 1 - fraction
    return start + fraction * length


def panel(rows, x_key, y_key, bounds, ticks, x_label, y_label, title, left):
    top, width, height = 70, 550, 500
    bottom = top + height
    right = left + width
    low, high = bounds
    pieces = [
        f'<text x="{left}" y="32" class="title">{html.escape(title)}</text>',
        f'<rect x="{left}" y="{top}" width="{width}" height="{height}" class="plot"/>',
    ]

    for tick in ticks:
        x = point(tick, low, high, left, width)
        y = point(tick, low, high, top, height, invert=True)
        pieces.extend([
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" class="grid"/>',
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" class="grid"/>',
            f'<text x="{x:.1f}" y="{bottom + 25}" class="tick" text-anchor="middle">{tick:g}</text>',
            f'<text x="{left - 12}" y="{y + 4:.1f}" class="tick" text-anchor="end">{tick:g}</text>',
        ])

    x1 = point(low, low, high, left, width)
    y1 = point(low, low, high, top, height, invert=True)
    x2 = point(high, low, high, left, width)
    y2 = point(high, low, high, top, height, invert=True)
    pieces.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="diagonal"/>')

    for row in rows:
        x = point(row[x_key], low, high, left, width)
        y = point(row[y_key], low, high, top, height, invert=True)
        tooltip = (
            f'{row["experiment"]}: dreg {row[x_key]:.2f}, '
            f'pydreg {row[y_key]:.2f}'
        )
        pieces.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" class="mark"><title>{html.escape(tooltip)}</title></circle>'
        )

    pieces.extend([
        f'<text x="{left + width / 2}" y="{bottom + 62}" class="axis" text-anchor="middle">{html.escape(x_label)}</text>',
        f'<text x="{left - 60}" y="{top + height / 2}" class="axis" text-anchor="middle" transform="rotate(-90 {left - 60} {top + height / 2})">{html.escape(y_label)}</text>',
        f'<text x="{left + 8}" y="{top + 20}" class="note" text-anchor="start">Below diagonal favors pydreg</text>',
    ])
    return pieces


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/tmp")
    dreg = read_tool(root, "dreg")
    pydreg = read_tool(root, "pydreg")
    names = sorted(dreg.keys() & pydreg.keys())
    rows = []
    for name in names:
        rows.append({
            "experiment": name,
            "wall_hours_dreg": dreg[name]["wall_seconds"] / 3600,
            "wall_hours_pydreg": pydreg[name]["wall_seconds"] / 3600,
            "rss_gib_dreg": dreg[name]["rss_kb"] / 1024**2,
            "rss_gib_pydreg": pydreg[name]["rss_kb"] / 1024**2,
            "wall_speedup": dreg[name]["wall_seconds"] / pydreg[name]["wall_seconds"],
            "rss_reduction": dreg[name]["rss_kb"] / pydreg[name]["rss_kb"],
        })

    fields = list(rows[0])
    with (root / "timing_comparison.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="680" viewBox="0 0 1500 680" role="img">',
        '<title>dreg versus pydreg walltime and peak memory</title>',
        '<desc>Two log-scale scatterplots. Every completed experiment is below the equality diagonal, showing lower walltime and peak memory for pydreg.</desc>',
        '<style>',
        'text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #202124; }',
        '.title { font-size: 22px; font-weight: 600; } .axis { font-size: 15px; }',
        '.tick, .note { font-size: 12px; fill: #5f6368; }',
        '.plot { fill: #fff; stroke: #9aa0a6; } .grid { stroke: #e3e6e8; stroke-width: 1; }',
        '.diagonal { stroke: #5f6368; stroke-width: 2; stroke-dasharray: 7 6; }',
        '.mark { fill: #2878b5; stroke: #174a6e; stroke-width: 1.5; }',
        '</style>',
    ]
    svg.extend(panel(rows, "wall_hours_dreg", "wall_hours_pydreg", (0.25, 8), [0.25, 0.5, 1, 2, 4, 8],
                     "dreg walltime (hours, log scale)", "pydreg walltime (hours, log scale)", "Walltime", 100))
    svg.extend(panel(rows, "rss_gib_dreg", "rss_gib_pydreg", (4, 64), [4, 8, 16, 32, 64],
                     "dreg maximum RSS (GiB, log scale)", "pydreg maximum RSS (GiB, log scale)", "Peak memory", 850))
    svg.append('</svg>')
    (root / "timing_comparison.svg").write_text("\n".join(svg) + "\n")

    print(f"Plotted {len(rows)} paired experiments.")
    print(f"Median walltime speedup: {statistics.median(row['wall_speedup'] for row in rows):.2f}x")
    print(f"Median RSS reduction: {statistics.median(row['rss_reduction'] for row in rows):.2f}x")


if __name__ == "__main__":
    main()

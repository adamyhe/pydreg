#!/usr/bin/env python3
"""Panel (c): dREG vs pydreg peak memory (max RSS) across the benchmark
libraries.

Same data source and run pairing as plot_walltime.py -- see that script's
docstring.

Usage:
    uv run python3 figures/plot_memory.py
"""

from __future__ import annotations

import statistics

from _common import LIBRARIES, PLOTS_DIR, fetch, parse_time_log, scatter_panel_svg


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for lib in LIBRARIES:
        dreg = parse_time_log(fetch("dreg", lib, "time.log"))
        pyd = parse_time_log(fetch("pydreg", lib, "time.log"))
        if dreg is None or pyd is None:
            print(f"Skipping {lib}: dREG or pydreg run did not complete (see its time.log)")
            continue
        rows.append(
            {
                "library": lib,
                "dreg_gib": dreg["rss_kb"] / 1024**2,
                "pydreg_gib": pyd["rss_kb"] / 1024**2,
            }
        )

    if not rows:
        raise SystemExit("No valid paired timing data found")

    svg = scatter_panel_svg(
        rows,
        "dreg_gib",
        "pydreg_gib",
        bounds=(4, 40),
        ticks=[4, 8, 16, 32],
        x_label="dREG maximum RSS (GiB, log scale)",
        y_label="pydreg maximum RSS (GiB, log scale)",
        title="Peak memory",
        note="Below diagonal favors pydreg",
        tooltip_fmt=lambda r: f'{r["library"]}: dREG {r["dreg_gib"]:.2f} GiB, pydreg {r["pydreg_gib"]:.2f} GiB',
    )
    out = PLOTS_DIR / "memory.svg"
    out.write_text(svg + "\n")
    print(f"Wrote {out}")

    reductions = [r["dreg_gib"] / r["pydreg_gib"] for r in rows]
    print(
        f"n={len(rows)} libraries, median RSS reduction {statistics.median(reductions):.2f}x "
        f"(range {min(reductions):.2f}x-{max(reductions):.2f}x)"
    )


if __name__ == "__main__":
    main()

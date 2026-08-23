#!/usr/bin/env python3
"""Panel (b): dREG vs pydreg wall-clock time across the benchmark libraries.

Downloads each library's paired *.time.log (`/usr/bin/time -v` output) from
the adamyhe/pydreg-supporting-data HF dataset -- see
timing_scripts_dreg_apptainer.sh / timing_scripts_pydreg_only.sh for exactly
how these were produced -- and
plots them as a log-log scatter against the equality diagonal (points below
the diagonal favor pydreg).

Usage:
    uv run python3 figures/plot_walltime.py
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
            print(
                f"Skipping {lib}: dREG or pydreg run did not complete (see its time.log)"
            )
            continue
        rows.append(
            {
                "library": lib,
                "dreg_hours": dreg["wall_seconds"] / 3600,
                "pydreg_hours": pyd["wall_seconds"] / 3600,
            }
        )

    if not rows:
        raise SystemExit("No valid paired timing data found")

    svg = scatter_panel_svg(
        rows,
        "dreg_hours",
        "pydreg_hours",
        bounds=(0.125, 5),
        ticks=[0.125, 0.25, 0.5, 1, 2, 4],
        x_label="dREG wall time (hours, log scale)",
        y_label="pydreg wall time (hours, log scale)",
        title="Wall time",
        note="Below diagonal favors pydreg",
        tooltip_fmt=lambda r: (
            f"{r['library']}: dREG {r['dreg_hours']:.2f}h, pydreg {r['pydreg_hours']:.2f}h"
        ),
    )
    out = PLOTS_DIR / "walltime.svg"
    out.write_text(svg + "\n")
    print(f"Wrote {out}")

    speedups = [r["dreg_hours"] / r["pydreg_hours"] for r in rows]
    print(
        f"n={len(rows)} libraries, median speedup {statistics.median(speedups):.2f}x "
        f"(range {min(speedups):.2f}x-{max(speedups):.2f}x)"
    )


if __name__ == "__main__":
    main()

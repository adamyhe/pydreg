"""Shared helpers for the cross-library GPU-profiling figure scripts
(plot_gpu_utilization.py, plot_gpu_time_breakdown.py,
plot_gpu_efficiency.py).

Each of those started life reading one hardcoded capture of a single
library (Jurkat_PROseq); they now take a whole sweep -- one dREG run and
one pydreg run per library -- and draw a row per library. This module
holds what they all need to agree on: how a sweep's runs are labeled and
discovered on disk, how a library's two runs get paired up and validated
against each other, and the shared color/SVG conventions.

Label convention, as produced by profile_gpu_all.sh:

    gpu_out/dreg_<LIBRARY>.{dmon.csv,log,nsys-rep,start_epoch}
    gpu_out/pydreg_<LIBRARY>.{dmon.csv,log,nsys-rep,start_epoch}
    gpu_out/summary_dreg_<LIBRARY>_vs_pydreg_<LIBRARY>.json

Captures made before that convention existed (labels without a library
suffix) can still be plotted by naming them explicitly:

    --pair Jurkat_PROseq=dreg:pydreg2

Not a package module -- imported as a sibling file, same as
figures/_common.py.
"""

from __future__ import annotations

import html
import json
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).parent
PLOTS_DIR = HERE.parent / "plots"
DEFAULT_GPU_OUT = HERE / "gpu_out"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
# LIBRARIES is the same 12-library list the rest of figures/ benchmarks
# against, imported rather than re-listed here so a library can't be added
# in one place and silently missed in the other.
from _common import LIBRARIES  # noqa: E402
from analyze_gpu_profile import (  # noqa: E402
    DEFAULT_PHASE_END_RE,
    DEFAULT_PHASE_START_RE,
    parse_dmon,
    parse_log_window,
    parse_positions,
    parse_query_chunk,
    select_active_gpu,
    window_df,
)

# Tool-identity colors, shared by plot_gpu_utilization.py and
# plot_gpu_efficiency.py. Deliberately NOT the same palette as the
# idle/kernel/memcpy colors below, which encode a different dimension
# (time TYPE, not which tool) -- reusing one for both would make the same
# color mean two things across the figure set. Both pairs were validated
# together via the dataviz skill's validate_palette.js: CVD delta-E 24.7,
# normal-vision delta-E 33.6, clear of the pass floors.
COLOR_DREG = "#eb6834"
COLOR_PYDREG = "#2a78d6"

COLOR_IDLE = "#c7c6bd"
COLOR_KERNEL = "#1baf7a"
COLOR_MEMCPY = "#4a3aa7"

SVG_STYLE = """
<style>
text { font-family: DejaVu Sans, Helvetica, Arial, sans-serif; fill: #202124; }
.title { font-size: 26px; font-weight: 600; }
.subtitle { font-size: 17px; fill: #5f6368; }
.colhead { font-size: 19px; font-weight: 600; }
.rowlabel { font-size: 16px; font-weight: 600; }
.panelstat { font-size: 14px; fill: #5f6368; }
.tick { font-size: 14px; fill: #5f6368; }
.axis { font-size: 16px; fill: #5f6368; }
.value { font-size: 14px; font-weight: 600; }
.gap { font-size: 14px; fill: #202124; }
.legend { font-size: 16px; }
.grid { stroke: #e3e6e8; stroke-width: 1; }
.plotbg { fill: #fbfbfa; stroke: #e3e6e8; }
.connector { stroke: #c7c6bd; stroke-width: 2; }
</style>
""".strip()


def escape(value):
    return html.escape(str(value))


@dataclass
class Pair:
    """One library's dREG run paired with its pydreg run."""

    library: str
    dreg_label: str
    pydreg_label: str
    outdir: Path

    @property
    def labels(self):
        return (self.dreg_label, self.pydreg_label)

    def dmon_path(self, label):
        return self.outdir / f"{label}.dmon.csv"

    def log_path(self, label):
        return self.outdir / f"{label}.log"

    def summary_path(self):
        return self.outdir / f"summary_{self.dreg_label}_vs_{self.pydreg_label}.json"

    def has_dmon(self):
        return all(self.dmon_path(lb).exists() for lb in self.labels)

    def has_summary(self):
        return self.summary_path().exists()

    def load_summary(self):
        """Returns {label: summary_dict} for this pair's analyzer output."""
        by_label = {s["label"]: s for s in json.loads(self.summary_path().read_text())}
        missing = [lb for lb in self.labels if lb not in by_label]
        if missing:
            raise SystemExit(
                f"{self.summary_path()} has no entry for {missing} -- it was "
                "written by an analyze_gpu_profile.py run over different "
                f"labels ({sorted(by_label)}). Re-run the analyzer for this pair."
            )
        return by_label

    def positions(self, by_label=None):
        """The informative-position count both runs scored.

        Prefers the analyzer's parsed value, falling back to re-reading the
        logs (which is what makes pre-`positions` summaries still usable).
        Raises if the two sides disagree: every per-operation ratio these
        figures draw divides by this number, so two runs over different
        position sets would produce a plausible-looking but meaningless
        dumbbell rather than an obvious error."""
        counts = {}
        for label in self.labels:
            n = (by_label or {}).get(label, {}).get("positions")
            if n is None:
                n = parse_positions(self.log_path(label))
            if n is not None:
                counts[label] = n
        if not counts:
            return None
        if len(set(counts.values())) > 1:
            raise SystemExit(
                f"[{self.library}] the two runs scored DIFFERENT position "
                f"counts ({counts}) -- they aren't comparable, and any "
                "per-operation ratio between them would be meaningless. "
                "Check the right two runs got paired."
            )
        return next(iter(counts.values()))

    def query_chunk(self):
        """The query-chunk size the pydreg run scored with, read from its
        own log, or None if it didn't state one. Only pydreg logs this;
        dREG has no equivalent knob."""
        return parse_query_chunk(self.log_path(self.pydreg_label))


def parse_pair_specs(specs):
    """Parses repeated --pair LIB=DREGLABEL:PYDREGLABEL options."""
    out = {}
    for spec in specs or []:
        try:
            library, labels = spec.split("=", 1)
            dreg_label, pydreg_label = labels.split(":", 1)
        except ValueError:
            raise SystemExit(
                f"--pair {spec!r} is malformed -- expected "
                "LIBRARY=DREG_LABEL:PYDREG_LABEL, e.g. "
                "Jurkat_PROseq=dreg:pydreg2"
            )
        out[library] = (dreg_label, pydreg_label)
    return out


def discover_pairs(outdir, libraries=None, pair_specs=None, require="summary"):
    """Finds every library in the sweep that has usable artifacts on disk.

    `require` is "summary" (the analyzer's JSON, for the two summary-driven
    figures) or "dmon" (the raw utilization series, which the utilization
    figure reads directly since the summary only keeps aggregates).

    Libraries with nothing captured yet are skipped with a note rather than
    erroring -- a partially-finished sweep should still plot what it has,
    since the full 12-library capture takes the better part of a GPU-day.
    """
    explicit = parse_pair_specs(pair_specs)
    wanted = list(libraries or LIBRARIES)
    for library in explicit:
        if library not in wanted:
            wanted.append(library)

    check = {"summary": Pair.has_summary, "dmon": Pair.has_dmon}[require]
    found, skipped = [], []
    for library in wanted:
        dreg_label, pydreg_label = explicit.get(
            library, (f"dreg_{library}", f"pydreg_{library}")
        )
        pair = Pair(library, dreg_label, pydreg_label, Path(outdir))
        (found if check(pair) else skipped).append(pair)

    if skipped:
        print(
            f"note: no {require} artifacts for "
            f"{', '.join(p.library for p in skipped)} -- skipping "
            f"({len(found)} of {len(wanted)} libraries plotted)"
        )
    if not found:
        raise SystemExit(
            f"no library in {outdir} has {require} artifacts under the "
            "expected dreg_<LIB>/pydreg_<LIB> label convention. Run "
            "profile_gpu_all.sh first, or name a non-conventional capture "
            "explicitly with --pair LIB=DREG_LABEL:PYDREG_LABEL."
        )
    return found


def add_common_args(parser):
    """The --outdir/--libraries/--pair/--gpu-index options all three
    cross-library figure scripts share."""
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_GPU_OUT,
        help="directory profile_gpu.sh wrote its artifacts to (default: %(default)s)",
    )
    parser.add_argument(
        "--libraries",
        default="",
        help="comma-separated subset of libraries to plot (default: all 12 "
        "in figures/_common.py's LIBRARIES that have artifacts on disk)",
    )
    parser.add_argument(
        "--pair",
        action="append",
        metavar="LIB=DREG_LABEL:PYDREG_LABEL",
        help="name a library's two runs explicitly, for captures that "
        "predate the dreg_<LIB>/pydreg_<LIB> label convention "
        "(e.g. --pair Jurkat_PROseq=dreg:pydreg2). Repeatable.",
    )
    return parser


def resolve_pairs(args, require):
    libraries = [x for x in args.libraries.split(",") if x] or None
    return discover_pairs(args.outdir, libraries, args.pair, require=require)


def load_series(pair, label, gpu_index, whole_trace):
    """The `sm` utilization time series for one run, restricted to the
    scoring phase (except for dREG, whose whole process already is that
    phase -- see README.md). Returns (seconds_since_phase_start, sm_pct)."""
    df = select_active_gpu(parse_dmon(pair.dmon_path(label)), label, gpu_index)
    window = (
        None
        if whole_trace
        else parse_log_window(
            pair.log_path(label), DEFAULT_PHASE_START_RE, DEFAULT_PHASE_END_RE
        )
    )
    df = window_df(df, window).sort_values("timestamp")
    t0 = df["timestamp"].min()
    seconds = (df["timestamp"] - t0).dt.total_seconds().tolist()
    return seconds, df["sm"].tolist()


def write_svg(parts, filename):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / filename
    out.write_text("\n".join(parts) + "\n")
    print(f"Wrote {out}")
    return out

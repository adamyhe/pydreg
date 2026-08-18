"""Summarizes GPU-profiling artifacts produced by profile_gpu.sh for one or
more labeled runs (e.g. "dreg" and "pydreg"), so the two can be compared
before deciding what's worth plotting. Two complementary views:

  - dmon utilization series: nvidia-smi's own "Volatile GPU-Util" (the `sm`
    column) and memory-controller utilization (`mem`) over wall-clock time.
    Coarse (~1 Hz, driver-smoothed) but directly answers "how spiky does it
    look."
  - nsys idle-gap analysis: gaps between consecutive GPU-side events
    (kernels/memcpys/memsets) from the `cuda_gpu_trace` report. This is what
    actually distinguishes *why* utilization is spiky -- regular,
    similarly-sized gaps between short bursts of GPU work point at a
    per-chunk host/device round trip with no overlap (host-bound scheduling,
    the same symptom pydreg's own extract/predict prefetch fix addressed --
    see docs/PERF_LOG.md's 2026-07-14/15 entries), whereas long *unbroken*
    GPU-busy stretches with low achieved throughput would point at the
    kernels themselves being memory-bound (the original sparse-kernel
    hypothesis) instead. dmon's utilization number alone can't tell these
    apart; this can.

No plotting here -- prints and dumps JSON summaries only, so real data can
be reviewed before deciding what a figure should look like.
"""

import argparse
import csv
import io
import json
import re
import shutil
import statistics
import subprocess
from pathlib import Path

import pandas as pd

NSYS_AVAILABLE = shutil.which("nsys") is not None

DEFAULT_PHASE_START_RE = r"scoring informative positions\.\.\."
DEFAULT_PHASE_END_RE = r"scoring informative positions done in"
# Python logging's default asctime format: "2026-08-17 14:32:01,123"
LOG_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")


def parse_log_window(log_path, start_re, end_re):
    """Returns (start, end) pandas Timestamps bracketing the named phase by
    reading it back out of pydreg's own -v INFO log, or None if either
    marker line (or the log file) is missing -- callers should fall back to
    the whole trace in that case, which is the correct behavior for a tool
    (like dREG's run_predict.bsh) that has no other phases to separate out."""
    if not log_path.exists():
        return None
    start = end = None
    start_pat, end_pat = re.compile(start_re), re.compile(end_re)
    for line in log_path.read_text(errors="replace").splitlines():
        m = LOG_TS_RE.match(line)
        if not m:
            continue
        if start is None and start_pat.search(line):
            start = m.group(1)
        elif start is not None and end is None and end_pat.search(line):
            end = m.group(1)
            break
    if start is None or end is None:
        return None
    fmt = "%Y-%m-%d %H:%M:%S,%f"
    return pd.to_datetime(start, format=fmt), pd.to_datetime(end, format=fmt)


def parse_dmon(dmon_path):
    """Parses `nvidia-smi dmon -s um -o DT` output. Column layout (names,
    ordering, presence of a units row) has drifted across driver versions,
    so this reads the header line dmon itself prints rather than
    hardcoding positions, and tolerates the extra units-only row."""
    lines = dmon_path.read_text(errors="replace").splitlines()
    header_lines = [l for l in lines if l.startswith("#")]
    data_lines = [l for l in lines if l.strip() and not l.startswith("#")]
    if not header_lines or not data_lines:
        raise ValueError(f"{dmon_path}: no dmon header/data lines found")
    columns = header_lines[0].lstrip("#").split()
    rows = [l.split() for l in data_lines if len(l.split()) == len(columns)]
    df = pd.DataFrame(rows, columns=columns)
    if not {"Date", "Time"} <= set(df.columns):
        raise ValueError(
            f"{dmon_path}: expected Date/Time columns from `-o DT`, got {list(df.columns)}"
        )
    df["timestamp"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], errors="coerce"
    )
    for col in df.columns:
        if col not in ("Date", "Time", "timestamp"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["timestamp"])


def select_active_gpu(df, label, gpu_index=None):
    """dmon logs one row per GPU per sample -- on a multi-GPU host, pooling
    every GPU's rows together silently dilutes every stat with whatever
    idle cards were never touched by the job (confirmed on real 2-GPU
    profiling data: doubled sample counts, ~2x inflated idle fraction).
    Picks the GPU with the most total activity (by `fb` memory used, or
    `sm` utilization if `fb` wasn't captured) unless gpu_index overrides it."""
    if "gpu" not in df.columns or df["gpu"].nunique() <= 1:
        return df
    metric = "fb" if "fb" in df.columns else "sm"
    totals = df.groupby("gpu")[metric].sum()
    selected = gpu_index if gpu_index is not None else totals.idxmax()
    print(
        f"[{label}] multiple GPUs in dmon log ({sorted(df['gpu'].unique().tolist())}) "
        f"-- using gpu {int(selected)} "
        f"({'auto-selected by total ' + metric if gpu_index is None else 'explicit --gpu-index'})"
    )
    return df[df["gpu"] == selected]


def window_df(df, window, pad=pd.Timedelta(seconds=1)):
    if window is None:
        return df
    start, end = window
    return df[(df["timestamp"] >= start - pad) & (df["timestamp"] <= end + pad)]


def summarize_util(df):
    """sm = Volatile GPU-Util equivalent; mem = memory-controller
    utilization (not memory *used* -- that's fb/bar1). Coefficient of
    variation is used as a cheap spikiness score: near 0 for a steady
    plateau, large when utilization swings between near-0 and near-100
    repeatedly."""
    out = {}
    for col in ("sm", "mem"):
        if col not in df.columns or df[col].dropna().empty:
            continue
        s = df[col].dropna()
        out[col] = {
            "mean": s.mean(),
            "median": s.median(),
            "p10": s.quantile(0.10),
            "p90": s.quantile(0.90),
            "stdev": s.std(),
            "coeff_of_variation": (s.std() / s.mean()) if s.mean() else None,
            "pct_time_idle_below_10pct": (s < 10).mean() * 100,
        }
    out["n_samples"] = len(df)
    if len(df) >= 2:
        out["span_seconds"] = (
            df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]
        ).total_seconds()
    return out


def _run_nsys_stats(rep_path, report):
    result = subprocess.run(
        ["nsys", "stats", "--report", report, "--format", "csv", str(rep_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        print(
            f"[{rep_path.name}] nsys stats --report {report} "
            f"(exit {result.returncode}) produced no usable stdout -- stderr:\n"
            f"{result.stderr.strip()}"
        )
    return result.stdout


def _parse_nsys_csv(text):
    """nsys stats prints a CSV block per requested report, sometimes
    preceded by non-CSV progress/banner lines on stdout depending on
    version. Finds the first line that parses as a header with >1 field and
    reads from there, rather than assuming stdout is clean CSV from line 1."""
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if "," in line and not line.startswith("=="):
            header_idx = i
            break
    if header_idx is None:
        return []
    reader = csv.DictReader(io.StringIO("\n".join(lines[header_idx:])))
    return list(reader)


def _num(row, *candidates):
    for key in candidates:
        if key in row and row[key] not in (None, ""):
            try:
                return float(row[key].replace(",", ""))
            except ValueError:
                pass
    return None


def gpu_trace_events(rep_path):
    """Returns (start_ns, end_ns, name) for every GPU-side event (kernel,
    memcpy, memset) in the trace, sorted by start. Column names in
    `cuda_gpu_trace` (Start/Duration vs. "Start (ns)"/"Duration (ns)", etc.)
    have varied across nsys releases -- tries the common candidates rather
    than assuming one."""
    rows = _parse_nsys_csv(_run_nsys_stats(rep_path, "cuda_gpu_trace"))
    events = []
    for row in rows:
        start = _num(row, "Start (ns)", "Start", "Start:ts_ns")
        dur = _num(row, "Duration (ns)", "Duration", "Duration:dur_ns")
        name = row.get("Name") or row.get("Kernel Name") or ""
        if start is None or dur is None:
            continue
        events.append((start, start + dur, name))
    return sorted(events)


def idle_gap_analysis(events):
    """Merges overlapping/concurrent-stream GPU events into busy intervals,
    then reports the gaps between them. Regular, similarly-sized gaps
    (low stdev relative to the mean -- reported as gap_coeff_of_variation)
    are the signature of a repeated host round trip between GPU bursts, as
    opposed to occasional large stalls."""
    if not events:
        return None
    merged = [list(events[0][:2])]
    for start, end, _ in events[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    gaps = [merged[i + 1][0] - merged[i][1] for i in range(len(merged) - 1)]
    busy_ns = sum(e - s for s, e in merged)
    total_ns = merged[-1][1] - merged[0][0]
    result = {
        "n_busy_intervals": len(merged),
        "busy_ns": busy_ns,
        "idle_ns": total_ns - busy_ns,
        "pct_busy": 100 * busy_ns / total_ns if total_ns else None,
        "n_gaps": len(gaps),
    }
    if gaps:
        mean_gap = statistics.mean(gaps)
        result["gap_mean_ns"] = mean_gap
        result["gap_median_ns"] = statistics.median(gaps)
        result["gap_stdev_ns"] = statistics.pstdev(gaps) if len(gaps) > 1 else 0.0
        result["gap_coeff_of_variation"] = (
            result["gap_stdev_ns"] / mean_gap if mean_gap else None
        )
        result["largest_gaps_ns"] = sorted(gaps, reverse=True)[:10]
    return result


def summarize_label(outdir, label, start_re, end_re, force_whole_trace, gpu_index=None):
    dmon_path = outdir / f"{label}.dmon.csv"
    log_path = outdir / f"{label}.log"
    rep_path = outdir / f"{label}.nsys-rep"

    window = None if force_whole_trace else parse_log_window(log_path, start_re, end_re)
    summary = {"label": label, "windowed": window is not None}

    if dmon_path.exists():
        df = select_active_gpu(parse_dmon(dmon_path), label, gpu_index)
        summary["dmon"] = summarize_util(window_df(df, window))
    else:
        print(f"[{label}] WARNING: {dmon_path} not found, skipping dmon summary")

    if rep_path.exists() and not NSYS_AVAILABLE:
        print(
            f"[{label}] {rep_path} found but `nsys` isn't on PATH on this "
            "machine -- skipping kernel/memcpy/idle-gap analysis (dmon "
            "utilization above is still valid). Run this script on a "
            "machine with nsys installed to get the rest."
        )
    elif rep_path.exists():
        events = gpu_trace_events(rep_path)
        if window is not None:
            # nsys timestamps are ns since profile start, not wall-clock --
            # without a captured profile-start epoch there's no exact way to
            # translate the log-derived wall-clock window into this
            # timeline, so kernel/memcpy-time reports below reflect the
            # WHOLE recorded trace, not just this phase, whenever a window
            # was found. See README.md's "nsys and windowing" section.
            print(
                f"[{label}] NOTE: nsys report reflects the whole recorded "
                "trace, not just the windowed phase -- see README.md"
            )
        summary["nsys_idle_gaps"] = idle_gap_analysis(events)
        summary["nsys_cuda_api_sum"] = _parse_nsys_csv(
            _run_nsys_stats(rep_path, "cuda_api_sum")
        )
        summary["nsys_cuda_gpu_kern_sum"] = _parse_nsys_csv(
            _run_nsys_stats(rep_path, "cuda_gpu_kern_sum")
        )
        summary["nsys_cuda_gpu_mem_time_sum"] = _parse_nsys_csv(
            _run_nsys_stats(rep_path, "cuda_gpu_mem_time_sum")
        )
    else:
        print(f"[{label}] no {rep_path} found, skipping nsys summary")

    return summary


def print_summary(summary):
    print(f"\n=== {summary['label']} (windowed={summary['windowed']}) ===")
    dmon = summary.get("dmon")
    if dmon:
        span = dmon.get("span_seconds")
        print(f"  dmon: {dmon.get('n_samples')} samples over {span:.1f}s" if span else "")
        for col in ("sm", "mem"):
            if col in dmon:
                d = dmon[col]
                print(
                    f"  {col}: mean={d['mean']:.1f}% median={d['median']:.1f}% "
                    f"p10={d['p10']:.1f}% p90={d['p90']:.1f}% "
                    f"cv={d['coeff_of_variation']:.2f} "
                    f"idle<10%={d['pct_time_idle_below_10pct']:.1f}% of samples"
                )
    gaps = summary.get("nsys_idle_gaps")
    if gaps:
        print(
            f"  nsys: {gaps['pct_busy']:.1f}% GPU-busy, {gaps['n_gaps']} idle gaps, "
            f"mean gap={gaps.get('gap_mean_ns', 0) / 1e6:.2f}ms "
            f"cv={gaps.get('gap_coeff_of_variation')}"
        )
        print(
            "    low gap cv -> regular periodic stalls (host round-trip per "
            "chunk); high cv -> occasional large stalls instead"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", type=Path)
    parser.add_argument("labels", nargs="+")
    parser.add_argument("--phase-start-regex", default=DEFAULT_PHASE_START_RE)
    parser.add_argument("--phase-end-regex", default=DEFAULT_PHASE_END_RE)
    parser.add_argument(
        "--whole-trace",
        default="",
        help="comma-separated labels to skip log-window slicing for "
        "(use for a tool with no separate phases to slice out, e.g. dREG's "
        "run_predict.bsh, whose whole process already is the target phase)",
    )
    parser.add_argument(
        "--gpu-index",
        default=None,
        help="which nvidia-smi GPU index each label's dmon data is actually "
        "on -- either a bare int applied to every label (single-GPU host), "
        "or label=index pairs, comma-separated (e.g. 'dreg=1,pydreg=0') for "
        "a shared multi-GPU host where jobs land on different physical "
        "cards, or where CUDA's device enumeration doesn't match "
        "nvidia-smi's (a real, confirmed gap -- Rgtsvm/dREG's own "
        "'GPU ID: 0' argument is a *CUDA* index, which silently mapped to "
        "nvidia-smi index 1 on real hardware; verify explicitly rather than "
        "trusting the auto-detect fallback below whenever possible). "
        "Omitted labels fall back to auto-selecting the GPU with the most "
        "total fb/sm activity, which is unreliable on a shared node where "
        "another user's job may dominate that metric on an unrelated card.",
    )
    args = parser.parse_args()
    whole_trace_labels = set(args.whole_trace.split(",")) if args.whole_trace else set()
    gpu_index_by_label = {}
    default_gpu_index = None
    if args.gpu_index is not None:
        if "=" in args.gpu_index:
            gpu_index_by_label = {
                k: int(v)
                for k, v in (pair.split("=") for pair in args.gpu_index.split(","))
            }
        else:
            default_gpu_index = int(args.gpu_index)

    unknown_keys = set(gpu_index_by_label) - set(args.labels)
    if unknown_keys:
        raise SystemExit(
            f"--gpu-index key(s) {sorted(unknown_keys)} don't match any "
            f"label in {args.labels} -- typo? Refusing to silently fall "
            "back to auto-detect for a label you thought you'd pinned."
        )

    summaries = []
    for label in args.labels:
        summary = summarize_label(
            args.outdir,
            label,
            args.phase_start_regex,
            args.phase_end_regex,
            force_whole_trace=label in whole_trace_labels,
            gpu_index=gpu_index_by_label.get(label, default_gpu_index),
        )
        summaries.append(summary)
        print_summary(summary)

    out_path = args.outdir / f"summary_{'_vs_'.join(args.labels)}.json"
    out_path.write_text(json.dumps(summaries, indent=2, default=str))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

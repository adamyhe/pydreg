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
    # --force-export=true: without it, nsys reuses a cached .sqlite export
    # next to the .nsys-rep if one exists, and errors out ("could not be
    # opened and appears to be invalid") rather than regenerating it if
    # that cache is stale/corrupt from an earlier interrupted run --
    # confirmed on real data. Always paying the re-export cost is cheap
    # relative to a silent, confusing failure on every subsequent call.
    result = subprocess.run(
        [
            "nsys",
            "stats",
            "--report",
            report,
            "--format",
            "csv",
            "--force-export=true",
            str(rep_path),
        ],
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


def load_start_epoch(outdir, label):
    """LABEL.start_epoch (written by profile_gpu.sh, captured immediately
    before nsys launched the command) is the wall-clock unix-epoch moment
    nsys's own internal ns-since-recording-start timeline begins at --
    without it, LABEL.log's wall-clock phase markers can't be translated
    into nsys's timestamps, so per-phase nsys windowing isn't possible
    (falls back to whole-trace). Returns None if missing (e.g. a capture
    made before this file existed)."""
    p = outdir / f"{label}.start_epoch"
    if not p.exists():
        return None
    try:
        return float(p.read_text().strip())
    except ValueError:
        return None


def gpu_trace_df(rep_path):
    """Returns a DataFrame (start_ns, end_ns, dur_ns, name) for every
    GPU-side event (kernel, memcpy, memset) in the trace, sorted by start.
    Column names in `cuda_gpu_trace` (Start/Duration vs. "Start (ns)"/
    "Duration (ns)", etc.) have varied across nsys releases -- tries the
    common candidates rather than assuming one."""
    rows = _parse_nsys_csv(_run_nsys_stats(rep_path, "cuda_gpu_trace"))
    records = []
    for row in rows:
        start = _num(row, "Start (ns)", "Start", "Start:ts_ns")
        dur = _num(row, "Duration (ns)", "Duration", "Duration:dur_ns")
        name = row.get("Name") or row.get("Kernel Name") or ""
        if start is None or dur is None:
            continue
        records.append({"start_ns": start, "dur_ns": dur, "end_ns": start + dur, "name": name})
    df = pd.DataFrame.from_records(records, columns=["start_ns", "end_ns", "dur_ns", "name"])
    return df.sort_values("start_ns").reset_index(drop=True)


def window_events_df(df, window, start_epoch):
    """Restricts a gpu_trace_df to a wall-clock window by converting nsys's
    recording-relative start_ns into wall-clock using start_epoch (see
    load_start_epoch). Returns df unchanged if either window or
    start_epoch is unavailable -- callers must check windowed separately
    rather than silently treating an unfiltered df as phase-scoped.

    Uses `Timestamp.to_pydatetime().timestamp()`, not `Timestamp.timestamp()`
    directly -- confirmed these disagree for naive timestamps: pandas'
    own `.timestamp()` always assumes UTC, while the stdlib's (and
    `date +%s.%N`'s, which wrote start_epoch) assumes local time. window's
    Timestamps are naive local wall-clock (parsed from LABEL.log, which is
    local time on whatever machine ran the job), so using pandas'
    UTC-assuming version here silently mis-windows by the local UTC
    offset on any non-UTC host. Also only correct if this function runs on
    the same machine that captured the data (same local timezone at both
    ends) -- true in practice since it's only reachable where `nsys` is
    installed, i.e. NSYS_AVAILABLE, which in this project has so far only
    ever been true on the capture host itself."""
    if df.empty or window is None or start_epoch is None:
        return df
    start, end = window
    wall_start_ns = df["start_ns"] + start_epoch * 1e9
    lo_ns = start.to_pydatetime().timestamp() * 1e9
    hi_ns = end.to_pydatetime().timestamp() * 1e9
    return df[(wall_start_ns >= lo_ns) & (wall_start_ns <= hi_ns)]


def _is_memory_op(name):
    lname = name.lower()
    return "memcpy" in lname or "memset" in lname


def summarize_events_by_name(df, kind_filter):
    """Aggregates a (possibly windowed) gpu_trace_df by event name, in the
    same column layout as nsys stats' own cuda_gpu_kern_sum/
    cuda_gpu_mem_time_sum reports -- computed locally, rather than via
    those reports directly, specifically so it can be windowed to one
    phase (nsys's own *_sum reports have no version-stable time-range
    filter to rely on)."""
    if df.empty:
        return []
    subset = df[df["name"].apply(kind_filter)]
    if subset.empty:
        return []
    total = subset["dur_ns"].sum()
    rows = []
    for name, s in subset.groupby("name")["dur_ns"]:
        rows.append(
            {
                "Name": name,
                "Time (%)": round(100 * s.sum() / total, 1) if total else 0.0,
                "Total Time (ns)": float(s.sum()),
                "Instances": int(s.count()),
                "Avg (ns)": float(s.mean()),
                "Med (ns)": float(s.median()),
                "Min (ns)": float(s.min()),
                "Max (ns)": float(s.max()),
                "StdDev (ns)": float(s.std()) if len(s) > 1 else 0.0,
            }
        )
    return sorted(rows, key=lambda r: -r["Total Time (ns)"])


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
        start_epoch = load_start_epoch(outdir, label)
        nsys_windowed = window is not None and start_epoch is not None
        summary["nsys_windowed"] = nsys_windowed
        if window is not None and not nsys_windowed:
            # No LABEL.start_epoch (capture predates profile_gpu.sh writing
            # it, or nsys wasn't used at capture time) -- there's no way to
            # translate the log-derived wall-clock window into nsys's
            # recording-relative timestamps, so every nsys number below
            # reflects the WHOLE recorded trace, not just this phase. See
            # README.md's "nsys and windowing" section.
            print(
                f"[{label}] NOTE: no {label}.start_epoch found -- nsys report "
                "reflects the whole recorded trace, not just the windowed "
                "phase -- see README.md"
            )

        full_df = gpu_trace_df(rep_path)
        df = window_events_df(full_df, window, start_epoch) if nsys_windowed else full_df

        events = list(zip(df["start_ns"], df["end_ns"], df["name"]))
        summary["nsys_idle_gaps"] = idle_gap_analysis(events)
        summary["nsys_cuda_gpu_kern_sum"] = summarize_events_by_name(
            df, lambda n: not _is_memory_op(n)
        )
        summary["nsys_cuda_gpu_mem_time_sum"] = summarize_events_by_name(
            df, _is_memory_op
        )
        # cuda_api_sum is CPU-side API call timing, from nsys's own
        # pre-aggregated report -- still whole-trace-only (cuda_api_trace's
        # raw per-call data would need the same local-aggregation treatment
        # as above to window it; not done here since the kernel/memcpy
        # numbers were what mattered for the scheduling/kernel-design
        # comparison this tooling was built for).
        summary["nsys_cuda_api_sum"] = _parse_nsys_csv(
            _run_nsys_stats(rep_path, "cuda_api_sum")
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
        print(f"  nsys_windowed={summary.get('nsys_windowed')}")
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

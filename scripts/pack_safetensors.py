"""Packs the raw .bin + meta.json files produced by export_dreg_model.R /
export_dreg_rf_model.R into a single .safetensors file per model — the
right format for putting these on HF, not the raw .bin dump.

Why not raw .bin, and why not pickle/joblib:

- The raw per-array .bin files (what export_*.R produces) are a minimal,
  dependency-free intermediate we chose so the R-side extraction script
  only needs base R (no package compiles required — this machine's R
  toolchain can't compile anything, see conversation). They're fine for
  that narrow purpose, but they're not a real "model format": no dtype/
  shape metadata travels with the array, and multiple files per model is
  awkward to distribute as one HF artifact.
- joblib.dump/pickle (the usual sklearn way to save a model) is explicitly
  NOT guaranteed compatible across sklearn versions, and unpickling
  executes arbitrary code — a bad idea for a file strangers will download
  from a public HF repo.
- safetensors is what HF actually built for this: a single file, a JSON
  header (dtype/shape/offsets) validated before any data is touched, no
  code execution, and it's the standard HF Hub already renders/expects for
  raw tensor weights. It doesn't know what an "SVR" is any more than a raw
  .bin does — reconstructing a real sklearn/cuml estimator from these
  tensors is still the job of dreg_model.py / dreg_rf_model.py — but it's
  the correct container for shipping the numbers themselves.

Usage:
    python pack_safetensors.py svr      <export_dir> <out.safetensors>
    python pack_safetensors.py peak_rf  <export_dir> <out.safetensors>
"""

import json
import os
import sys

import numpy as np
from safetensors.numpy import save_file


def _load_meta(export_dir):
    return json.load(open(os.path.join(export_dir, "meta.json")))


def _str_metadata(meta):
    # safetensors metadata values must be strings.
    return {k: json.dumps(v) if not isinstance(v, str) else v for k, v in meta.items()}


def pack_svr(export_dir, out_path):
    meta = _load_meta(export_dir)
    n_sv, n_features = meta["n_sv"], meta["n_features"]

    def load(name, dtype, shape):
        return np.fromfile(os.path.join(export_dir, name), dtype=dtype).reshape(shape)

    tensors = {
        "support_vectors": load("sv_matrix_f64.bin", "<f8", (n_sv, n_features)),
        "dual_coef": load("coefs_f64.bin", "<f8", (n_sv,)),
        "x_center": load("x_center_f64.bin", "<f8", (n_features,)),
        "x_scale": load("x_scale_f64.bin", "<f8", (n_features,)),
    }
    save_file(tensors, out_path, metadata=_str_metadata(meta))


def pack_peak_rf(export_dir, out_path):
    meta = _load_meta(export_dir)
    n_trees, n_nodes = meta["n_trees"], meta["n_nodes"]

    def load(name, dtype, shape):
        return np.fromfile(os.path.join(export_dir, name), dtype=dtype).reshape(shape)

    tensors = {
        "nodestatus": load("nodestatus_i32.bin", "<i4", (n_trees, n_nodes)),
        "bestvar": load("bestvar_i32.bin", "<i4", (n_trees, n_nodes)),
        "left_daughter": load("left_daughter_i32.bin", "<i4", (n_trees, n_nodes)),
        "right_daughter": load("right_daughter_i32.bin", "<i4", (n_trees, n_nodes)),
        "xbestsplit": load("xbestsplit_f64.bin", "<f8", (n_trees, n_nodes)),
        "nodepred": load("nodepred_f64.bin", "<f8", (n_trees, n_nodes)),
    }
    save_file(tensors, out_path, metadata=_str_metadata(meta))


if __name__ == "__main__":
    kind, export_dir, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    {"svr": pack_svr, "peak_rf": pack_peak_rf}[kind](export_dir, out_path)
    print(f"Wrote {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")

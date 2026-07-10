"""Opens a .safetensors file, transparently decompressing a .safetensors.zst
first if that's what was given. Measured compression on these models is
substantial (RF: ~88% smaller; SVR: ~76% smaller at zstd level 19) and
safetensors itself has no built-in compression, so distributing the .zst
form (e.g. on HF) and decompressing on load is worth the one extra step.
"""

import contextlib
import os
import tempfile

from safetensors.numpy import safe_open


@contextlib.contextmanager
def open_safetensors(path):
    if not path.endswith(".zst"):
        with safe_open(path, framework="numpy") as f:
            yield f
        return

    import zstandard

    fd, tmp_path = tempfile.mkstemp(suffix=".safetensors")
    try:
        with os.fdopen(fd, "wb") as out, open(path, "rb") as src:
            zstandard.ZstdDecompressor().copy_stream(src, out)
        with safe_open(tmp_path, framework="numpy") as f:
            yield f
    finally:
        os.remove(tmp_path)

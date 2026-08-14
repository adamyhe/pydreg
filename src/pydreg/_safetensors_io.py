"""Opens a .safetensors file, transparently decompressing a .safetensors.zst
first if that's what was given. Measured compression on these models is
substantial (RF: ~88% smaller; SVR: ~76% smaller at zstd level 19) and
safetensors itself has no built-in compression, so distributing the .zst
form (e.g. on HF) and decompressing on load is worth the one extra step.

Loads entirely in memory via `safetensors.numpy.load(bytes)` rather than
`safe_open(path)` on a temp file -- measured directly (not assumed) that
`safe_open().get_tensor()` is dramatically slower than it looks for a large
tensor: on the real pretrained SVR (605,187 x 360 float64 support vectors,
~1.7GB decompressed), `get_tensor("support_vectors")` alone took ~3.65s
against a freshly-written temp file, vs. ~0.52s for `safetensors.numpy.load`
to materialize *all four* of the model's tensors from the same bytes already
in memory -- a >7x difference attributable to `safe_open`'s mmap-based
per-tensor access path, not to the temp file or decompression themselves
(reading the compressed file and decompressing it are both well under 1.5s
on their own). `safe_open`'s only advantage here is lazy/selective tensor
access, which pydreg.models never uses -- every call site reads every
tensor in the file -- so there's no lazy-loading benefit being given up.

safetensors' `load(bytes)` doesn't expose the file's `__metadata__` header
field the way `safe_open(...).metadata()` does, so it's parsed directly here
instead: the safetensors format is a stable, documented layout (8-byte
little-endian header length, followed by that many bytes of UTF-8 JSON
containing per-tensor shape/dtype/offsets plus an optional `__metadata__`
key of string->string) -- parsing just that JSON header is microseconds,
not worth a round trip through a slower API for.
"""

import contextlib
import json
import struct

from safetensors.numpy import load as _st_load


class _LoadedSafetensors:
    """Duck-types the two `safe_open(...)` methods pydreg.models actually
    uses (`metadata()`, `get_tensor(name)`), backed by tensors/metadata
    already fully materialized in memory."""

    def __init__(self, data):
        header_len = struct.unpack("<Q", data[:8])[0]
        header = json.loads(data[8 : 8 + header_len])
        self._metadata = header.get("__metadata__", {})
        self._tensors = _st_load(data)

    def metadata(self):
        return self._metadata

    def get_tensor(self, name):
        return self._tensors[name]


@contextlib.contextmanager
def open_safetensors(path):
    with open(path, "rb") as f:
        data = f.read()

    if path.endswith(".zst"):
        import zstandard

        data = zstandard.ZstdDecompressor().decompress(data)

    yield _LoadedSafetensors(data)

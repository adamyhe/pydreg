import numpy as np
import zstandard
from safetensors.numpy import save_file

from pydreg._safetensors_io import open_safetensors


def _write_synthetic(path, compress):
    tensors = {
        "weights": np.arange(12, dtype=np.float64).reshape(3, 4),
        "bias": np.array([1.5, -2.5], dtype=np.float64),
    }
    metadata = {"gamma": "0.5", "shape_list": "[1, 2, 3]"}
    save_file(tensors, path, metadata=metadata)

    if compress:
        with open(path, "rb") as f:
            raw = f.read()
        compressed_path = path + ".zst"
        with open(compressed_path, "wb") as f:
            f.write(zstandard.ZstdCompressor().compress(raw))
        return compressed_path, tensors, metadata
    return path, tensors, metadata


def test_open_safetensors_reads_plain_file(tmp_path):
    path, tensors, metadata = _write_synthetic(str(tmp_path / "model.safetensors"), compress=False)

    with open_safetensors(path) as f:
        assert f.metadata() == metadata
        np.testing.assert_array_equal(f.get_tensor("weights"), tensors["weights"])
        np.testing.assert_array_equal(f.get_tensor("bias"), tensors["bias"])


def test_open_safetensors_decompresses_zst_file(tmp_path):
    path, tensors, metadata = _write_synthetic(str(tmp_path / "model.safetensors"), compress=True)
    assert path.endswith(".zst")

    with open_safetensors(path) as f:
        assert f.metadata() == metadata
        np.testing.assert_array_equal(f.get_tensor("weights"), tensors["weights"])
        np.testing.assert_array_equal(f.get_tensor("bias"), tensors["bias"])


def test_open_safetensors_zst_matches_plain():
    """The .zst and plain paths must agree on both metadata and tensors --
    catches a bug where one path's header/tensor parsing silently diverges
    from the other's."""
    import tempfile
    import os

    tmpdir = tempfile.mkdtemp()
    try:
        plain_path, tensors, metadata = _write_synthetic(
            os.path.join(tmpdir, "model.safetensors"), compress=False
        )
        zst_path, _, _ = _write_synthetic(
            os.path.join(tmpdir, "model2.safetensors"), compress=True
        )

        with open_safetensors(plain_path) as f_plain, open_safetensors(zst_path) as f_zst:
            assert f_plain.metadata() == f_zst.metadata()
            for name in tensors:
                np.testing.assert_array_equal(
                    f_plain.get_tensor(name), f_zst.get_tensor(name)
                )
    finally:
        import shutil

        shutil.rmtree(tmpdir)

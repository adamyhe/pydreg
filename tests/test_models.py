import sys
import types

from pydreg import models


def test_cached_or_download_uses_existing_hf_cache(monkeypatch):
    calls = []

    def try_to_load_from_cache(repo_id, filename):
        calls.append(("cache", repo_id, filename))
        return "/tmp/cached-model.safetensors.zst"

    def hf_hub_download(**kwargs):
        raise AssertionError("hf_hub_download should not be called on cache hit")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(
            try_to_load_from_cache=try_to_load_from_cache,
            hf_hub_download=hf_hub_download,
        ),
    )

    assert (
        models._cached_or_download("repo/id", "model.safetensors.zst")
        == "/tmp/cached-model.safetensors.zst"
    )
    assert calls == [("cache", "repo/id", "model.safetensors.zst")]


def test_cached_or_download_falls_back_to_hf_download(monkeypatch):
    calls = []

    def try_to_load_from_cache(repo_id, filename):
        calls.append(("cache", repo_id, filename))
        return None

    def hf_hub_download(**kwargs):
        calls.append(("download", kwargs))
        return "/tmp/downloaded-model.safetensors.zst"

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(
            try_to_load_from_cache=try_to_load_from_cache,
            hf_hub_download=hf_hub_download,
        ),
    )

    assert (
        models._cached_or_download(
            "repo/id", "model.safetensors.zst", local_files_only=True
        )
        == "/tmp/downloaded-model.safetensors.zst"
    )
    assert calls == [
        ("cache", "repo/id", "model.safetensors.zst"),
        (
            "download",
            {
                "repo_id": "repo/id",
                "filename": "model.safetensors.zst",
                "local_files_only": True,
            },
        ),
    ]

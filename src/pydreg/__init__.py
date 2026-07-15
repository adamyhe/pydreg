"""pydreg: a python port of dREG for peak calling with PRO/GRO-seq."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pydreg")
except PackageNotFoundError:
    __version__ = "unknown"

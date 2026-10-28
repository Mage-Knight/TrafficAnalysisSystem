"""Traffic analytics on UA-DETRAC."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("traffic-analytics")
except PackageNotFoundError:
    __version__ = "0.0.0"

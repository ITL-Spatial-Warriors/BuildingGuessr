"""API service package."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("api")
except PackageNotFoundError:
    __version__ = "unknown"

"""
Thin Triton Client Library.

This package provides a minimal client interface for interacting with Triton Inference Server.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("triton-client")
except PackageNotFoundError:
    __version__ = "unknown"

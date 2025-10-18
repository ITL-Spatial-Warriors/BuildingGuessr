"""
Thin Triton Client Library.

This package provides a minimal client interface for interacting with Triton Inference Server.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("triton-client")
except PackageNotFoundError:
    __version__ = "unknown"

# Public API re-exports
from .http_client import (
    TritonHttpConfig,
    make_http_client,
    is_server_ready,
    is_model_ready,
    get_model_config,
    infer_np,
)

__all__ = [
    "__version__",
    "TritonHttpConfig",
    "make_http_client",
    "is_server_ready",
    "is_model_ready",
    "get_model_config",
    "infer_np",
]

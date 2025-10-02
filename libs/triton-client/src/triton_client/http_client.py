"""Typed helpers for NVIDIA Triton HTTP client.

This module provides a minimal, dependency-light wrapper around
``tritonclient.http.InferenceServerClient`` suitable for synchronous
MVP calls from the API service.

The helpers intentionally keep pre/post-processing out of scope. They
only translate between NumPy tensors and Triton request/response structures.
"""

from __future__ import annotations

from dataclasses import dataclass
import uuid
from typing import Any, Mapping, Sequence

import numpy as np

try:
    import tritonclient.http as http
    from tritonclient.utils import np_to_triton_dtype
except Exception as exc:  # pragma: no cover
    raise RuntimeError("tritonclient[http] must be installed to use triton-client helpers") from exc


@dataclass(frozen=True)
class TritonHttpConfig:
    """Configuration for connecting to a Triton Inference Server over HTTP.

    Attributes:
        url: Base URL for Triton server, e.g. "localhost:8000".
        verbose: Enable verbose client logging for debugging.
        connection_timeout_s: Socket connect timeout in seconds.
        network_timeout_s: Network read timeout in seconds.
    """

    url: str
    verbose: bool = False
    connection_timeout_s: float | None = None
    network_timeout_s: float | None = None


def make_http_client(config: TritonHttpConfig) -> http.InferenceServerClient:
    """Create a Triton HTTP client.

    Args:
        config: HTTP client configuration.

    Returns:
        An initialized ``InferenceServerClient``.

    Examples:
        >>> client = make_http_client(TritonHttpConfig(url="localhost:8000"))
        >>> isinstance(client.is_server_ready(), bool)
        True
    """

    return http.InferenceServerClient(
        url=config.url,
        verbose=config.verbose,
        connection_timeout=config.connection_timeout_s,
        network_timeout=config.network_timeout_s,
    )


def is_server_ready(client: http.InferenceServerClient) -> bool:
    """Return True if Triton server reports ready state."""

    return bool(client.is_server_ready())


def is_model_ready(client: http.InferenceServerClient, model_name: str) -> bool:
    """Return True if a given model is loaded and ready.

    Args:
        client: Triton HTTP client.
        model_name: Triton model name.
    """

    return bool(client.is_model_ready(model_name=model_name))


def get_model_config(client: http.InferenceServerClient, model_name: str) -> dict[str, Any]:
    """Fetch model configuration as a JSON-serializable dictionary.

    Args:
        client: Triton HTTP client.
        model_name: Triton model name.

    Returns:
        Model configuration dictionary as reported by Triton.
    """

    return client.get_model_config(model_name=model_name)


def _build_inputs(inputs: Mapping[str, np.ndarray], *, binary_data: bool) -> list[http.InferInput]:
    """Create Triton HTTP ``InferInput`` objects from numpy arrays.

    Args:
        inputs: Mapping from input name to numpy array. Arrays must be C-contiguous.
        binary_data: If True, send binary payloads for efficiency.

    Returns:
        List of initialized ``InferInput`` objects with attached data.
    """

    infer_inputs: list[http.InferInput] = []
    for name, array in inputs.items():
        if not isinstance(array, np.ndarray):  # pragma: no cover
            raise TypeError(f"Input '{name}' must be a numpy.ndarray, got {type(array)!r}")
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)

        inp = http.InferInput(name, list(array.shape), np_to_triton_dtype(array.dtype))
        inp.set_data_from_numpy(array, binary_data=binary_data)
        infer_inputs.append(inp)
    return infer_inputs


def _build_outputs(output_names: Sequence[str], *, binary_data: bool) -> list[http.InferRequestedOutput]:
    """Create Triton HTTP ``InferRequestedOutput`` objects for given names."""

    return [http.InferRequestedOutput(name, binary_data=binary_data) for name in output_names]


def infer_np(
    client: http.InferenceServerClient,
    *,
    model_name: str,
    inputs: Mapping[str, np.ndarray],
    outputs: Sequence[str],
    request_id: str | None = None,
    headers: Mapping[str, str] | None = None,
    binary_data: bool = True,
) -> dict[str, np.ndarray]:
    """Perform a synchronous inference call and return numpy outputs.

    Args:
        client: Triton HTTP client.
        model_name: Triton model name to invoke.
        inputs: Mapping of input name to numpy tensor. dtypes/shapes must match model.
        outputs: Names of requested outputs to fetch.
        request_id: Optional request identifier for tracing.
        headers: Optional HTTP headers to include.
        binary_data: Use binary wire format for inputs/outputs (recommended).

    Returns:
        Mapping of output name to numpy array.

    Raises:
        Exception: Any error propagated by the triton client.

    Examples:
        >>> import numpy as np
        >>> client = make_http_client(TritonHttpConfig(url="localhost:8000"))
        >>> _ = is_server_ready(client)  # doctest: +SKIP
        >>> y = infer_np(
        ...     client,
        ...     model_name="identity",
        ...     inputs={"INPUT": np.array([[1, 2]], dtype=np.float32)},
        ...     outputs=["OUTPUT"],
        ... )  # doctest: +SKIP
        >>> isinstance(y, dict)  # doctest: +SKIP
        True
    """

    infer_inputs = _build_inputs(inputs, binary_data=binary_data)
    infer_outputs = _build_outputs(outputs, binary_data=binary_data)

    # Triton expects 'id' field to be a string; coerce or generate one.
    req_id: str | None
    if request_id is None:
        req_id = f"req-{uuid.uuid4().hex}"
    elif isinstance(request_id, str):
        req_id = request_id
    else:
        req_id = str(request_id)

    response = client.infer(
        model_name=model_name,
        inputs=infer_inputs,
        outputs=infer_outputs,
        request_id=req_id,
        headers=dict(headers) if headers else None,
    )

    result: dict[str, np.ndarray] = {}
    for name in outputs:
        result[name] = response.as_numpy(name)
    return result

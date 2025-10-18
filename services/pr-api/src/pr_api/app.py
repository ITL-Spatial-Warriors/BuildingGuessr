"""Place Recognition (PR) API service.

Exposes a FastAPI app with a single endpoint to compute a global image
embedding for place recognition using a MegaLoc model loaded via torch.hub.
"""

from __future__ import annotations

import math
import os
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI()


class TritonTensor(BaseModel):
    """Triton-like tensor descriptor for JSON inference requests."""

    name: str
    datatype: str
    shape: list[int]
    data: Any


class TritonRequest(BaseModel):
    """Triton-like inference request body with inputs and optional parameters."""

    inputs: list[TritonTensor]
    parameters: dict[str, Any] | None = None


class TritonTensorOut(BaseModel):
    """Triton-like tensor descriptor for inference responses."""

    name: str
    datatype: str
    shape: list[int]
    data: Any


class TritonResponse(BaseModel):
    """Triton-like inference response body with outputs."""

    outputs: list[TritonTensorOut]


def _l2_normalize(vec: torch.Tensor) -> torch.Tensor:
    """L2-normalize a 1D tensor; returns original if norm is zero."""
    denom = torch.linalg.vector_norm(vec)
    if denom.item() == 0.0:
        return vec
    return vec / denom


@app.on_event("startup")
def load_model() -> None:
    """Load the MegaLoc PR model via torch.hub and store it in app state.

    The following environment variables are supported:
    - PR_DEVICE (default: "cpu")
    """
    device = os.getenv("PR_DEVICE", "cpu")

    try:
        model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
    except Exception as exc:  # pragma: no cover - import-time guard
        raise RuntimeError(
            "Failed to load PR model via torch.hub: gmberton/MegaLoc:get_trained_model"
        ) from exc

    model.eval()
    try:
        model.to(device)  # type: ignore[call-arg]
    except Exception:
        # If the model lacks .to, silently continue (some wrappers may not be nn.Module)
        pass

    app.state.model = model
    app.state.device = device


@app.post("/embed", response_model=TritonResponse)
def embed(request: TritonRequest) -> TritonResponse:
    """Compute an embedding vector from a Triton-style JSON request.

    Expects one input tensor named "IMAGE" with datatype "FP32" and shape
    [1, 3, H, W]. Values must be pre-normalized to [0, 1]. Returns a single
    output tensor named "VECTOR" with the L2-normalized embedding.
    """
    if not hasattr(app.state, "model"):
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.inputs:
        raise HTTPException(status_code=400, detail="Missing inputs")

    # Find IMAGE input (case-sensitive to match Triton conventions)
    image_input = None
    for t in request.inputs:
        if t.name == "IMAGE":
            image_input = t
            break
    if image_input is None:
        raise HTTPException(status_code=400, detail='Input tensor "IMAGE" is required')

    if image_input.datatype.upper() != "FP32":
        raise HTTPException(status_code=415, detail='Only FP32 datatype is supported for "IMAGE"')

    if len(image_input.shape) != 4 or image_input.shape[0] != 1 or image_input.shape[1] != 3:
        raise HTTPException(status_code=400, detail='"IMAGE" must have shape [1,3,H,W]')

    try:
        x = torch.as_tensor(image_input.data, dtype=torch.float32)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid IMAGE data: {exc}") from exc

    if list(x.shape) != image_input.shape:
        raise HTTPException(status_code=400, detail="Data shape does not match declared shape")

    # Move to device
    device = getattr(app.state, "device", "cpu")
    x = x.to(device)

    # Forward pass (support common callable patterns)
    try:
        out = app.state.model(x)
    except Exception:
        out = app.state.model.forward(x)  # type: ignore[attr-defined]

    if isinstance(out, torch.Tensor):
        vec_t = out.squeeze()
    elif isinstance(out, (list, tuple)) and len(out) > 0:
        vec_t = torch.as_tensor(out[0], dtype=torch.float32).squeeze()
    elif isinstance(out, dict):
        vec_t = None
        for key in ("descriptor", "embedding", "vector", "feat", "features"):
            v = out.get(key)
            if v is not None:
                vec_t = torch.as_tensor(v, dtype=torch.float32).squeeze()
                break
        if vec_t is None:
            raise HTTPException(status_code=500, detail="Model output dict lacks a known embedding key")
    else:
        raise HTTPException(status_code=500, detail="Unsupported model output type for embedding")

    # Ensure 1D float tensor on CPU
    vec_t = vec_t.detach().float().flatten().cpu()
    vec_t = _l2_normalize(vec_t)

    vector = vec_t.tolist()
    dim = len(vector)
    if not (isinstance(dim, int) and dim > 0 and math.isfinite(vec_t.norm().item())):
        raise HTTPException(status_code=500, detail="Invalid embedding produced by model")

    return TritonResponse(outputs=[TritonTensorOut(name="VECTOR", datatype="FP32", shape=[dim], data=vector)])


def main() -> None:
    """Entrypoint for running the service with uvicorn.

    Example:
        uv run pr-api  # via [project.scripts]
    """
    import uvicorn

    host = os.getenv("PR_API_HOST", "0.0.0.0")
    port = int(os.getenv("PR_API_PORT", "8080"))
    uvicorn.run("pr_api.app:app", host=host, port=port, reload=False)

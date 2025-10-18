"""CLI to send fake image tensor to 'detection' model on Triton via HTTP.

This script queries Triton for model config to infer the expected input name,
dtype, and shape. It then generates a random image tensor matching that shape
and performs a single inference, printing output keys and shapes.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from triton_client import (
    TritonHttpConfig,
    make_http_client,
    is_server_ready,
    is_model_ready,
    get_model_config,
    infer_np,
)


def _choose_input(model_cfg: dict[str, Any]) -> tuple[str, np.dtype, list[int]]:
    inputs = model_cfg.get("config", {}).get("input", []) or model_cfg.get("input", [])
    if not inputs:
        raise RuntimeError("Model config has no inputs")
    inp = inputs[0]
    name = inp["name"]
    dtype_str = inp["data_type"].replace("TYPE_", "")
    # Map Triton dtype to numpy dtype
    dtype_map = {
        "FP32": np.float32,
        "FP16": np.float16,
        "UINT8": np.uint8,
        "INT8": np.int8,
    }
    dtype = dtype_map.get(dtype_str, np.float32)
    dims = inp["dims"]
    # Replace any negative/unknown dims with a default (e.g., 224 for H/W)
    shape = [d if d > 0 else 224 for d in dims]
    return name, dtype, shape


def _choose_outputs(model_cfg: dict[str, Any]) -> list[str]:
    outputs = model_cfg.get("config", {}).get("output", []) or model_cfg.get("output", [])
    if not outputs:
        raise RuntimeError("Model config has no outputs")
    return [o["name"] for o in outputs]


def main() -> None:
    parser = argparse.ArgumentParser(description="Send fake image to Triton detection model")
    parser.add_argument("--url", default="localhost:8000", help="Triton HTTP endpoint host:port")
    parser.add_argument("--model", default="detection", help="Model name to query")
    args = parser.parse_args()

    client = make_http_client(TritonHttpConfig(url=args.url))
    if not is_server_ready(client):
        raise SystemExit("Triton server is not ready")
    if not is_model_ready(client, args.model):
        raise SystemExit(f"Model '{args.model}' is not ready")

    cfg = get_model_config(client, args.model)
    inp_name, inp_dtype, inp_shape = _choose_input(cfg)
    out_names = _choose_outputs(cfg)

    # Create fake image tensor. If model expects CHW, the config should reflect it.
    fake = np.random.rand(*inp_shape).astype(inp_dtype)
    if np.issubdtype(inp_dtype, np.integer):
        fake = (fake * 255).astype(inp_dtype)

    result = infer_np(
        client,
        model_name=args.model,
        inputs={inp_name: fake},
        outputs=out_names,
        binary_data=True,
    )

    print(f"Sent input '{inp_name}' with shape {tuple(fake.shape)}")
    for k, v in result.items():
        print(f"Output '{k}': shape={None if v is None else v.shape}, dtype={None if v is None else v.dtype}")


if __name__ == "__main__":  # pragma: no cover
    main()

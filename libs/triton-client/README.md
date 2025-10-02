# Thin Triton Client Library

Minimal typed helpers for NVIDIA Triton HTTP client.

## Install (managed by repo via uv)

The package declares `numpy` and `tritonclient[http]` as dependencies.

## Usage

```python
import numpy as np
from triton_client import (
    TritonHttpConfig,
    make_http_client,
    is_server_ready,
    is_model_ready,
    infer_np,
)

client = make_http_client(TritonHttpConfig(url="localhost:8000"))
assert is_server_ready(client)
assert is_model_ready(client, "detection")

# Example inference (shapes/dtypes must match model config)
inputs = {"IMAGE": np.random.rand(3, 224, 224).astype(np.float32)}
outputs = ["BOXES", "SCORES", "LABELS"]
result = infer_np(client, model_name="detection", inputs=inputs, outputs=outputs)
```

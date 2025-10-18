"""Detector API service.

Exposes a FastAPI app with a single endpoint to run YOLO predictions on an
uploaded image and return bounding boxes with confidences and class labels.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

from fastapi import File, FastAPI, HTTPException, UploadFile

try:
    # The ultralytics package is expected to be provided by the environment
    # (user manages dependencies with uv).
    from ultralytics import YOLOE  # type: ignore
except Exception as exc:  # pragma: no cover - import-time guard
    raise RuntimeError(
        "ultralytics is required for the detector service; please add it to dependencies"
    ) from exc


app = FastAPI()


@app.on_event("startup")
def load_model() -> None:
    """Load the YOLO model and configure class names.

    The model path and class names can be configured via environment variables:
    - MODEL_PATH: path or model name (default: "yoloe-11l-seg.pt")
    - DETECTOR_CLASS_NAMES: comma-separated names (default: "building,house")
    """
    model_path = os.getenv("MODEL_PATH", "yoloe-11l-seg.pt")
    class_names_env = os.getenv("DETECTOR_CLASS_NAMES", "building,house")
    class_names = [n.strip() for n in class_names_env.split(",") if n.strip()]

    model = YOLOE(model_path)

    # Configure classes if supported by the model interface shown in the sample
    try:
        text_pe = model.get_text_pe(class_names)
        model.set_classes(class_names, text_pe)
    except Exception:
        # If this specific API is unavailable, proceed without explicit class setup
        pass

    app.state.model = model
    app.state.class_names = class_names


@app.post("/detect")
def detect(image: UploadFile = File(...)) -> dict[str, Any]:
    """Run detection on the uploaded image and return predictions.

    Args:
        image: Uploaded image file.

    Returns:
        A JSON-compatible dict with a list of predictions. Each prediction
        contains xyxy box coordinates, confidence, class id, and class name.
    """
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Unsupported media type; expected an image")

    if not hasattr(app.state, "model"):
        raise HTTPException(status_code=503, detail="Model not loaded")

    suffix = os.path.splitext(image.filename or "upload.jpg")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(image.file.read())
        tmp_path = tmp.name

    conf = float(os.getenv("DETECT_CONF", "0.01"))

    try:
        results = app.state.model.predict(tmp_path, conf=conf, verbose=False)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    if not results:
        return {"predictions": []}

    res = results[0]

    predictions: list[dict[str, Any]] = []
    try:
        boxes_xyxy = res.boxes.xyxy.cpu().numpy().tolist()  # type: ignore[attr-defined]
        scores = res.boxes.conf.cpu().numpy().tolist()  # type: ignore[attr-defined]
        cls_ids = (
            res.boxes.cls.int().cpu().tolist()  # type: ignore[attr-defined]
            if getattr(res.boxes, "cls", None) is not None
            else [None] * len(boxes_xyxy)
        )

        for (x1, y1, x2, y2), score, cls_id in zip(boxes_xyxy, scores, cls_ids):
            class_name = None
            if cls_id is not None and hasattr(app.state, "class_names"):
                names_list = app.state.class_names
                if isinstance(cls_id, int) and 0 <= cls_id < len(names_list):
                    class_name = names_list[cls_id]

            predictions.append(
                {
                    "box": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                    "confidence": float(score),
                    "class_id": int(cls_id) if cls_id is not None else None,
                    "class_name": class_name,
                }
            )
    except Exception:
        # If the result structure differs, return an empty list rather than 500
        predictions = []

    return {"predictions": predictions}

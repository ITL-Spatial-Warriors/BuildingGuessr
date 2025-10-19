"""Detector API service.

Exposes a FastAPI app with a single endpoint to run Grounding DINO predictions on an
uploaded image and return bounding boxes with confidences and class labels.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import File, FastAPI, HTTPException, UploadFile

try:
    # Grounding DINO (transformers) and torch are required for this service.
    import torch
    from PIL import Image
    from transformers import (
        AutoProcessor,
        AutoModelForZeroShotObjectDetection,
    )
except Exception as exc:  # pragma: no cover - import-time guard
    raise RuntimeError(
        "transformers, torch and pillow are required for the detector service; please add them to dependencies"
    ) from exc


app = FastAPI()


@app.on_event("startup")
def load_model() -> None:
    """Load the Grounding DINO model and prepare text queries.

    Configurable via environment variables:
    - MODEL_ID: transformers model id (default: "IDEA-Research/grounding-dino-base")
    - DETECTOR_CLASS_NAMES: comma-separated names (default: "building,house")
    """
    model_id = os.getenv("MODEL_ID", "IDEA-Research/grounding-dino-base")
    class_names_env = os.getenv("DETECTOR_CLASS_NAMES", "building")
    class_names = [n.strip() for n in class_names_env.split(",") if n.strip()]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    def make_query(name: str) -> str:
        name_l = name.lower().strip()
        if not name_l.endswith("."):
            name_l = name_l + "."
        if not (name_l.startswith("a ") or name_l.startswith("an ") or name_l.startswith("the ")):
            name_l = "a " + name_l
        return name_l

    text_queries = [make_query(n) for n in class_names] if class_names else ["object."]

    app.state.processor = processor
    app.state.model = model
    app.state.device = device
    app.state.class_names = class_names
    app.state.text_queries = text_queries


@app.post("/detect")
def detect(image: UploadFile = File(...)) -> dict[str, Any]:
    """Run detection on the uploaded image using Grounding DINO and return predictions.

    Returns predictions as list of dicts:
      - box: dict with x1, y1, x2, y2 (pixel coords)
      - confidence: float
      - class_id: int | None (index into DETECTOR_CLASS_NAMES)
      - class_name: str | None
    """
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Unsupported media type; expected an image")

    if not hasattr(app.state, "model") or not hasattr(app.state, "processor"):
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        image.file.seek(0)
        pil_image = Image.open(image.file).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    box_conf = float(os.getenv("DETECT_CONF", "0.3"))
    text_thresh = float(os.getenv("DETECT_TEXT_THRESH", "0.3"))

    processor = app.state.processor
    model = app.state.model
    device = app.state.device
    class_names = getattr(app.state, "class_names", [])
    text = getattr(app.state, "text_queries", "object.")

    try:
        inputs = processor(images=pil_image, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_conf,
            text_threshold=text_thresh,
            target_sizes=[pil_image.size[::-1]],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    if not results:
        return {"predictions": []}

    res = results[0]

    predictions: list[dict[str, Any]] = []
    try:
        boxes = res.get("boxes", [])
        scores = res.get("scores", [])
        labels = res.get("labels", None)

        for idx, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = [float(v) for v in box]
            class_name = labels[idx]

            predictions.append(
                {
                    "box": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                    "confidence": float(score),
                    "class_id": text.index(class_name + "."),
                    "class_name": class_name,
                }
            )
    except Exception:
        predictions = []

    return {"predictions": predictions}

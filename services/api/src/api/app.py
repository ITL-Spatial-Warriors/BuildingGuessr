"""FastAPI application setup and endpoints.

Exposes health and synchronous `/locate` endpoints.
"""

from fastapi import FastAPI, Response, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time
import uuid

from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_413_REQUEST_ENTITY_TOO_LARGE,
    HTTP_415_UNSUPPORTED_MEDIA_TYPE,
)

from api import __version__
from api.schemas import LocateRequest, LocateResponse
from api.settings import get_settings
from api.storage_s3 import upload_bytes
from api.image_ops import read_and_validate, to_jpeg_bytes, image_to_numpy

app = FastAPI(title="BuildingGuessr API", version=__version__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

process_start_time = time.monotonic()


@app.get("/health", include_in_schema=False)
def health(response: Response) -> dict[str, str | float]:
    """Return service liveness with version and uptime in seconds."""
    response.headers["Cache-Control"] = "no-store"
    return {
        "status": "ok",
        "version": __version__,
        "uptime": time.monotonic() - process_start_time,
    }


@app.post("/locate", tags=["locate"], summary="Locate building/place by image", response_model=LocateResponse)
async def locate(
    file: UploadFile = File(..., description="Query image file (e.g., JPEG/PNG)"),
    topk: int = Form(3, description="Number of top results to return (>=1)"),
) -> LocateResponse:
    """Accept an image and return top-k candidate places.

    This is a synchronous MVP endpoint that validates inputs only and returns a
    stub response. The actual pipeline (decode → Triton calls → Milvus search →
    post-processing) will be implemented in subsequent steps.

    Args:
        file: Uploaded image file.
        topk: Number of results requested.

    Returns:
        A response dictionary with an empty `results` list placeholder.
    """

    # Validate request options using Pydantic model
    _ = LocateRequest(topk=topk)

    settings = get_settings()

    # Read all bytes (bounded by max size)
    try:
        raw = await file.read()
    except Exception as exc:  # pragma: no cover
        raise HTTPException(HTTP_400_BAD_REQUEST, detail="failed_to_read_file") from exc

    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(raw) > max_bytes:
        raise HTTPException(HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="file_too_large")

    # Validate and decode
    try:
        pil_img, detected_ct, _size = read_and_validate(
            raw_bytes=raw,
            max_bytes=max_bytes,
            allowed_content_types=("image/jpeg", "image/png"),
            content_type=file.content_type,
        )
    except ValueError as e:
        if str(e) == "invalid_size":
            raise HTTPException(HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="file_too_large") from e
        raise HTTPException(HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="invalid_image") from e

    # Normalize to JPEG
    jpeg_buf = to_jpeg_bytes(pil_img, quality=settings.jpeg_quality)

    # Generate ID (UUIDv4 used as time-safe default until UUIDv7 available)
    place_id = str(uuid.uuid4())
    key = f"{settings.s3_prefix}{place_id}.jpg"

    print(f"Uploading to S3: {key}")
    print(f"Bucket: {settings.s3_bucket}")
    # Upload to S3
    try:
        uri = upload_bytes(
            content=jpeg_buf,
            bucket=settings.s3_bucket,
            key=key,
            content_type="image/jpeg",
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=502, detail="s3_upload_failed") from exc

    # Prepare decoded data for downstream steps (not yet used)
    _ = image_to_numpy(pil_img)

    # Stub response for MVP
    return LocateResponse(results=[])

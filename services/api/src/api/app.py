"""FastAPI application setup and endpoints.

Exposes health and synchronous `/locate` endpoints.
"""

from fastapi import FastAPI, Response, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time
import uuid
import json
from io import BytesIO

import numpy as np

from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_413_REQUEST_ENTITY_TOO_LARGE,
    HTTP_415_UNSUPPORTED_MEDIA_TYPE,
)

from api import __version__
from api.schemas import LocateRequest, LocateResponse, LocateResult, Evidence, BBoxOnQuery
from api.settings import get_settings
from api.storage_s3 import upload_bytes
from api.image_ops import read_and_validate, to_jpeg_bytes, image_to_numpy

app = FastAPI(title="BuildingGuessr API", version=__version__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

process_start_time = time.monotonic()


def fake_pr_response(vector_size: int = 256) -> np.ndarray:
    """Generate a fake place-recognition embedding array.

    Simulates a Triton model output by returning a NumPy array of shape
    (1, `vector_size`) with dtype float32, filled with random values in the
    half-open interval [0.0, 1.0).

    Args:
        vector_size: Length of the embedding vector to generate.

    Returns:
        numpy.ndarray: Array with shape (1, vector_size) and dtype float32.

    Examples:
        >>> vec = fake_pr_response(4)
        >>> vec.shape
        (1, 4)
    """
    return np.random.rand(1, vector_size).astype(np.float32)


def fake_request_to_milvus(embedding: np.ndarray, top_k: int) -> LocateResult:
    """Simulate a Milvus vector search and return a single top result.

    Accepts an embedding of shape (1, D) and a requested `top_k` and
    returns a stubbed `LocateResult` representing the best match.

    Args:
        embedding: Query embedding array with shape (1, D).
        top_k: Number of requested nearest neighbors (unused beyond scoring stub).

    Returns:
        LocateResult: Single candidate with stubbed fields and evidence.
    """
    _ = embedding  # placeholder to acknowledge input
    k = max(int(top_k), 1)

    settings = get_settings()

    # Build a gallery URL similar to the one used in the /locate stub
    best_meme_key = "best_meme.jpeg"
    if settings.s3_endpoint_url:
        gallery_url = f"{settings.s3_endpoint_url.rstrip('/')}/{settings.s3_bucket}/{best_meme_key}"
    elif settings.s3_region:
        gallery_url = f"https://{settings.s3_bucket}.s3.{settings.s3_region}.amazonaws.com/{best_meme_key}"
    else:
        gallery_url = f"https://{settings.s3_bucket}.s3.amazonaws.com/{best_meme_key}"

    # Score/distance stubs influenced by top_k for deterministic-ish variation
    score = float(np.clip(1.0 - 0.05 * (k - 1), 0.0, 1.0))
    distance = float(np.clip(0.2 * k, 0.0, 10.0))

    candidate_id = uuid.uuid4()
    return LocateResult(
        place_id=candidate_id,
        lat=55.7517,
        lon=37.6175,
        address="Stub: Red Square, Moscow",
        score=score,
        source="place",
        evidence=Evidence(
            distance=distance,
            gallery_image_uri=gallery_url,
            query_image_uri="s3://stub/query.jpg",
            bboxes_on_query=[BBoxOnQuery(bbox=(0.0, 0.0, 1.0, 1.0), conf=0.99)],
        ),
    )


def fake_detect_buildings(img: np.ndarray, max_detections: int = 2) -> list[BBoxOnQuery]:
    """Generate fake building detections as bounding boxes on the query image.

    Produces up to `max_detections` boxes in absolute pixel coordinates with
    confidence values in [0.0, 1.0].

    Args:
        img: Query image as NumPy array with shape (H, W, C).
        max_detections: Maximum number of boxes to return.

    Returns:
        List of `BBoxOnQuery` instances.
    """
    height, width = int(img.shape[0]), int(img.shape[1])
    rng = np.random.default_rng()
    num = int(max_detections)
    results: list[BBoxOnQuery] = []
    for _ in range(num):
        x1 = float(rng.integers(0, max(1, width // 2)))
        y1 = float(rng.integers(0, max(1, height // 2)))
        x2 = float(rng.integers(int(x1) + 1, width))
        y2 = float(rng.integers(int(y1) + 1, height))
        conf = float(rng.uniform(0.7, 0.99))
        results.append(BBoxOnQuery(bbox=(x1, y1, x2, y2), conf=conf))
    return results


def fake_search_places_in_milvus(
    *,
    embedding: np.ndarray,
    top_k: int,
    query_image_uri: str,
    bboxes_on_query: list[BBoxOnQuery],
) -> list[LocateResult]:
    """Return `top_k` fake nearest neighbors for the given embedding.

    Args:
        embedding: Query embedding of shape (1, D).
        top_k: Number of neighbors to return.
        query_image_uri: URI of the uploaded query image for evidence.
        bboxes_on_query: Detector bboxes overlay for the query image.

    Returns:
        List of `LocateResult` sorted by descending score.
    """
    _ = embedding
    k = max(int(top_k), 1)
    settings = get_settings()

    best_meme_key = "best_meme.jpeg"
    if settings.s3_endpoint_url:
        gallery_url = f"{settings.s3_endpoint_url.rstrip('/')}/{settings.s3_bucket}/{best_meme_key}"
    elif settings.s3_region:
        gallery_url = f"https://{settings.s3_bucket}.s3.{settings.s3_region}.amazonaws.com/{best_meme_key}"
    else:
        gallery_url = f"https://{settings.s3_bucket}.s3.amazonaws.com/{best_meme_key}"

    base_score = 0.95
    score_decay = 0.05
    base_distance = 0.12
    distance_step = 0.08

    results: list[LocateResult] = []
    for i in range(k):
        score = float(np.clip(base_score - score_decay * i, 0.0, 1.0))
        distance = float(max(0.0, base_distance + distance_step * i))
        candidate_id = uuid.uuid4()
        lat = 55.7517 + 0.001 * i
        lon = 37.6175 + 0.001 * i
        address = f"Stub: Candidate #{i + 1}, Moscow"
        results.append(
            LocateResult(
                place_id=candidate_id,
                lat=lat,
                lon=lon,
                address=address,
                score=score,
                source="place",
                evidence=Evidence(
                    distance=distance,
                    gallery_image_uri=gallery_url,
                    query_image_uri=query_image_uri,
                    bboxes_on_query=bboxes_on_query,
                ),
            )
        )
    return results


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
    meta: UploadFile | None = File(None, description="Meta file (e.g., JSON)"),
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

    # If optional meta JSON file is provided: upload next to image and parse fields
    meta_lat: float | None = None
    meta_lon: float | None = None
    meta_angle: float | None = None
    if meta is not None:
        try:
            meta_bytes = await meta.read()
        except Exception:
            meta_bytes = b""

        # Save meta JSON side-by-side with same basename
        try:
            meta_key = f"{settings.s3_prefix}{place_id}.json"
            if meta_bytes:
                upload_bytes(
                    content=BytesIO(meta_bytes),
                    bucket=settings.s3_bucket,
                    key=meta_key,
                    content_type="application/json",
                )
        except Exception:
            # Meta is optional; ignore upload failures
            pass

        # Safely parse latitude/longitude/angle
        if meta_bytes:
            try:
                payload = json.loads(meta_bytes.decode("utf-8", errors="replace"))
                if isinstance(payload, dict):

                    def _to_float(value: object) -> float | None:
                        try:
                            if value is None:
                                return None
                            return float(value)
                        except (TypeError, ValueError):
                            return None

                    meta_lat = _to_float(payload.get("latitude"))
                    meta_lon = _to_float(payload.get("longitude"))
                    meta_angle = _to_float(payload.get("angle"))
            except Exception:
                # Ignore malformed JSON
                pass

    if meta is not None:
        print(f"Parsed meta: latitude={meta_lat}, longitude={meta_lon}, angle={meta_angle}")

    # Prepare decoded data for downstream steps
    np_img = image_to_numpy(pil_img)

    # 3) Fake building detection (returns absolute pixel bboxes)
    bboxes_on_query = fake_detect_buildings(np_img, max_detections=2)

    # 4) Fake place recognition embedding via Triton stub
    embedding = fake_pr_response(vector_size=256)

    # 5) Fake Milvus search for top-k
    results = fake_search_places_in_milvus(
        embedding=embedding,
        top_k=topk,
        query_image_uri=str(uri),
        bboxes_on_query=bboxes_on_query,
    )

    return LocateResponse(results=results)

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
from api.image_ops import read_and_validate, to_jpeg_bytes, image_to_numpy, preprocess_for_pr
import httpx
from api.vector_db import search_places

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


async def pr_embed_request(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    tensor: np.ndarray,
    shape: list[int],
    timeout_s: float,
) -> np.ndarray:
    """Call PR service `/embed` with Triton-like JSON and return (1, D) array.

    Args:
        client: Shared async HTTP client.
        base_url: PR service base URL, e.g., "http://pr-api:8080".
        tensor: Input tensor [1,3,H,W] float32 in [0,1].
        shape: Declared shape list matching tensor shape.
        timeout_s: Request timeout seconds.

    Returns:
        np.ndarray: Array of shape (1, D), dtype=float32.
    """

    payload = {
        "inputs": [
            {
                "name": "IMAGE",
                "datatype": "FP32",
                "shape": shape,
                "data": tensor.tolist(),
            }
        ],
        "parameters": {},
    }

    url = base_url.rstrip("/") + "/embed"
    try:
        resp = await client.post(url, json=payload, timeout=timeout_s)
    except httpx.ConnectTimeout as exc:
        raise HTTPException(status_code=502, detail="pr_upstream_timeout") from exc
    except httpx.ReadTimeout as exc:
        raise HTTPException(status_code=502, detail="pr_upstream_timeout") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="pr_upstream_error") from exc

    if resp.status_code == 503:
        raise HTTPException(status_code=503, detail="pr_unavailable")
    if resp.status_code >= 400:
        raise HTTPException(status_code=502, detail="pr_upstream_bad_status")

    try:
        body = resp.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail="pr_invalid_json") from exc

    outputs = body.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise HTTPException(status_code=502, detail="pr_missing_outputs")
    vector_out = None
    for out in outputs:
        if isinstance(out, dict) and out.get("name") == "VECTOR":
            vector_out = out
            break
    if vector_out is None:
        raise HTTPException(status_code=502, detail="pr_vector_not_found")

    data = vector_out.get("data")
    if not isinstance(data, list):
        raise HTTPException(status_code=502, detail="pr_invalid_vector_data")

    vec = np.asarray(data, dtype=np.float32)
    if vec.ndim != 1 or vec.size == 0:
        raise HTTPException(status_code=502, detail="pr_invalid_vector_shape")
    return vec.reshape(1, -1)


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
    meta_issues_bboxes: list[list[int]] = []
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

                    issues = payload.get("issues")
                    if isinstance(issues, list):

                        def _to_int(value: object) -> int | None:
                            try:
                                if value is None:
                                    return None
                                return int(value)
                            except (TypeError, ValueError):
                                return None

                        for issue in issues:
                            if not isinstance(issue, dict):
                                continue
                            bbox = issue.get("bbox")
                            if not isinstance(bbox, dict):
                                continue
                            x = _to_int(bbox.get("x"))
                            y = _to_int(bbox.get("y"))
                            w = _to_int(bbox.get("w"))
                            h = _to_int(bbox.get("h"))
                            if None not in (x, y, w, h):
                                meta_issues_bboxes.append([int(x), int(y), int(w), int(h)])
            except Exception:
                # Ignore malformed JSON
                pass

    if meta is not None:
        print(f"Parsed meta: latitude={meta_lat}, longitude={meta_lon}, angle={meta_angle}")
        if meta_issues_bboxes:
            print(f"Parsed issues bboxes: {meta_issues_bboxes}")

    # Prepare decoded data for downstream steps
    np_img = image_to_numpy(pil_img)

    fake_results = [
        LocateResult(
            place_id=0,
            lat=55.9517,
            lon=37.5175,
            address="Pervomayskaya 3, Dolgoprudny",
            score=0.95,
            source="place",
            evidence=Evidence(
                distance=0.12,
                gallery_image_uri="https://storage.yandexcloud.net/building-guessr-data/match1.png",
                query_image_uri="https://storage.yandexcloud.net/building-guessr-data/query.png",
                bboxes_on_query=[[0, 0, 100, 100]],
            ),
        ),
        LocateResult(
            place_id=1,
            lat=55.9517,
            lon=37.5175,
            address="Pervomayskaya 3, Dolgoprudny",
            score=0.95,
            source="place",
            evidence=Evidence(
                distance=0.12,
                gallery_image_uri="https://storage.yandexcloud.net/building-guessr-data/match2.png",
                query_image_uri="https://storage.yandexcloud.net/building-guessr-data/query.png",
                bboxes_on_query=[[0, 0, 100, 100]],
            ),
        ),
        LocateResult(
            place_id=2,
            lat=55.9517,
            lon=37.5175,
            address="Pervomayskaya 3, Dolgoprudny",
            score=0.95,
            source="place",
            evidence=Evidence(
                distance=0.12,
                gallery_image_uri="https://storage.yandexcloud.net/building-guessr-data/match3.png",
                query_image_uri="https://storage.yandexcloud.net/building-guessr-data/query.png",
                bboxes_on_query=[[0, 0, 100, 100]],
            ),
        ),
    ]

    return LocateResponse(results=fake_results)

    # 3) Fake building detection (returns absolute pixel bboxes)
    bboxes_on_query = fake_detect_buildings(np_img, max_detections=2)

    # 4) Place recognition embedding via PR API
    # Preprocess into [1,3,224,224] float32 [0,1]
    x, shape = preprocess_for_pr(pil_img, target_size=get_settings().pr_input_size)
    settings = get_settings()
    async with httpx.AsyncClient() as client:
        embedding = await pr_embed_request(
            client=client,
            base_url=settings.pr_api_url,
            tensor=x,
            shape=shape,
            timeout_s=settings.pr_api_timeout_s,
        )

    # 5) Milvus Lite search for top-k
    hits = search_places(
        db_path=settings.milvus_db_path,
        collection=settings.milvus_collection,
        vector_field=settings.milvus_vector_field,
        query_vec=embedding,
        top_k=topk,
        output_fields=("place_id", "lat", "lon", "address", "image_uri"),
    )

    results: list[LocateResult] = []
    for hit in hits:
        ent = hit.get("entity", {})
        distance = float(hit.get("distance", 0.0))
        # Convert cosine distance to similarity-like score in [0,1]
        score = float(max(0.0, min(1.0, 1.0 - distance)))
        place_id_val = ent.get("place_id")
        lat_val = ent.get("lat", 0.0)
        lon_val = ent.get("lon", 0.0)
        addr_val = ent.get("address") or ""
        gallery_uri = ent.get("image_uri") or ""
        results.append(
            LocateResult(
                place_id=place_id_val,
                lat=float(lat_val),
                lon=float(lon_val),
                address=str(addr_val),
                score=score,
                source="place",
                evidence=Evidence(
                    distance=distance,
                    gallery_image_uri=str(gallery_uri),
                    query_image_uri=str(uri),
                    bboxes_on_query=bboxes_on_query,
                ),
            )
        )

    return LocateResponse(results=results)

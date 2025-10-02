"""Pydantic schemas for API requests and responses.

This module contains request/response models used by public API endpoints.
"""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class LocateRequest(BaseModel):
    """Request options for the synchronous `/locate` endpoint.

    Args:
        topk: Number of top results to return. Must be a positive integer.

    Examples:
        >>> LocateRequest()  # default top-k
        LocateRequest(topk=3)
        >>> LocateRequest(topk=5)
        LocateRequest(topk=5)
    """

    topk: int = Field(
        default=3,
        ge=1,
        description="Number of top results to return (must be >= 1).",
    )


class BBoxOnQuery(BaseModel):
    """Detected bounding box on the query image.

    Args:
        bbox: Bounding box coordinates in absolute pixels (x1, y1, x2, y2).
        conf: Detector confidence in [0.0, 1.0].
    """

    bbox: tuple[float, float, float, float]
    conf: float = Field(ge=0.0, le=1.0)


class Evidence(BaseModel):
    """Supporting evidence for a single result item.

    Args:
        distance: Raw vector distance/similarity metric used for ranking.
        gallery_image_uri: URI of the matched gallery image (e.g., s3://... ).
        query_image_uri: URI of the uploaded query image (e.g., s3://... ).
        bboxes_on_query: Detector bboxes overlaid on the query image.
    """

    distance: float = Field(ge=0.0)
    gallery_image_uri: str
    query_image_uri: str
    bboxes_on_query: list[BBoxOnQuery] = Field(default_factory=list)


class LocateResult(BaseModel):
    """Single candidate location returned by `/locate`.

    Args:
        place_id: Identifier of the candidate place (UUID).
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        address: Human-readable address string for the predicted location.
        score: Normalized confidence score in [0.0, 1.0].
        source: Which subsystem produced the candidate; MVP uses "place".
        evidence: Supporting evidence used to produce this candidate.
    """

    place_id: UUID
    lat: float
    lon: float
    address: str
    score: float = Field(ge=0.0, le=1.0)
    source: Literal["place"]
    evidence: Evidence


class LocateResponse(BaseModel):
    """Response body for `/locate`.

    Args:
        results: Ranked list of candidate locations.
    """

    results: list[LocateResult] = Field(default_factory=list)

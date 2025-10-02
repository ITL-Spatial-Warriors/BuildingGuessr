"""Pydantic schemas for API requests and responses.

This module contains request/response models used by public API endpoints.
"""

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

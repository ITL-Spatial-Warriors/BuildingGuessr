"""Thin S3 client helper for uploading image bytes."""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from typing import Any

import boto3
from botocore.client import BaseClient
from botocore.config import Config as BotoConfig

from api.settings import get_settings


@lru_cache(maxsize=1)
def get_s3_client() -> BaseClient:
    """Return a cached Boto3 S3 client configured from settings.

    Returns:
        BaseClient: boto3 S3 client instance
    """

    settings = get_settings()
    config = BotoConfig(region_name=settings.s3_region) if settings.s3_region else None
    if settings.s3_endpoint_url:
        return boto3.client("s3", endpoint_url=settings.s3_endpoint_url, config=config)
    return boto3.client("s3", config=config)


def upload_bytes(
    *,
    content: BytesIO,
    bucket: str,
    key: str,
    content_type: str,
    extra_args: dict[str, Any] | None = None,
) -> str:
    """Upload in-memory bytes to S3 and return the object URI.

    Args:
        content: BytesIO positioned at the beginning.
        bucket: Target S3 bucket name.
        key: Object key (path inside bucket).
        content_type: MIME content type of the object.
        extra_args: Additional ExtraArgs for upload (ACL, SSE, etc.).

    Returns:
        str: s3 URI (e.g., s3://bucket/key)
    """

    s3 = get_s3_client()
    content.seek(0)
    args: dict[str, Any] = {"ContentType": content_type, "ACL": "private"}

    settings = get_settings()
    if settings.s3_sse:
        args["ServerSideEncryption"] = settings.s3_sse
    if settings.s3_sse_kms_key_id:
        args["SSEKMSKeyId"] = settings.s3_sse_kms_key_id

    if extra_args:
        args.update(extra_args)

    s3.upload_fileobj(content, bucket, key, ExtraArgs=args)
    return f"s3://{bucket}/{key}"

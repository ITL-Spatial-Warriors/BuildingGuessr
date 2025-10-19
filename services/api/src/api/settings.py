"""Service settings loaded from environment variables.

Provides a cached accessor for configuration relevant to image intake and S3.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os


@dataclass(frozen=True)
class Settings:
    """Immutable application settings.

    Attributes:
        s3_bucket: Target S3 bucket to store original images.
        s3_region: AWS region for the S3 client.
        s3_prefix: Key prefix for stored objects (e.g., "places/").
        s3_endpoint_url: Custom S3 endpoint (optional, e.g., for localstack).
        s3_sse: Server-side encryption algorithm (e.g., "AES256").
        s3_sse_kms_key_id: KMS key id/arn when using SSE-KMS.
        max_upload_mb: Maximum upload size in megabytes.
        jpeg_quality: JPEG quality for normalized images (1-100).
        pr_api_url: Base URL of the Place Recognition service (e.g., http://pr-api:8080).
        pr_api_timeout_s: Timeout in seconds for PR API requests.
        pr_input_size: Target square size (pixels) for PR model input (e.g., 224).
        milvus_db_path: Local path to Milvus Lite DB file (e.g., ./moscow2019_MegaLoc.db).
        milvus_collection: Collection name for places (e.g., "places").
        milvus_vector_field: Name of the vector field (e.g., "vec").
    """

    s3_bucket: str
    s3_region: str | None
    s3_prefix: str
    s3_endpoint_url: str | None
    s3_sse: str | None
    s3_sse_kms_key_id: str | None
    max_upload_mb: int
    jpeg_quality: int
    pr_api_url: str
    pr_api_timeout_s: float
    pr_input_size: int
    milvus_db_path: str
    milvus_collection: str
    milvus_vector_field: str


def _get_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:  # pragma: no cover
        raise RuntimeError(f"Invalid integer for {name}: {value}") from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings from environment, validate, and cache the result.

    Raises:
        RuntimeError: If required variables are missing or invalid.

    Returns:
        Settings: Frozen settings instance.
    """

    s3_bucket = os.environ.get("S3_BUCKET")
    if not s3_bucket:
        raise RuntimeError("S3_BUCKET is required")

    return Settings(
        s3_bucket=s3_bucket,
        s3_region=os.environ.get("S3_REGION"),
        s3_prefix=os.environ.get("S3_PREFIX", "places/"),
        s3_endpoint_url=os.environ.get("S3_ENDPOINT_URL"),
        s3_sse=os.environ.get("S3_SSE"),
        s3_sse_kms_key_id=os.environ.get("S3_SSE_KMS_KEY_ID"),
        max_upload_mb=_get_env_int("MAX_UPLOAD_MB", 5),
        jpeg_quality=_get_env_int("JPEG_QUALITY", 90),
        pr_api_url=os.environ.get("PR_API_URL", "http://pr-api:8080"),
        pr_api_timeout_s=float(os.environ.get("PR_API_TIMEOUT_S", "5")),
        pr_input_size=_get_env_int("PR_INPUT_SIZE", 224),
        milvus_db_path=os.environ.get("MILVUS_DB_PATH", "./moscow2019_MegaLoc.db"),
        milvus_collection=os.environ.get("MILVUS_COLLECTION", "places"),
        milvus_vector_field=os.environ.get("MILVUS_VECTOR_FIELD", "vec"),
    )

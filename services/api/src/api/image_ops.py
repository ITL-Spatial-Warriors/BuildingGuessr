"""Image validation and normalization utilities."""

from __future__ import annotations

from io import BytesIO
from typing import Iterable, Tuple

import numpy as np
from PIL import Image, ImageOps


def read_and_validate(
    *,
    raw_bytes: bytes,
    max_bytes: int,
    allowed_content_types: Iterable[str],
    content_type: str | None,
) -> tuple[Image.Image, str, int]:
    """Validate uploaded image and return decoded PIL image.

    Args:
        raw_bytes: Raw file bytes as provided by the client.
        max_bytes: Maximum allowed size in bytes.
        allowed_content_types: Allowed MIME types.
        content_type: Reported MIME type (may be None).

    Returns:
        (pil_image, detected_content_type, size_bytes)
    """

    size = len(raw_bytes)
    if size == 0 or size > max_bytes:
        raise ValueError("invalid_size")

    ct = (content_type or "").lower()
    if ct not in {c.lower() for c in allowed_content_types}:
        # We still try to decode to differentiate bad type vs bad content later.
        pass

    try:
        with Image.open(BytesIO(raw_bytes)) as img:
            img.load()
            # Apply EXIF orientation and convert to RGB for consistency
            img = ImageOps.exif_transpose(img)
            pil = img.convert("RGB")
    except Exception as exc:  # pragma: no cover
        raise ValueError("invalid_image") from exc

    # Prefer actual decoded type over header when possible
    detected_ct = "image/jpeg" if pil.format == "JPEG" else (ct or "application/octet-stream")
    return pil, detected_ct, size


def to_jpeg_bytes(img: Image.Image, *, quality: int = 90) -> BytesIO:
    """Encode PIL image into JPEG bytes with given quality.

    Args:
        img: PIL image (assumed RGB).
        quality: JPEG quality (1-100).

    Returns:
        BytesIO with JPEG payload positioned at 0.
    """

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=int(quality), optimize=True)
    buf.seek(0)
    return buf


def image_to_numpy(img: Image.Image) -> np.ndarray:
    """Convert PIL image to NumPy array (H, W, C), dtype=uint8."""

    return np.asarray(img)


def preprocess_for_pr(img: Image.Image, target_size: int = 224) -> tuple[np.ndarray, list[int]]:
    """Preprocess image into PR model input tensor.

    Converts a PIL RGB image to a float32 tensor with values in [0, 1], shape
    [1, 3, target_size, target_size] (NCHW), suitable for Triton-like JSON.

    Args:
        img: PIL image in RGB mode.
        target_size: Target square size (pixels), e.g., 224.

    Returns:
        Tuple of (tensor, shape), where tensor is np.ndarray dtype=float32 with
        shape [1, 3, target_size, target_size], and shape is the same as a list
        for inclusion in JSON payloads.
    """

    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize to target_size x target_size using bilinear resampling
    resized = img.resize((int(target_size), int(target_size)), Image.BILINEAR)

    # To numpy HWC uint8, then normalize to [0,1] float32
    hwc = np.asarray(resized, dtype=np.uint8)
    x = hwc.astype(np.float32) / 255.0

    # HWC -> CHW
    chw = np.transpose(x, (2, 0, 1))

    # Add batch dimension: NCHW
    nchw = np.expand_dims(chw, axis=0)

    shape = [1, 3, int(target_size), int(target_size)]
    return nchw, shape

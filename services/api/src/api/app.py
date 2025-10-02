"""FastAPI application setup and endpoints.

Exposes health and synchronous `/locate` endpoints.
"""

from fastapi import FastAPI, Response, UploadFile, File, Form
import time

from api import __version__
from api.schemas import LocateRequest, LocateResponse

app = FastAPI(title="BuildingGuessr API", version=__version__)

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

    # TODO: implement validation of file content-type/size and the full pipeline
    return LocateResponse(results=[])

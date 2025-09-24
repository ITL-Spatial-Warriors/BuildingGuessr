"""FastAPI application setup and health endpoint."""

from fastapi import FastAPI, Response
import time

from api import __version__

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

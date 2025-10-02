from __future__ import annotations

import os

import uvicorn


def main() -> None:
    """Run the detector FastAPI app with uvicorn.

    Honors PORT and HOST environment variables if provided.
    """
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload_flag = os.getenv("RELOAD", "false").lower() == "true"

    uvicorn.run("detector_api.app:app", host=host, port=port, reload=reload_flag)

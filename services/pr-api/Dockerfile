FROM nvcr.io/nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y --no-install-recommends python3.12 python3.12-venv python3.12-distutils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN mkdir -p services/pr-api

# Prime third-party deps using the service's project context
# Copy only lock and project files to leverage Docker layer caching
COPY uv.lock pyproject.toml /app/
COPY services/pr-api/pyproject.toml /app/services/pr-api/pyproject.toml
ENV UV_LINK_MODE=copy
WORKDIR /app/services/pr-api
RUN --mount=type=cache,target=/root/.cache/uv uv sync --no-install-workspace

# Copy the repository and install the workspace + project
WORKDIR /app
ADD . /app
WORKDIR /app/services/pr-api
RUN --mount=type=cache,target=/root/.cache/uv uv sync --no-editable


FROM nvcr.io/nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y --no-install-recommends python3.12 python3.12-venv python3.12-distutils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Expose API port
EXPOSE 8080

# Run uvicorn using the installed environment
CMD ["uvicorn", "pr_api.app:app", "--app-dir", "services/pr-api/src", "--host", "0.0.0.0", "--port", "8080"]

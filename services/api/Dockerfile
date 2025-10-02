FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
RUN mkdir -p services/api

# Prime third-party deps using the service's project context
# Copy only lock and project files to leverage Docker layer caching
COPY uv.lock pyproject.toml /app/
COPY services/api/pyproject.toml /app/services/api/pyproject.toml
ENV UV_LINK_MODE=copy
WORKDIR /app/services/api
RUN --mount=type=cache,target=/root/.cache/uv uv sync --no-install-workspace

# Copy the repository and install the workspace + project
WORKDIR /app
ADD . /app
WORKDIR /app/services/api
RUN --mount=type=cache,target=/root/.cache/uv uv sync --no-editable

FROM python:3.12-slim
WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Expose API port
EXPOSE 8000

# Run uvicorn using the installed environment
CMD ["uvicorn", "api.app:app", "--app-dir", "services/api/src", "--host", "0.0.0.0", "--port", "8000"]

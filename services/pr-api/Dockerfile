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

# Bring uv into the image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Non-interactive apt
ENV DEBIAN_FRONTEND=noninteractive

# Install build deps and build Python 3.12 from source (Ubuntu 22.04 lacks ready 3.12)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential wget curl ca-certificates git \
       zlib1g-dev libssl-dev libbz2-dev libreadline-dev libsqlite3-dev \
       libffi-dev libncursesw5-dev libgdbm-dev liblzma-dev tk-dev uuid-dev \
    && rm -rf /var/lib/apt/lists/* \
    && wget -q https://www.python.org/ftp/python/3.12.5/Python-3.12.5.tgz \
    && tar -xzf Python-3.12.5.tgz \
    && cd Python-3.12.5 \
    && ./configure --enable-optimizations --with-ensurepip=install \
    && make -j"$(nproc)" \
    && make altinstall \
    && ln -sf /usr/local/bin/python3.12 /usr/local/bin/python3 \
    && ln -sf /usr/local/bin/python3.12 /usr/local/bin/python \
    && cd /app \
    && rm -rf Python-3.12.5 Python-3.12.5.tgz

# Prime deps with caching using uv
RUN mkdir -p /app/services/pr-api
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

# Activate the venv by default
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Expose API port
EXPOSE 8080

# Run uvicorn using the installed environment
CMD ["uvicorn", "pr_api.app:app", "--app-dir", "services/pr-api/src", "--host", "0.0.0.0", "--port", "8080"]

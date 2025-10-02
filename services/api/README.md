# API service

FastAPI entrypoint for BuildingGuessr.

- Run with Docker Compose (from repo root; see `docker-compose.yml`):
  - `docker compose up --build api`

- Endpoints:
  - `GET /health` — liveness, version, uptime.
  - `POST /locate` — multipart `file`, form `topk?=3`; returns `{ results: [...] }` (MVP stub).

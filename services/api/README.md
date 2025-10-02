# API service

FastAPI entrypoint for BuildingGuessr.

- Run with Docker Compose (from repo root; see `docker-compose.yml`):
  - `docker compose up --build api`

- Endpoints:
  - `GET /health` — liveness, version, uptime.
  - `POST /locate` — multipart `file`, form `topk?=3`; returns `{ results: [...] }` (MVP stub).

- OpenAPI docs endpoints:
  - `GET /docs` — Swagger UI (OpenAPI).
  - `GET /redoc` — ReDoc.
  - `GET /openapi.json` — OpenAPI schema.

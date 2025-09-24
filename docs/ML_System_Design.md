# ML System Design — MVP

> ⚠️ WORK IN PROGRESS

> **Scope (MVP):** сервис определяет координаты зданий по фотографиям для Москвы/МО. Целевая точность MVP: средняя ошибка ≤ **50 м**. Фокус — скорость разработки + база для масштабирования.

---

## 1) Overview

* **Пользовательская задача:** загрузить фото → получить список кандидатов (координаты/адреса) с оценками уверенности.
* **Высокоуровневая схема:** API → Orchestrator → \[Detection, Place Rec, Building Rec] → Milvus ↔ MinIO → Ранжирование → Результат.

## 2) Goals / Non‑goals

* **Goals:** работающий E2E пайплайн; топ‑k кандидатов; карта/каталог; средняя ошибка ≤ 50 м.
* **Non‑goals (MVP):** онлайн‑обучение, автоскейл в облаке, сложная авторизация, полноценный MLOps‑контур.

## 3) Requirements (TBD)

* **Качество:**

  * Геоточность: средняя ошибка ≤ 50 м (основная метрика MVP).
  * Offline: Recall\@k для Place/Building; сведение к финальной гео‑ошибке.
* **SLA (TBD):** p50/p95 латентности на 1 изображение; целевой throughput.
* **Ресурсы (TBD):** 1×GPU для инференса, объёмы памяти/диска.

## 4) Data & Storage (TBD)

* **S3‑совместимое:** `s3://bucket/places/{place_id}.jpg`.
* **Решение:** кропы зданий **не храним**; делаем **динамический кроп** по bbox на лету.
* **Milvus (векторное хранилище):**

  * Коллекция `Places`: `place_id (PK)`, `vec`, `lat`, `lon`, `yaw?`, `image_uri`, `city`, `ts`, `extras(json)`.
  * Коллекция `Buildings`: `building_id (PK)`, `place_id`, `bbox`, `vec`, `lat`, `lon`, `address?`, `extras(json)`.

## 5) System Architecture

* **Сервисы:**

  1. **API & Ingestion (FastAPI)** — приём запросов (file/URL), загрузка в **MinIO**, создание `job_id`, публикация задачи в очередь (**Redis**).
  2. **Orchestrator / Ranking (worker)** — тянет задачи из очереди; делает **препроцессинг** (декод, resize/normalize, кропы по bbox), вызывает **Triton** (HTTP/gRPC) для `detection`, `place_embed`, `building_embed`; выполняет **поиск/апсерты в Milvus**, **слияние/ранжирование** результатов и формирование ответа/метаданных.
  3. **Triton Inference Server** — единый сервер инференса моделей:

     * `model_repository`: модели `detection`, `place_embed`, `building_embed`;
     * порты: **HTTP :8000**, **gRPC :8001**, **Prometheus :8002**.

* **Зависимости:** **Redis** (очередь/кэш), **Milvus** (векторные коллекции и скалярные поля), **MinIO** (S3‑совместимое хранилище). *(Опционально: Prometheus/Grafana для метрик Triton.)*

* **Доступ к Milvus:** на MVP — общая Python‑библиотека `libs/vecdb` (CRUD + search). В перспективе — выделенный **Vector Index Service**.

* **Внутренние общие библиотеки:** `libs/triton_client` (тонкий клиент Triton), `libs/image_ops` (декод/resize/кропы), `libs/ranking` (фьюжн/эвристики), `libs/storage` (MinIO), `libs/contracts` (схемы Pydantic), `libs/settings` (конфиги).

* **Границы ответственности:**

  * **Triton** — только инференс моделей (без бизнес‑логики и доступа к БД/хранилищам).
  * **Orchestrator** — препроцессинг/постпроцессинг, работа с БД/хранилищами, бизнес‑правила, ретраи/таймауты/идемпотентность.

## 6) Inference Path (E2E)

1. API принимает фото/URL, кладёт в MinIO, создаёт `job_id`, ставит задачу в Redis.
2. Orchestrator: **Detection** → bbox; параллельно **PlaceRec(embed)** и **BuildingRec(embed)** для кропов; поиск в Milvus по Places/Buildings.
3. **Фьюжн/ранжирование:** объединение Place/Building кандидатов, гео‑фильтры/эвристики → финальный топ‑k.
4. Запись результатов/метаданных; `GET /jobs/{job_id}` отдаёт статус и список кандидатов (координаты, score, evidence).

## 7) API Contracts (TBD)

* **Public API:**

  * `POST /jobs/locate` → `{ job_id }`
    body: `{ file | url | s3_uri, options?: { topk?: int } }`
  * `GET /jobs/{job_id}` → `{ status: queued|running|done|error, results?: [{ lat, lon, score, source, evidence }] }`
* **ML‑сервисы:**

  * Detection: `POST /infer` → `[ { bbox:[x1,y1,x2,y2], conf:float } ]`
  * Embedding (place/building): `POST /embed` → `{ vector:[...], dim:int }`
* **ID:** `job_id`, `place_id`, `building_id` — UUIDv4/7.

## 8) Search & Index (Milvus) TBD

* **Метрика/индекс (TBD):** cosine/IP; IVF/HNSW, параметры (ef/nprobe).
* **Размерности (TBD):** dim\_place, dim\_building.
* **top‑k по умолчанию:** 20 (можно менять в options).

## 9) Reliability & Security

* **Идемпотентность** по `job_id`, **ретраи** с backoff, **таймауты** на вызовы ML‑сервисов.
* **Auth (MVP):** API‑key; приватный бакет S3; ограниченные роли в Milvus.

## 10) Monitoring (TBD)

* Метрики: латентность по шагам, error‑rate, доля no‑match, распределение скор/дистанций.
* Логи: JSON с `job_id`/`trace_id`; `/health` у всех сервисов.

## 11) Versions & Artifacts (TBD)

* `model_version` (place/building), `schema_version` (Milvus), `index_version`.

## 12) Risks & Open Questions

* Параметры Milvus (индекс/метрика) — **TBD**.
* Строгие SLA p95/p99 — **TBD**.
* Источники/лицензии галереи (Москва/МО) — **TBD**.
* Стратегия фьюжна и пороги отсечки — **TBD**.

## 13) Roadmap (после MVP) TBD

* Выделить Vector Index Service; добавить Geocoder/Reverse‑geocoder; Admin UI; сбор inference‑логов для ретрейнинга; канареечные выкладки.

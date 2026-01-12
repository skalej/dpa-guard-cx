# DPA Guard MVP Scaffold

This repository contains the initial scaffold for the DPA Guard MVP. It wires the monorepo layout, services, and stubs only (no LLM logic or contract analysis yet).

## Structure
- `frontend/`: Next.js skeleton
- `backend/`: FastAPI API skeleton + SQLAlchemy + Alembic
- `worker/`: Celery worker skeleton
- `docker-compose.yml`: Local dev stack (Postgres, Redis, MinIO, API, worker, frontend)

## Local dev (Docker)
1. Copy env file:
   ```bash
   cp .env.example .env
   ```
2. Build and run:
   ```bash
   docker compose up --build
   ```
3. API health:
   - `GET http://localhost:8000/health/live`
   - `GET http://localhost:8000/health/ready`

## Local dev (without Docker)
- Backend:
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install -r backend/requirements.txt
  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
  ```
- Worker:
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install -r worker/requirements.txt
  celery -A app.celery_app worker --loglevel=INFO
  ```
- Frontend:
  ```bash
  cd frontend
  npm install
  npm run dev
  ```

## Notes
- Reviews flow stores metadata, uploads to MinIO, and writes placeholder results.
- Do not log raw contract text; log only IDs/status/metadata.

## Common commands
- `make up` / `make down`
- `make db-migrate`

## Smoke test (reviews flow)
```bash
API=http://localhost:8000
REVIEW_ID=$(curl -s -X POST "$API/reviews" -H "Content-Type: application/json" -d '{"context": {"project":"demo"}, "vendor_name":"Acme"}' | python -c 'import json,sys; print(json.load(sys.stdin)[\"id\"])')

curl -s -X POST "$API/reviews/$REVIEW_ID/upload" \\
  -F "file=@/path/to/contract.pdf" \\
  | python -m json.tool

curl -s -X POST "$API/reviews/$REVIEW_ID/start" | python -m json.tool

# Poll until completed
curl -s "$API/reviews/$REVIEW_ID" | python -m json.tool
curl -s "$API/reviews/$REVIEW_ID/results" | python -m json.tool

# Extraction metadata
curl -s "$API/reviews/$REVIEW_ID/text" | python -m json.tool
```

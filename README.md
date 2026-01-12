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
- Review endpoints are stubs and return HTTP 501.
- Do not log raw contract text; log only IDs/status/metadata.

## Common commands
- `make up` / `make down`
- `make db-migrate`

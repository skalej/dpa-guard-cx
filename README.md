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
  Set `NEXT_PUBLIC_API_BASE` (default `http://127.0.0.1:8000`). Open `/reviews/new` to start, `/reviews` to see recent local reviews, and `/reviews/{id}` for details.

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

# Results include evidence quotes with spans
curl -s "$API/reviews/$REVIEW_ID/results" | python -m json.tool

# Explain findings with evidence + playbook guidance + negotiation asks
curl -s "$API/reviews/$REVIEW_ID/explain" | python -m json.tool

# Export PDF (signed URL may expire; regenerate if needed)
curl -s "$API/reviews/$REVIEW_ID/export/pdf" | python -m json.tool
```

PDF export smoke test (local):
```bash
PYTHONPATH=backend python backend/tests/test_pdf_export.py
```

Note: For meaningful playbook results, upload a real DPA/contract PDF (not PRD/TDD docs).
For best playbook retrieval from PDF/text, structure sections with headings like `DPA-BR-01 – Breach Notification` or clear plain headings such as `Processor Obligations` and `Confidentiality`.

## Frontend UX manual check
1. Start the frontend and backend, then open `http://localhost:3000/reviews/new`.
2. Upload a PDF and select a playbook (or enter a playbook ID).
3. After redirect, verify the detail page shows status, exec summary, risk table, and explain panels.
4. Click "Download PDF" and open the presigned URL in a new tab (regenerate if expired).

PDF export:
```bash
curl -s "$API/reviews/$REVIEW_ID/export/pdf" | python -m json.tool
# If no URL returned, open MinIO console at http://localhost:9001 and download from bucket dpa-guard.
```

Set `S3_PUBLIC_ENDPOINT` (e.g., `http://127.0.0.1:9000`) so presigned URLs are browser-accessible from the host. Use `S3_INTERNAL_ENDPOINT` for in-network access (e.g., `http://minio:9000`).

## Playbook RAG smoke test
```bash
export RAG_ENABLED=true
export OPENAI_API_KEY=your_key
PB_ID=$(curl -s -X POST "$API/playbooks/upload" -F "file=@backend/app/playbooks/eu_controller_v0_1.yaml" | python -c 'import json,sys; print(json.load(sys.stdin)[\"id\"])')
sleep 2
curl -s "$API/playbook/versions" | python -m json.tool
curl -s "$API/playbooks/$PB_ID/search?q=breach%20notification&k=3" | python -m json.tool
# Reindex if you update a playbook or want to apply section-aware chunking
curl -s -X POST "$API/playbooks/$PB_ID/reindex" | python -m json.tool
```

## LLM summary (optional)
Set `LLM_ENABLED=true` and `OPENAI_API_KEY`. Only evidence quotes and playbook chunks are sent.
```bash
curl -s -X POST "$API/reviews/$REVIEW_ID/rerun_llm" | python -m json.tool
```

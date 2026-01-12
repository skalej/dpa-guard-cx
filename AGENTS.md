# DPA Guard (MVP) — Codex Instructions

## Ground truth docs
- Follow docs/prd.pdf and docs/tdd.pdf as the source of truth for scope and architecture.
- MVP outputs: exec summary, clause-by-clause risk table with evidence quotes, negotiation pack, PDF export.

## Non-negotiables (Trust & Safety)
- Evidence-first: never flag an issue without exact quotes; validate quotes by deterministic substring/span matching.
- Low hallucination: no invented clauses, no invented citations, no “assumed” wording.
- Never log raw contract text in application logs (only IDs, step status, and safe metadata).
- If PDF is likely scanned (low text density), fail gracefully: “OCR not supported in MVP.”

## MVP architecture (preferred)
- Frontend: Next.js
- Backend API: FastAPI
- Worker: Celery or RQ with Redis queue
- DB: Postgres
- Object storage: S3-compatible (uploads + generated PDFs)

## MVP endpoints (from TDD)
- POST /reviews (create review + minimal context + vendor_name)
- POST /reviews/{id}/upload (multipart upload)
- POST /reviews/{id}/start (enqueue processing)
- GET /reviews/{id} (metadata + status)
- GET /reviews/{id}/results (exec summary + table)
- GET /reviews/{id}/export/pdf (signed URL or streamed)
- GET /playbook/versions

## Working agreements
- Prefer small, reviewable commits.
- Keep changes minimal and testable.
- When adding dependencies, choose stable, common libs.
- Add README instructions + a local dev path (docker compose or simple local run) as early as possible.

## Definition of done for each task
- Code compiles/runs locally.
- Basic tests or smoke steps documented.
- No raw contract text in logs.


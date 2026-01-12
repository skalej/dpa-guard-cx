.PHONY: up down logs api worker web db-migrate

up:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f

api:
	cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

worker:
	cd worker && celery -A app.celery_app worker --loglevel=INFO

web:
	cd frontend && npm run dev

db-migrate:
	cd backend && PYTHONPATH=. alembic upgrade head

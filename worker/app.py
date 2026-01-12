import os

from celery import Celery

redis_url = os.getenv("DPA_REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery("dpa_guard", broker=redis_url, backend=redis_url)


@celery_app.task
def placeholder_task():
    return {"status": "not_implemented"}

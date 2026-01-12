import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.celery_app import celery_app
from app.core.config import settings
from app.core.s3 import get_s3_client
from app.db.session import get_db
from app.models.review import Review

router = APIRouter()

ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


class CreateReviewRequest(BaseModel):
    context: dict[str, Any] | None = None
    vendor_name: str | None = None


class CreateReviewResponse(BaseModel):
    id: uuid.UUID
    status: str
    created_at: datetime


@router.post("/reviews", response_model=CreateReviewResponse)
def create_review(payload: CreateReviewRequest, db: Session = Depends(get_db)):
    review = Review(
        status="draft",
        context_json={"context": payload.context, "vendor_name": payload.vendor_name},
    )
    db.add(review)
    db.commit()
    db.refresh(review)
    return CreateReviewResponse(id=review.id, status=review.status, created_at=review.created_at)


@router.post("/reviews/{review_id}/upload")
def upload_review(
    review_id: uuid.UUID,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    review = db.scalar(select(Review).where(Review.id == review_id))
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")

    safe_name = Path(file.filename or "upload").name
    object_key = f"reviews/{review_id}/source/{uuid.uuid4()}_{safe_name}"

    s3 = get_s3_client()
    s3.upload_fileobj(file.file, settings.s3_bucket, object_key)

    review.source_object_key = object_key
    review.source_filename = safe_name
    review.source_mime = file.content_type
    review.error_message = None
    db.add(review)
    db.commit()
    db.refresh(review)

    return {
        "id": str(review.id),
        "status": review.status,
        "source_filename": review.source_filename,
        "source_object_key": review.source_object_key,
    }


@router.post("/reviews/{review_id}/start")
def start_review(review_id: uuid.UUID, db: Session = Depends(get_db)):
    review = db.scalar(select(Review).where(Review.id == review_id))
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    if not review.source_object_key:
        raise HTTPException(status_code=400, detail="Source file not uploaded")

    review.status = "processing"
    review.error_message = None
    db.add(review)
    db.commit()

    celery_app.send_task("process_review", args=[str(review_id)])

    return {"id": str(review.id), "status": review.status}


@router.get("/reviews/{review_id}")
def get_review(review_id: uuid.UUID, db: Session = Depends(get_db)):
    review = db.scalar(select(Review).where(Review.id == review_id))
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")

    return {
        "id": str(review.id),
        "status": review.status,
        "source_filename": review.source_filename,
        "source_object_key": review.source_object_key,
        "source_mime": review.source_mime,
        "error_message": review.error_message,
        "created_at": review.created_at,
        "updated_at": review.updated_at,
    }


@router.get("/reviews/{review_id}/results")
def get_results(review_id: uuid.UUID, db: Session = Depends(get_db)):
    review = db.scalar(select(Review).where(Review.id == review_id))
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    if review.status != "completed" or review.results_json is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Results not ready")

    return review.results_json


@router.get("/reviews/{review_id}/export/pdf")
def export_pdf(review_id: uuid.UUID):
    raise HTTPException(status_code=501, detail="Not implemented in MVP scaffold")


@router.get("/playbook/versions")
def playbook_versions():
    raise HTTPException(status_code=501, detail="Not implemented in MVP scaffold")

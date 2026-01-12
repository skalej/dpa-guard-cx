import io
import uuid
from datetime import datetime, timezone
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.core.celery_app import celery_app
from app.core.config import settings
from app.core.s3 import get_internal_s3_client, get_presign_s3_client
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
    playbook_id: uuid.UUID | None = None


class CreateReviewResponse(BaseModel):
    id: uuid.UUID
    status: str
    created_at: datetime


@router.post("/reviews", response_model=CreateReviewResponse)
def create_review(payload: CreateReviewRequest, db: Session = Depends(get_db)):
    review = Review(
        status="draft",
        context_json={"context": payload.context, "vendor_name": payload.vendor_name},
        playbook_id=payload.playbook_id,
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

    s3 = get_internal_s3_client()
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
        "playbook_id": str(review.playbook_id) if review.playbook_id else None,
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


@router.get("/reviews/{review_id}/rag")
def get_rag(review_id: uuid.UUID, db: Session = Depends(get_db)):
    # TODO: protect this endpoint (internal/dev only).
    review = db.scalar(select(Review).where(Review.id == review_id))
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    if not review.results_json or "rag" not in review.results_json:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="RAG not ready")
    return review.results_json.get("rag")


@router.get("/reviews/{review_id}/text")
def get_extracted_text(
    review_id: uuid.UUID,
    response: Response,
    include_text: bool = False,
    db: Session = Depends(get_db),
):
    # TODO: protect this endpoint (internal/dev only).
    review = db.scalar(select(Review).where(Review.id == review_id))
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")

    if response is not None:
        response.headers["Cache-Control"] = "no-store"

    payload: dict[str, Any] = {"extracted_meta": review.extracted_meta}
    if include_text:
        payload["extracted_text"] = review.extracted_text
    return payload


@router.get("/reviews/{review_id}/export/pdf")
def export_pdf(review_id: uuid.UUID, db: Session = Depends(get_db)):
    review = db.scalar(select(Review).where(Review.id == review_id))
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    if review.status != "completed" or not review.results_json:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Results not ready")

    results = review.results_json
    vendor_name = None
    if isinstance(review.context_json, dict):
        vendor_name = review.context_json.get("vendor_name")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, title="DPA Guard Report")
    styles = getSampleStyleSheet()
    story = []

    title = vendor_name or f"Review {review_id}"
    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 12))

    warnings = results.get("warnings") or []
    if warnings:
        story.append(Paragraph("Warnings", styles["Heading2"]))
        for warning in warnings:
            story.append(Paragraph(f"- {warning}", styles["BodyText"]))
        story.append(Spacer(1, 12))

    story.append(Paragraph("Executive Summary", styles["Heading2"]))
    story.append(Paragraph(results.get("exec_summary", ""), styles["BodyText"]))
    story.append(Spacer(1, 12))

    extraction = results.get("extraction", {})
    extraction_table = Table(
        [
            ["Chars", extraction.get("chars"), "Pages", extraction.get("pages")],
            ["Chunks", extraction.get("chunks"), "Doc Type Score", extraction.get("doc_type_score")],
        ]
    )
    extraction_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]
        )
    )
    story.append(Paragraph("Extraction Stats", styles["Heading2"]))
    story.append(extraction_table)
    story.append(Spacer(1, 12))

    risk_table = results.get("risk_table", [])
    if risk_table:
        story.append(Paragraph("Risk Table", styles["Heading2"]))
        table_data = [["Check", "Severity", "Evidence (short)"]]
        for finding in risk_table:
            quote = ""
            quotes = finding.get("evidence_quotes") or []
            if quotes:
                quote = quotes[0].get("quote", "")[:240]
            table_data.append([finding.get("title"), finding.get("severity"), quote])
        risk_table_el = Table(table_data, colWidths=[180, 80, 280])
        risk_table_el.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(risk_table_el)
        story.append(Spacer(1, 12))

    negotiation_pack = results.get("negotiation_pack", [])
    if negotiation_pack:
        story.append(Paragraph("Negotiation Pack", styles["Heading2"]))
        for item in negotiation_pack:
            story.append(Paragraph(item.get("title", ""), styles["Heading3"]))
            story.append(Paragraph(f"Severity: {item.get('severity')}", styles["BodyText"]))
            story.append(Paragraph(f"Ask: {item.get('ask')}", styles["BodyText"]))
            story.append(Paragraph(f"Fallback: {item.get('fallback')}", styles["BodyText"]))
            story.append(Paragraph(f"Rationale: {item.get('rationale')}", styles["BodyText"]))
            story.append(Spacer(1, 8))

    for finding in risk_table:
        quotes = finding.get("evidence_quotes") or []
        if not quotes:
            continue
        story.append(Paragraph(f"Evidence: {finding.get('title')}", styles["Heading3"]))
        for quote in quotes:
            story.append(Paragraph(quote.get("quote", ""), styles["BodyText"]))
        story.append(Spacer(1, 8))

    doc.build(story)
    buffer.seek(0)
    pdf_bytes = buffer.read()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    object_key = f"reviews/{review_id}/export/report_{timestamp}.pdf"
    s3 = get_internal_s3_client()
    s3.put_object(Bucket=settings.s3_bucket, Key=object_key, Body=pdf_bytes, ContentType="application/pdf")

    review.export_object_key = object_key
    review.export_created_at = datetime.now(timezone.utc)
    db.add(review)
    db.commit()

    url = None
    try:
        presign = get_presign_s3_client()
        url = presign.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.s3_bucket, "Key": object_key},
            ExpiresIn=3600,
        )
    except Exception:
        url = None

    response = {"object_key": object_key}
    if url:
        response["url"] = url
    else:
        response["url"] = None
        response["note"] = "Presigned URL unavailable; fetch from MinIO console."
    return response

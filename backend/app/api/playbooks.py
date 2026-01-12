import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.celery_app import celery_app
from app.core.config import settings
from app.core.s3 import get_internal_s3_client
from app.db.session import get_db
from app.models.playbook import Playbook, PlaybookChunk
from app.playbooks.loader import list_playbooks
from app.llm.provider import embed_texts
from app.llm.config import llm_settings

router = APIRouter()

ALLOWED_PLAYBOOK_TYPES = {
    "application/pdf",
    "application/json",
    "application/yaml",
    "text/yaml",
    "text/plain",
    "text/markdown",
}


def _safe_name(filename: str | None) -> str:
    return Path(filename or "upload").name


@router.post("/playbooks/upload")
def upload_playbook(
    file: UploadFile = File(...),
    title: str | None = None,
    version: str | None = None,
    db: Session = Depends(get_db),
):
    if file.content_type not in ALLOWED_PLAYBOOK_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported playbook type")

    playbook = Playbook(title=title, version=version, status="uploaded")
    db.add(playbook)
    db.commit()
    db.refresh(playbook)

    safe_name = _safe_name(file.filename)
    object_key = f"playbooks/{playbook.id}/source/{uuid.uuid4()}_{safe_name}"
    s3 = get_internal_s3_client()
    s3.upload_fileobj(file.file, settings.s3_bucket, object_key)

    playbook.source_object_key = object_key
    playbook.source_filename = safe_name
    db.add(playbook)
    db.commit()

    celery_app.send_task("index_playbook", args=[str(playbook.id)])

    return {"id": str(playbook.id), "status": playbook.status}


@router.get("/playbook/versions")
def playbook_versions(db: Session = Depends(get_db)):
    built_in = list_playbooks()
    uploaded = db.scalars(select(Playbook)).all()
    uploaded_payload = [
        {
            "id": str(item.id),
            "title": item.title,
            "version": item.version,
            "status": item.status,
            "error_message": item.error_message,
        }
        for item in uploaded
    ]
    return {"playbooks": built_in + uploaded_payload}


@router.get("/playbooks/{playbook_id}/search")
def search_playbook(
    playbook_id: uuid.UUID,
    q: str,
    k: int = 6,
    db: Session = Depends(get_db),
):
    playbook = db.scalar(select(Playbook).where(Playbook.id == playbook_id))
    if not playbook:
        raise HTTPException(status_code=404, detail="Playbook not found")
    if playbook.status != "ready":
        raise HTTPException(status_code=409, detail="Playbook not ready")
    if not llm_settings.rag_enabled:
        raise HTTPException(status_code=400, detail="RAG is disabled")

    embedding = embed_texts([q])[0]
    stmt = (
        select(PlaybookChunk)
        .where(PlaybookChunk.playbook_id == playbook_id)
        .order_by(PlaybookChunk.embedding.cosine_distance(embedding))
        .limit(k)
    )
    chunks = db.scalars(stmt).all()

    return {
        "playbook_id": str(playbook_id),
        "query": q,
        "chunks": [
            {
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "meta_json": chunk.meta_json,
            }
            for chunk in chunks
        ],
    }


@router.post("/playbooks/{playbook_id}/reindex")
def reindex_playbook(playbook_id: uuid.UUID, db: Session = Depends(get_db)):
    playbook = db.scalar(select(Playbook).where(Playbook.id == playbook_id))
    if not playbook:
        raise HTTPException(status_code=404, detail="Playbook not found")
    celery_app.send_task("reindex_playbook", args=[str(playbook_id)])
    return {"id": str(playbook_id), "status": "queued"}

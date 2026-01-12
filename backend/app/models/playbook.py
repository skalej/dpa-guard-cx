import uuid

from sqlalchemy import Column, DateTime, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from pgvector.sqlalchemy import Vector

from app.models.base import Base


class Playbook(Base):
    __tablename__ = "playbooks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(Text, nullable=True)
    version = Column(String(32), nullable=True)
    source_object_key = Column(Text, nullable=True)
    source_filename = Column(Text, nullable=True)
    status = Column(String(32), nullable=False, default="uploaded")
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class PlaybookChunk(Base):
    __tablename__ = "playbook_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    playbook_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    meta_json = Column(JSONB, nullable=True)
    embedding = Column(Vector(1536), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

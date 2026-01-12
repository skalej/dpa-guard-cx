import os
import uuid

import boto3
from celery import Celery
from sqlalchemy import Column, DateTime, String, Text, create_engine, func, select
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Session, declarative_base, sessionmaker

redis_url = os.getenv("DPA_REDIS_URL", "redis://localhost:6379/0")
database_url = os.getenv("DPA_DATABASE_URL", "postgresql+psycopg2://dpa:dpa@localhost:5432/dpa")
s3_endpoint = os.getenv("DPA_S3_ENDPOINT", "http://localhost:9000")
s3_access_key = os.getenv("DPA_S3_ACCESS_KEY", "minio")
s3_secret_key = os.getenv("DPA_S3_SECRET_KEY", "minio123")
s3_bucket = os.getenv("DPA_S3_BUCKET", "dpa-guard")
s3_region = os.getenv("DPA_S3_REGION", "us-east-1")

celery_app = Celery("dpa_guard", broker=redis_url, backend=redis_url)

engine = create_engine(database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class Review(Base):
    __tablename__ = "reviews"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    status = Column(String(32), nullable=False, default="draft")
    context_json = Column(JSONB, nullable=True)
    results_json = Column(JSONB, nullable=True)
    source_object_key = Column(Text, nullable=True)
    source_filename = Column(Text, nullable=True)
    source_mime = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
        region_name=s3_region,
    )


@celery_app.task(name="process_review")
def process_review(review_id: str):
    # IMPORTANT: never log raw contract text; only IDs/status/metadata.
    db: Session = SessionLocal()
    try:
        try:
            review_uuid = uuid.UUID(review_id)
        except ValueError:
            return {"status": "failed"}

        review = db.scalar(select(Review).where(Review.id == review_uuid))
        if not review:
            return {"status": "not_found"}
        if not review.source_object_key:
            review.status = "failed"
            review.error_message = "Source file missing"
            db.add(review)
            db.commit()
            return {"status": "failed"}

        s3 = get_s3_client()
        head = s3.head_object(Bucket=s3_bucket, Key=review.source_object_key)
        size = head.get("ContentLength", 0)
        if not size or size <= 0:
            raise ValueError("Source file empty")

        review.results_json = {
            "exec_summary": "Placeholder summary. Analysis not implemented.",
            "risk_table": [],
        }
        review.status = "completed"
        review.error_message = None
        db.add(review)
        db.commit()
        return {"status": "completed"}
    except Exception as exc:  # noqa: BLE001
        review = db.scalar(select(Review).where(Review.id == review_uuid))
        if review:
            review.status = "failed"
            review.error_message = str(exc)
            db.add(review)
            db.commit()
        return {"status": "failed"}
    finally:
        db.close()

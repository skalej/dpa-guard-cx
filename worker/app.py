import io
import os
import re
import uuid
from pathlib import Path

import boto3
from docx import Document
from pypdf import PdfReader
import yaml

from negotiation_templates import get_template
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

PLAYBOOK_DIR = Path(__file__).parent / "playbooks"

NON_DPA_KEYWORDS = {"prd", "tdd", "use case", "requirements", "mvp", "roadmap"}
DPA_KEYWORDS = {
    "data processing agreement",
    "data processing addendum",
    "controller",
    "processor",
    "article 28",
    "gdpr",
    "annex",
    "schedule",
}
PRD_TDD_KEYWORDS = {
    "product requirements document",
    "technical design document",
    "mvp",
    "use cases",
    "non-functional requirements",
    "last updated",
}


class Review(Base):
    __tablename__ = "reviews"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    status = Column(String(32), nullable=False, default="draft")
    context_json = Column(JSONB, nullable=True)
    results_json = Column(JSONB, nullable=True)
    extracted_text = Column(Text, nullable=True)
    extracted_meta = Column(JSONB, nullable=True)
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


def normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    collapsed = []
    blank_streak = 0
    for line in lines:
        if line:
            if blank_streak:
                collapsed.append("")
                blank_streak = 0
            collapsed.append(" ".join(line.split()))
        else:
            blank_streak += 1
    return "\n".join(collapsed).strip()


def chunk_text(
    text: str, chunk_size: int = 2000, overlap: int = 200
) -> list[dict[str, int]]:
    chunks: list[dict[str, int]] = []
    if not text:
        return chunks
    step = max(chunk_size - overlap, 1)
    start = 0
    index = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append({"index": index, "start_char": start, "end_char": end})
        index += 1
        start += step
    return chunks


def validate_quote(extracted_text: str, quote: str, start: int, end: int) -> tuple[int, int] | None:
    if start >= 0 and end <= len(extracted_text) and extracted_text[start:end] == quote:
        return start, end
    fallback = extracted_text.find(quote)
    if fallback != -1:
        return fallback, fallback + len(quote)
    return None


def load_playbook(playbook_id: str = "eu_controller") -> dict:
    for path in PLAYBOOK_DIR.glob("*.yaml"):
        data = yaml.safe_load(path.read_text())
        if data.get("id") == playbook_id:
            return data
    raise ValueError(f"Playbook not found: {playbook_id}")


def select_playbook_id(context_json: dict | None) -> str:
    if not context_json:
        return "eu_controller"
    context = context_json.get("context") if isinstance(context_json, dict) else None
    if isinstance(context, dict):
        region = context.get("region")
        role = context.get("company_role")
    else:
        region = context_json.get("region")
        role = context_json.get("company_role")
    if (region or "").lower() in {"eu", "europe", "european_union"}:
        return "eu_controller"
    if (role or "").lower() == "controller":
        return "eu_controller"
    return "eu_controller"


def find_chunk_index(chunks: list[dict[str, int]], start_char: int, end_char: int) -> int | None:
    for chunk in chunks:
        if chunk["start_char"] <= start_char < chunk["end_char"]:
            return chunk["index"]
    return None


def run_checks(extracted_text: str, chunks: list[dict[str, int]], playbook: dict) -> list[dict]:
    findings: list[dict] = []
    lowered = extracted_text.lower()
    for check in playbook.get("checks", []):
        evidence_quotes: list[dict] = []
        for pattern in check.get("patterns", []):
            for match in re.finditer(pattern, extracted_text, flags=re.IGNORECASE | re.MULTILINE):
                if len(evidence_quotes) >= 3:
                    break
                start = match.start()
                end = match.end()
                snippet_start = max(0, start - 80)
                snippet_end = min(len(extracted_text), snippet_start + 240)
                quote = extracted_text[snippet_start:snippet_end]
                if any(keyword in quote.lower() for keyword in NON_DPA_KEYWORDS):
                    continue
                validated = validate_quote(extracted_text, quote, snippet_start, snippet_end)
                if not validated:
                    continue
                v_start, v_end = validated
                chunk_index = find_chunk_index(chunks, v_start, v_end)
                evidence_quotes.append(
                    {
                        "quote": extracted_text[v_start:v_end],
                        "start_char": v_start,
                        "end_char": v_end,
                        "chunk_index": chunk_index,
                    }
                )
            if len(evidence_quotes) >= 3:
                break
        if not evidence_quotes:
            continue
        findings.append(
            {
                "check_id": check.get("check_id"),
                "title": check.get("title"),
                "severity": check.get("severity"),
                "what_good_looks_like": check.get("what_good_looks_like"),
                "recommendation": check.get("recommendation"),
                "evidence_quotes": evidence_quotes,
            }
        )
    return findings


def summarize_findings(findings: list[dict]) -> str:
    counts = {"high": 0, "medium": 0, "med": 0, "low": 0}
    for finding in findings:
        severity = (finding.get("severity") or "").lower()
        if severity in counts:
            counts[severity] += 1
    total_high = counts["high"]
    total_med = counts["medium"] + counts["med"]
    total_low = counts["low"]
    top_titles = [f["title"] for f in findings[:3] if f.get("title")]
    top_part = ", ".join(top_titles) if top_titles else "none"
    return f"Findings: high={total_high}, med={total_med}, low={total_low}. Top: {top_part}."


def detect_doc_type(extracted_text: str) -> dict:
    lowered = extracted_text.lower()
    dpa_score = sum(1 for keyword in DPA_KEYWORDS if keyword in lowered)
    non_dpa_score = sum(1 for keyword in PRD_TDD_KEYWORDS if keyword in lowered)
    warnings = []
    if non_dpa_score >= max(2, dpa_score + 1):
        warnings.append("Document appears non-DPA (PRD/TDD indicators detected); results may be unreliable.")
    return {"dpa": dpa_score, "non_dpa": non_dpa_score, "warnings": warnings}


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

        obj = s3.get_object(Bucket=s3_bucket, Key=review.source_object_key)
        data = obj["Body"].read()
        if not data:
            raise ValueError("Source file empty")

        extracted_text = ""
        pages = None
        if review.source_mime == "application/pdf":
            reader = PdfReader(io.BytesIO(data))
            pages = len(reader.pages)
            extracted_text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
        elif review.source_mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(io.BytesIO(data))
            extracted_text = "\n\n".join(p.text for p in doc.paragraphs)
        else:
            raise ValueError("Unsupported source type")

        normalized = normalize_text(extracted_text)
        if review.source_mime == "application/pdf" and len(normalized) < 500:
            review.status = "failed"
            review.error_message = (
                "PDF appears scanned or contains insufficient text; OCR not supported in MVP."
            )
            review.extracted_text = normalized
            review.extracted_meta = {"chars": len(normalized), "pages": pages, "chunks": []}
            db.add(review)
            db.commit()
            return {"status": "failed"}

        chunks = chunk_text(normalized)
        review.extracted_text = normalized
        review.extracted_meta = {
            "chars": len(normalized),
            "pages": pages,
            "chunks": chunks,
        }

        playbook_id = select_playbook_id(review.context_json or {})
        playbook = load_playbook(playbook_id)
        findings = run_checks(normalized, chunks, playbook)
        exec_summary = summarize_findings(findings)
        doc_type = detect_doc_type(normalized)
        if doc_type["warnings"]:
            exec_summary = f"{doc_type['warnings'][0]} {exec_summary}"

        negotiation_pack = []
        for finding in findings:
            template = get_template(finding.get("check_id", ""))
            negotiation_pack.append(
                {
                    "check_id": finding.get("check_id"),
                    "title": finding.get("title"),
                    "severity": finding.get("severity"),
                    "ask": template.get("ask"),
                    "fallback": template.get("fallback"),
                    "rationale": template.get("rationale"),
                }
            )

        review.results_json = {
            "playbook": {
                "id": playbook.get("id"),
                "version": playbook.get("version"),
                "title": playbook.get("title"),
            },
            "exec_summary": exec_summary,
            "risk_table": findings,
            "negotiation_pack": negotiation_pack,
            "extraction": {
                "chars": len(normalized),
                "pages": pages,
                "chunks": len(chunks),
                "doc_type_score": {"dpa": doc_type["dpa"], "non_dpa": doc_type["non_dpa"]},
            },
        }
        if doc_type["warnings"]:
            review.results_json["warnings"] = doc_type["warnings"]
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

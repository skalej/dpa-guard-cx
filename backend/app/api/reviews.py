import io
import hashlib
import re
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
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

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
        "llm_generated": bool(review.results_json and review.results_json.get("llm")),
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
    findings = review.results_json.get("risk_table") or []
    per_finding = [
        {
            "check_id": finding.get("check_id"),
            "chunks": len((finding.get("rag") or {}).get("chunks") or []),
        }
        for finding in findings
        if finding.get("check_id")
    ]
    return {
        "global": review.results_json.get("rag"),
        "per_finding": per_finding,
    }


@router.get("/reviews/{review_id}/explain")
def explain_review(review_id: uuid.UUID, db: Session = Depends(get_db)):
    review = db.scalar(select(Review).where(Review.id == review_id))
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    if review.status != "completed" or not review.results_json:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Results not ready")

    results = review.results_json
    findings = results.get("risk_table") or []
    negotiation_pack = {item.get("check_id"): item for item in results.get("negotiation_pack") or []}

    response_findings = []
    for finding in findings:
        check_id = finding.get("check_id")
        if not check_id:
            continue
        entry = {
            "check_id": check_id,
            "title": finding.get("title"),
            "severity": finding.get("severity"),
            "recommendation": finding.get("recommendation"),
            "what_good_looks_like": finding.get("what_good_looks_like"),
            "evidence_quotes": finding.get("evidence_quotes") or [],
        }
        rag = (finding.get("rag") or {}).get("chunks") or []
        if rag:
            first = rag[0]
            meta = first.get("meta_json") or {}
            entry["playbook_guidance"] = {
                "heading": meta.get("heading"),
                "content": (first.get("content") or "")[:600],
            }
        negotiation = negotiation_pack.get(check_id)
        if negotiation:
            entry["negotiation"] = {
                "ask": negotiation.get("ask"),
                "fallback": negotiation.get("fallback"),
                "rationale": negotiation.get("rationale"),
            }
        response_findings.append(entry)

    return {
        "review_id": str(review.id),
        "status": review.status,
        "playbook_id": str(review.playbook_id) if review.playbook_id else None,
        "findings": response_findings,
    }


@router.post("/reviews/{review_id}/rerun_llm")
def rerun_llm(review_id: uuid.UUID, db: Session = Depends(get_db)):
    review = db.scalar(select(Review).where(Review.id == review_id))
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    if not review.results_json:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Results not ready")
    celery_app.send_task("rerun_llm", args=[str(review_id)])
    return {"id": str(review.id), "status": "queued"}


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

    pdf_bytes = _build_report_pdf(
        results,
        vendor_name,
        str(review_id),
        review.extracted_text,
        (review.extracted_meta or {}).get("contract_sections") or [],
    )

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


def normalize_text(value: str) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value.replace("\t", " ").replace("\n", " ")).strip()


def sanitize_text(value: str) -> str:
    if not value:
        return ""
    text = value.replace("\x00", "")
    text = text.replace("“", "\"").replace("”", "\"")
    text = text.replace("‘", "'").replace("’", "'")
    return normalize_text(text)


def smart_truncate(value: str, max_chars: int) -> str:
    text = sanitize_text(value)
    if len(text) <= max_chars:
        return text
    window_start = max(max_chars - 80, 0)
    window = text[window_start:max_chars]
    for marker in [". ", "; ", ": ", "? ", "! "]:
        cut = window.rfind(marker)
        if cut != -1:
            end = window_start + cut + len(marker) - 1
            return text[:end] + "..."
    space_cut = text.rfind(" ", 0, max_chars)
    if space_cut > 0:
        return text[:space_cut] + "..."
    return text[:max_chars] + "..."


def _dedupe_quotes(quotes: list[dict], max_items: int = 2) -> list[dict]:
    seen = set()
    deduped = []
    for quote in quotes:
        normalized = sanitize_text(quote.get("quote", ""))
        if not normalized:
            continue
        key = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(quote)
        if len(deduped) >= max_items:
            break
    return deduped


def _clean_evidence_text(value: str) -> str:
    text = sanitize_text(value)
    if not text:
        return ""
    tokens = text.split()
    prefix = ""
    if tokens and tokens[0].islower() and len(tokens[0]) <= 4:
        tokens = tokens[1:]
        prefix = "… "
    if tokens and tokens[-1].islower() and len(tokens[-1]) <= 4:
        tokens = tokens[:-1]
        if tokens:
            tokens[-1] = tokens[-1] + "…"
            return prefix + " ".join(tokens)
        return ""
    return prefix + " ".join(tokens)


def _is_word_char(char: str) -> bool:
    return char.isalnum() or char in {"'", "’"}


def _expand_word_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    if not text:
        return start, end
    start = max(0, min(start, len(text)))
    end = max(0, min(end, len(text)))
    while start > 0 and _is_word_char(text[start - 1]) and _is_word_char(text[start]):
        start -= 1
    while end < len(text) and end > 0 and _is_word_char(text[end - 1]) and _is_word_char(text[end]):
        end += 1
    return start, end


def _extract_sentence_excerpt(text: str, start: int, end: int, clamp_start: int, clamp_end: int) -> str:
    if not text:
        return ""
    start = max(clamp_start, min(start, clamp_end))
    end = max(clamp_start, min(end, clamp_end))
    left = max(
        text.rfind(".", clamp_start, start),
        text.rfind("!", clamp_start, start),
        text.rfind("?", clamp_start, start),
    )
    if left != -1:
        sentence_start = left + 1
    else:
        sentence_start = clamp_start
        if start > clamp_start and start <= clamp_end:
            window = text[max(start - 400, clamp_start) : start]
            nl = window.rfind("\n")
            if nl != -1:
                sentence_start = max(start - 400, clamp_start) + nl + 1
    right_candidates = [
        text.find(".", end, clamp_end),
        text.find("!", end, clamp_end),
        text.find("?", end, clamp_end),
    ]
    right_candidates = [idx for idx in right_candidates if idx != -1]
    if right_candidates:
        sentence_end = min(right_candidates) + 1
    else:
        sentence_end = clamp_end
        if end < clamp_end:
            window = text[end : min(end + 400, clamp_end)]
            nl = window.find("\n")
            if nl != -1:
                sentence_end = end + nl
    excerpt = text[sentence_start:sentence_end].strip()

    if len(excerpt) < 120:
        next_start = sentence_end
        next_right_candidates = [
            text.find(".", next_start, clamp_end),
            text.find("!", next_start, clamp_end),
            text.find("?", next_start, clamp_end),
        ]
        next_right_candidates = [idx for idx in next_right_candidates if idx != -1]
        if next_right_candidates:
            next_end = min(next_right_candidates) + 1
            excerpt = text[sentence_start:next_end].strip()
    return excerpt


def _is_title_case_heading(line: str) -> bool:
    text = line.strip()
    if not (5 <= len(text) <= 80):
        return False
    if text.endswith("."):
        return False
    if text.count(":") > 1:
        return False
    if re.search(r"[;,?!]", text):
        return False
    if re.match(r"^[-*•]\s", text):
        return False
    if re.match(r"^\d+(\.\d+)*[.)]?\s", text):
        return False
    words = [w for w in re.split(r"\s+", text) if re.search(r"[A-Za-z]", w)]
    if not words:
        return False
    letters = [c for c in text if c.isalpha()]
    if letters and all(c.isupper() for c in letters):
        return True
    title_case = sum(1 for w in words if w[0].isupper())
    return title_case / len(words) >= 0.6


def _build_section_index(text: str) -> list[dict]:
    sections = []
    lines = text.splitlines(keepends=True)
    offset = 0
    headings = []
    numbered_re = re.compile(r"^\s*(\d+(?:\.\d+)*)\.\s+(.+)$")
    roman_re = re.compile(r"^\s*([IVXLCDM]+)\.\s+(.+)$", re.IGNORECASE)
    for line in lines:
        stripped = line.strip()
        if not stripped:
            offset += len(line)
            continue
        match = numbered_re.match(stripped)
        if match:
            headings.append((offset, match.group(1), match.group(2).strip()))
            offset += len(line)
            continue
        match = roman_re.match(stripped)
        if match:
            headings.append((offset, match.group(1).upper(), match.group(2).strip()))
            offset += len(line)
            continue
        if _is_title_case_heading(stripped):
            headings.append((offset, None, stripped))
        offset += len(line)

    for idx, (start, num, title) in enumerate(headings):
        end = headings[idx + 1][0] if idx + 1 < len(headings) else len(text)
        sections.append(
            {
                "num": num,
                "title": title,
                "start_char": start,
                "end_char": end,
            }
        )
    return sections


def _find_section(sections: list[dict], start: int) -> dict | None:
    for section in sections:
        if section["start_char"] <= start < section["end_char"]:
            return section
    return None


def _extract_section_sentence(section_text: str, local_start: int, local_end: int) -> str:
    if not section_text:
        return ""
    local_start = max(0, min(local_start, len(section_text)))
    local_end = max(0, min(local_end, len(section_text)))
    left = max(
        section_text.rfind(".", 0, local_start),
        section_text.rfind("!", 0, local_start),
        section_text.rfind("?", 0, local_start),
    )
    right_candidates = [
        section_text.find(".", local_end),
        section_text.find("!", local_end),
        section_text.find("?", local_end),
    ]
    right_candidates = [idx for idx in right_candidates if idx != -1]
    if left != -1 and right_candidates:
        sentence_start = left + 1
        sentence_end = min(right_candidates) + 1
        return section_text[sentence_start:sentence_end].strip()

    line_start = section_text.rfind("\n", 0, local_start)
    line_end = section_text.find("\n", local_end)
    if line_start == -1:
        line_start = 0
    else:
        line_start += 1
    if line_end == -1:
        line_end = len(section_text)
    return section_text[line_start:line_end].strip()


def _keyword_overlap_score(text: str, tokens: list[str]) -> int:
    if not text:
        return 0
    lowered = text.lower()
    return sum(1 for token in tokens if token in lowered)


_STOPWORDS = {
    "with",
    "and",
    "or",
    "of",
    "the",
    "upon",
    "to",
    "for",
    "a",
    "an",
    "in",
    "on",
    "by",
}

_KEYWORD_MAP = {
    "breach_notification_timeline": [
        "breach",
        "personal data breach",
        "notify",
        "notification",
        "undue delay",
        "72",
    ],
    "subprocessor_authorization": ["subprocessor", "subprocessors"],
    "deletion_or_return": ["delete", "deletion", "return", "termination"],
    "assistance_with_dsar_and_security": [
        "assist",
        "assistance",
        "data subject",
        "requests",
        "security",
        "measures",
        "audit",
    ],
    "confidentiality_of_personnel": ["confidential", "confidentiality"],
    "purpose_limitation_and_instructions": [
        "documented instructions",
        "instructions",
        "purpose",
        "process only",
    ],
}

_TARGET_HEADING_KEYWORDS = {
    "assistance_with_dsar_and_security": [
        ["security measures", "toms"],
        ["assistance", "audits", "data subject"],
        ["processor obligations"],
    ],
    "breach_notification_timeline": [
        ["personal data breach"],
        ["breach"],
        ["notification"],
    ],
    "subprocessor_authorization": [["subprocessors", "subprocessor"]],
    "deletion_or_return": [["deletion"], ["return"], ["termination"]],
    "confidentiality_of_personnel": [["confidentiality"]],
    "purpose_limitation_and_instructions": [
        ["processor obligations"],
        ["documented instructions", "instructions"],
    ],
}


def _build_keywords(finding: dict) -> list[str]:
    text = f"{finding.get('title','')} {finding.get('check_id','')}".lower()
    tokens = [
        token
        for token in re.split(r"[^a-zA-Z0-9]+", text)
        if len(token) >= 3 and token not in _STOPWORDS
    ]
    tokens.extend(_KEYWORD_MAP.get(finding.get("check_id"), []))
    return list(dict.fromkeys(tokens))


def _target_heading_groups(finding: dict) -> list[list[str]]:
    groups = _TARGET_HEADING_KEYWORDS.get(finding.get("check_id"))
    if groups:
        return groups
    return []


def _split_sentences_with_offsets(text: str, win_start: int, win_end: int) -> list[dict]:
    window = text[win_start:win_end]
    sentences = []
    start = 0
    idx = 0
    while idx < len(window):
        ch = window[idx]
        if ch == "\n":
            if idx > start:
                sentences.append(
                    {
                        "text": window[start:idx],
                        "start": win_start + start,
                        "end": win_start + idx,
                    }
                )
            start = idx + 1
            idx += 1
            continue
        if ch in ".?!":
            end = idx + 1
            if idx + 1 == len(window) or window[idx + 1].isspace():
                if end > start:
                    sentences.append(
                        {
                            "text": window[start:end],
                            "start": win_start + start,
                            "end": win_start + end,
                        }
                    )
                start = idx + 1
        idx += 1
    if start < len(window):
        sentences.append(
            {
                "text": window[start:],
                "start": win_start + start,
                "end": win_start + len(window),
            }
        )
    return sentences


def _select_sentence_for_quote(
    extracted_text: str, start: int, end: int, keywords: list[str]
) -> dict:
    win_start = max(0, start - 600)
    win_end = min(len(extracted_text), end + 600)
    sentences = _split_sentences_with_offsets(extracted_text, win_start, win_end)
    candidates = []
    for idx, sentence in enumerate(sentences):
        sent_start = sentence["start"]
        sent_end = sentence["end"]
        overlaps = sent_end >= start and sent_start <= end
        near = abs(sent_start - start) <= 200 or abs(sent_end - end) <= 200
        if not (overlaps or near):
            continue
        cleaned = sanitize_text(sentence["text"])
        if not cleaned:
            continue
        score = _keyword_overlap_score(cleaned, keywords)
        candidates.append(
            {
                "index": idx,
                "text": cleaned,
                "start": sent_start,
                "end": sent_end,
                "score": score,
                "overlaps": overlaps,
            }
        )
    if candidates:
        overlap_candidates = [item for item in candidates if item["overlaps"]]
        pool = overlap_candidates if overlap_candidates else candidates
        pool.sort(key=lambda item: (-item["score"], -len(item["text"]), item["start"]))
        chosen = pool[0] if pool[0]["score"] > 0 else max(pool, key=lambda c: len(c["text"]))
    else:
        chosen = {
            "index": None,
            "text": sanitize_text(extracted_text[start:end]),
            "start": start,
            "end": end,
            "score": 0,
        }

    if len(chosen["text"]) < 60 and sentences:
        if chosen.get("index") is None:
            nearest_idx = min(
                range(len(sentences)),
                key=lambda i: abs(sentences[i]["start"] - start),
            )
        else:
            nearest_idx = chosen["index"]
        pieces = [chosen["text"]]
        if len(pieces[0]) < 120:
            next_idx = nearest_idx + 1 if nearest_idx + 1 < len(sentences) else None
            prev_idx = nearest_idx - 1 if nearest_idx - 1 >= 0 else None
            if next_idx is not None:
                next_text = sanitize_text(sentences[next_idx]["text"])
                if next_text:
                    pieces.append(next_text)
            elif prev_idx is not None:
                prev_text = sanitize_text(sentences[prev_idx]["text"])
                if prev_text:
                    pieces.insert(0, prev_text)
        chosen["text"] = sanitize_text(" ".join(pieces))
    return chosen


def _section_sort_key(section: dict) -> tuple:
    num = section.get("num") or ""
    parts = []
    for part in re.split(r"[.]", str(num)):
        if part.isdigit():
            parts.append(int(part))
        else:
            parts.append(part)
    return tuple(parts)


def _build_evidence_entries(
    finding: dict,
    quotes: list[dict],
    extracted_text: str | None,
    sections: list[dict],
) -> list[dict]:
    if not extracted_text:
        return []
    keywords = _build_keywords(finding)
    entries_by_section: dict[str, dict] = {}
    for quote in quotes:
        if quote.get("start_char") is None or quote.get("end_char") is None:
            continue
        start = int(quote["start_char"])
        end = int(quote["end_char"])
        chosen = _select_sentence_for_quote(extracted_text, start, end, keywords)
        section = _find_section(sections, chosen["start"])
        if not section:
            continue
        heading = (
            f"{section.get('num')}. {section.get('title')}"
            if section.get("num")
            else section.get("title")
        )
        if not heading:
            continue
        excerpt = smart_truncate(chosen["text"], 450)
        key = heading.lower()
        current = entries_by_section.get(key)
        if not current or chosen["score"] > current["score"]:
            entries_by_section[key] = {
                "heading": heading,
                "excerpt": excerpt,
                "score": chosen["score"],
                "section": section,
            }

    entries = list(entries_by_section.values())
    if not entries:
        return []

    remaining = list(entries)
    selected: list[dict] = []
    for group in _target_heading_groups(finding):
        best = None
        for entry in remaining:
            heading_lower = entry["heading"].lower()
            if any(keyword in heading_lower for keyword in group):
                if not best or entry["score"] > best["score"]:
                    best = entry
        if best:
            selected.append(best)
            remaining.remove(best)
        if len(selected) >= 3:
            break

    if len(selected) < 3 and remaining:
        remaining.sort(key=lambda item: (-item["score"], _section_sort_key(item["section"])))
        for entry in remaining:
            if len(selected) >= 3:
                break
            selected.append(entry)

    return selected

def _build_evidence_excerpt(
    quote: dict,
    extracted_text: str | None,
    sections: list[dict],
    max_chars: int,
) -> dict:
    raw_quote = quote.get("quote", "")
    if not extracted_text or quote.get("start_char") is None or quote.get("end_char") is None:
        cleaned = _clean_evidence_text(raw_quote)
        return {"section": None, "excerpt": smart_truncate(cleaned, max_chars)}

    raw_start = int(quote.get("start_char", 0))
    raw_end = int(quote.get("end_char", 0))
    section = _find_section(sections, raw_start)
    clamp_start = section["start_char"] if section else 0
    clamp_end = section["end_char"] if section else len(extracted_text)
    section_text = extracted_text[clamp_start:clamp_end]
    local_start = raw_start - clamp_start
    local_end = raw_end - clamp_start
    excerpt = _extract_section_sentence(section_text, local_start, local_end)
    if len(excerpt) < 120:
        remainder = section_text[local_end:]
        next_sentence = _extract_section_sentence(remainder, 0, 0)
        if next_sentence:
            excerpt = f"{excerpt} {next_sentence}".strip()
    cleaned = sanitize_text(excerpt).strip(" \t\n\r\"'“”‘’")
    return {"section": section, "excerpt": smart_truncate(cleaned, max_chars)}


def _build_report_pdf(
    results: dict,
    vendor_name: str | None,
    review_id: str,
    extracted_text: str | None = None,
    contract_sections: list[dict] | None = None,
) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        title="DPA Guard Report",
        leftMargin=48,
        rightMargin=48,
        topMargin=48,
        bottomMargin=48,
    )
    styles = getSampleStyleSheet()
    table_style = ParagraphStyle(
        "TableCell",
        parent=styles["BodyText"],
        fontSize=9,
        leading=11,
    )
    header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["BodyText"],
        fontSize=9,
        leading=11,
        spaceAfter=2,
    )
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=16,
        leading=20,
    )
    story = []

    title = vendor_name or f"Review {review_id}"
    story.append(Paragraph(sanitize_text(title), title_style))
    story.append(Spacer(1, 12))

    warnings = results.get("warnings") or []
    if warnings:
        story.append(Paragraph("Warnings", styles["Heading2"]))
        for warning in warnings:
            story.append(Paragraph(sanitize_text(f"- {warning}"), styles["BodyText"]))
        story.append(Spacer(1, 12))

    story.append(Paragraph("Executive Summary", styles["Heading2"]))
    story.append(Paragraph(sanitize_text(results.get("exec_summary", "")), styles["BodyText"]))
    story.append(Spacer(1, 12))

    extraction = results.get("extraction", {})
    extraction_table = Table(
        [
            [
                Paragraph("Chars", header_style),
                Paragraph(str(extraction.get("chars")), table_style),
                Paragraph("Pages", header_style),
                Paragraph(str(extraction.get("pages")), table_style),
            ],
            [
                Paragraph("Chunks", header_style),
                Paragraph(str(extraction.get("chunks")), table_style),
                Paragraph("Doc Type Score", header_style),
                Paragraph(str(extraction.get("doc_type_score")), table_style),
            ],
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
    sections = _build_section_index(extracted_text or "")
    if risk_table:
        story.append(Paragraph("Risk Table", styles["Heading2"]))
        table_data = [
            [
                Paragraph("Check", header_style),
                Paragraph("Severity", header_style),
                Paragraph("Evidence (short)", header_style),
            ]
        ]
        for finding in risk_table:
            quote = ""
            quotes = finding.get("evidence_quotes") or []
            if quotes:
                entries = _build_evidence_entries(finding, quotes, extracted_text, sections)
                quote = entries[0]["excerpt"] if entries else ""
            table_data.append(
                [
                    Paragraph(sanitize_text(str(finding.get("title") or "")), table_style),
                    Paragraph(sanitize_text(str(finding.get("severity") or "")), table_style),
                    Paragraph(sanitize_text(quote), table_style),
                ]
            )
        evidence_width = max(doc.width - 220, 200)
        risk_table_el = Table(
            table_data, colWidths=[160, 60, evidence_width], repeatRows=1
        )
        risk_table_el.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("WORDWRAP", (0, 0), (-1, -1), "LTR"),
                ]
            )
        )
        story.append(risk_table_el)
        story.append(Spacer(1, 12))

    negotiation_pack = results.get("negotiation_pack", [])
    if negotiation_pack:
        story.append(Paragraph("Negotiation Pack", styles["Heading2"]))
        for item in negotiation_pack:
            story.append(Paragraph(sanitize_text(item.get("title", "")), styles["Heading3"]))
            story.append(Paragraph(sanitize_text(f"Severity: {item.get('severity')}"), styles["BodyText"]))
            story.append(Paragraph(sanitize_text(f"Ask: {item.get('ask')}"), styles["BodyText"]))
            story.append(Paragraph(sanitize_text(f"Fallback: {item.get('fallback')}"), styles["BodyText"]))
            story.append(Paragraph(sanitize_text(f"Rationale: {item.get('rationale')}"), styles["BodyText"]))
            story.append(Spacer(1, 8))

    for finding in risk_table:
        quotes = _dedupe_quotes(finding.get("evidence_quotes") or [], max_items=6)
        if not quotes:
            continue
        block = [Paragraph(sanitize_text(f"Evidence: {finding.get('title')}"), styles["Heading3"])]
        evidence_items = _build_evidence_entries(finding, quotes, extracted_text, sections)
        for item in evidence_items:
            block.append(Paragraph(f"<b>{sanitize_text(item['heading'])}</b>", styles["BodyText"]))
            block.append(Paragraph(sanitize_text(item["excerpt"]), styles["BodyText"]))
        block.append(Spacer(1, 8))
        story.append(KeepTogether(block))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()

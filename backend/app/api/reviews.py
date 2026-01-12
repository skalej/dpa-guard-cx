import uuid

from fastapi import APIRouter, HTTPException

router = APIRouter()


def not_implemented():
    raise HTTPException(status_code=501, detail="Not implemented in scaffold")


@router.post("/reviews")
def create_review():
    not_implemented()


@router.post("/reviews/{review_id}/upload")
def upload_review(review_id: uuid.UUID):
    not_implemented()


@router.post("/reviews/{review_id}/start")
def start_review(review_id: uuid.UUID):
    not_implemented()


@router.get("/reviews/{review_id}")
def get_review(review_id: uuid.UUID):
    not_implemented()


@router.get("/reviews/{review_id}/results")
def get_results(review_id: uuid.UUID):
    not_implemented()


@router.get("/reviews/{review_id}/export/pdf")
def export_pdf(review_id: uuid.UUID):
    not_implemented()


@router.get("/playbook/versions")
def playbook_versions():
    not_implemented()

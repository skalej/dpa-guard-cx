from fastapi import APIRouter

router = APIRouter()


@router.get("/health/live")
def health_live():
    return {"status": "live"}


@router.get("/health/ready")
def health_ready():
    return {"status": "ready"}

from fastapi import FastAPI

from app.api.router import api_router
from app.core.logging import configure_logging

configure_logging()

app = FastAPI(title="DPA Guard API")
app.include_router(api_router)

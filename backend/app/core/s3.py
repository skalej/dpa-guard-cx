import boto3
from botocore.config import Config

from app.core.config import settings


def _base_client(endpoint_url: str | None):
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        region_name=settings.s3_region,
        config=Config(s3={"addressing_style": "path"}, signature_version="s3v4"),
    )


def get_internal_s3_client():
    return _base_client(settings.s3_internal_endpoint)


def get_presign_s3_client():
    endpoint = settings.s3_public_endpoint or settings.s3_internal_endpoint
    return _base_client(endpoint)

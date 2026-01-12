from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DPA_", case_sensitive=False)

    database_url: str = "postgresql+psycopg2://dpa:dpa@localhost:5432/dpa"
    redis_url: str = "redis://localhost:6379/0"
    s3_internal_endpoint: str = Field(
        default="http://minio:9000", validation_alias="S3_INTERNAL_ENDPOINT"
    )
    s3_public_endpoint: str | None = Field(default=None, validation_alias="S3_PUBLIC_ENDPOINT")
    s3_access_key: str = "minio"
    s3_secret_key: str = "minio123"
    s3_bucket: str = "dpa-guard"
    s3_region: str = "us-east-1"


settings = Settings()

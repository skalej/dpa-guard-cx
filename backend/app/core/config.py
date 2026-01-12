from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DPA_", case_sensitive=False)

    database_url: str = "postgresql+psycopg2://dpa:dpa@localhost:5432/dpa"
    redis_url: str = "redis://localhost:6379/0"
    s3_endpoint: str = "http://localhost:9000"
    s3_access_key: str = "minio"
    s3_secret_key: str = "minio123"
    s3_bucket: str = "dpa-guard"


settings = Settings()

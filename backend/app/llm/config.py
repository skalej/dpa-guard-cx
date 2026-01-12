from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)

    llm_provider: str = Field(default="openai", validation_alias="LLM_PROVIDER")
    llm_enabled: bool = Field(default=False, validation_alias="LLM_ENABLED")
    rag_enabled: bool = Field(default=False, validation_alias="RAG_ENABLED")

    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", validation_alias="OPENAI_MODEL")
    openai_embed_model: str = Field(
        default="text-embedding-3-small", validation_alias="OPENAI_EMBED_MODEL"
    )


llm_settings = LLMSettings()

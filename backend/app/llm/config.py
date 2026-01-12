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
    openai_embed_max_texts_per_call: int = Field(
        default=64, validation_alias="OPENAI_EMBED_MAX_TEXTS_PER_CALL"
    )
    openai_embed_max_chars_per_text: int = Field(
        default=2000, validation_alias="OPENAI_EMBED_MAX_CHARS_PER_TEXT"
    )
    openai_response_max_output_tokens: int = Field(
        default=600, validation_alias="OPENAI_RESPONSE_MAX_OUTPUT_TOKENS"
    )


llm_settings = LLMSettings()

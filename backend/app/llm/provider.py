from app.llm.config import llm_settings
from app.llm.openai_client import embed_texts as openai_embed_texts
from app.llm.openai_client import generate_structured as openai_generate_structured


def embed_texts(texts: list[str]) -> list[list[float]]:
    provider = (llm_settings.llm_provider or "openai").lower()
    if provider == "openai":
        return openai_embed_texts(texts)
    raise ValueError(f"Unsupported LLM provider: {provider}")


def generate_structured(prompt: str, json_schema: dict) -> dict:
    provider = (llm_settings.llm_provider or "openai").lower()
    if provider == "openai":
        return openai_generate_structured(prompt, json_schema)
    raise ValueError(f"Unsupported LLM provider: {provider}")

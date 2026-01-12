import json
import time

from openai import OpenAI, APIConnectionError, APIError, RateLimitError, APITimeoutError

from app.llm.config import llm_settings


class LLMConfigError(RuntimeError):
    pass


def _get_client() -> OpenAI:
    if (llm_settings.llm_enabled or llm_settings.rag_enabled) and not llm_settings.openai_api_key:
        raise LLMConfigError("OPENAI_API_KEY is required when LLM/RAG is enabled")
    return OpenAI(api_key=llm_settings.openai_api_key)


def _retry(func, attempts: int = 2):
    for attempt in range(attempts):
        try:
            return func()
        except (APIConnectionError, APIError, RateLimitError, APITimeoutError):
            if attempt + 1 >= attempts:
                raise
            time.sleep(0.5)


def embed_texts(texts: list[str]) -> list[list[float]]:
    client = _get_client()

    def _call():
        response = client.embeddings.create(
            model=llm_settings.openai_embed_model,
            input=texts,
            timeout=30,
        )
        return [item.embedding for item in response.data]

    return _retry(_call)


def generate_structured(prompt: str, json_schema: dict) -> dict:
    client = _get_client()

    def _call():
        response = client.chat.completions.create(
            model=llm_settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_schema", "json_schema": json_schema},
            timeout=30,
        )
        content = response.choices[0].message.content or ""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(content[start : end + 1])
                except json.JSONDecodeError:
                    pass
            raise RuntimeError("LLM returned invalid JSON")

    return _retry(_call)

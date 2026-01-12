import hashlib
import json
import random
import time

from openai import OpenAI, APIConnectionError, APIError, RateLimitError, APITimeoutError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.llm.config import llm_settings
from app.db.session import SessionLocal
from app.models.embedding_cache import EmbeddingCache


class LLMConfigError(RuntimeError):
    pass


def _get_client() -> OpenAI:
    if (llm_settings.llm_enabled or llm_settings.rag_enabled) and not llm_settings.openai_api_key:
        raise LLMConfigError("OPENAI_API_KEY is required when LLM/RAG is enabled")
    return OpenAI(api_key=llm_settings.openai_api_key)


def _retry(func, attempts: int = 3, base_sleep: float = 0.5):
    for attempt in range(attempts):
        try:
            return func()
        except (APIConnectionError, APIError, RateLimitError, APITimeoutError):
            if attempt + 1 >= attempts:
                raise
            sleep = base_sleep * (2**attempt)
            time.sleep(sleep + random.uniform(0, 0.25))


def _hash_embedding_input(model: str, text: str) -> str:
    payload = f"{model}\n{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def embed_texts(texts: list[str]) -> list[list[float]]:
    client = _get_client()
    max_texts = llm_settings.openai_embed_max_texts_per_call
    max_chars = llm_settings.openai_embed_max_chars_per_text
    sanitized = [text[:max_chars] for text in texts]
    embeddings: list[list[float] | None] = [None] * len(sanitized)
    pending: list[tuple[int, str, str]] = []
    db = SessionLocal()
    try:
        for idx, text in enumerate(sanitized):
            sha = _hash_embedding_input(llm_settings.openai_embed_model, text)
            cached = db.scalar(
                select(EmbeddingCache).where(EmbeddingCache.model == llm_settings.openai_embed_model).where(
                    EmbeddingCache.sha256 == sha
                )
            )
            if cached:
                embeddings[idx] = cached.embedding
            else:
                pending.append((idx, text, sha))

        for batch_start in range(0, len(pending), max_texts):
            batch = pending[batch_start : batch_start + max_texts]
            inputs = [item[1] for item in batch]

            def _call():
                response = client.embeddings.create(
                    model=llm_settings.openai_embed_model,
                    input=inputs,
                    timeout=30,
                )
                return [item.embedding for item in response.data]

            batch_embeddings = _retry(_call)
            for (idx, text, sha), embedding in zip(batch, batch_embeddings, strict=False):
                embeddings[idx] = embedding
                dims = len(embedding)
                try:
                    db.add(
                        EmbeddingCache(
                            model=llm_settings.openai_embed_model,
                            sha256=sha,
                            dims=dims,
                            embedding=embedding,
                        )
                    )
                except IntegrityError:
                    db.rollback()
            try:
                db.commit()
            except IntegrityError:
                db.rollback()
    finally:
        db.close()

    normalized = []
    for embedding in embeddings:
        if embedding is None:
            normalized.append([])
        else:
            normalized.append(list(embedding))
    return normalized


def generate_structured(prompt: str, json_schema: dict) -> dict:
    client = _get_client()

    def _call():
        response = client.chat.completions.create(
            model=llm_settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_schema", "json_schema": json_schema},
            max_tokens=llm_settings.openai_response_max_output_tokens,
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

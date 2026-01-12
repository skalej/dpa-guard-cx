from app.llm.config import llm_settings
from app.llm.provider import embed_texts, generate_structured


def test_llm_imports_smoke():
    assert llm_settings.llm_enabled is False
    assert llm_settings.rag_enabled is False
    assert callable(embed_texts)
    assert callable(generate_structured)

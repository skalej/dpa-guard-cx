import unittest
from types import SimpleNamespace
from unittest.mock import patch

from _test_utils import load_worker_app

app = load_worker_app()


class FakeEmbeddings:
    def __init__(self, calls):
        self.calls = calls

    def create(self, model, input, timeout):
        self.calls["count"] += 1
        return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 2.0, 3.0]) for _ in input])


class FakeOpenAI:
    def __init__(self, api_key=None):
        self.embeddings = FakeEmbeddings(self.calls)

    calls = {"count": 0}


class EmbeddingCacheTests(unittest.TestCase):
    def test_cache_hit_skips_second_call(self):
        cache = {}

        def fake_get(db, model, sha):
            return cache.get(sha)

        def fake_store(db, model, sha, embedding):
            cache[sha] = SimpleNamespace(embedding=embedding)

        class DummyDB:
            def commit(self):
                return None

            def rollback(self):
                return None

        with patch.object(app, "OpenAI", FakeOpenAI), patch.object(
            app, "_get_cached_embedding", fake_get
        ), patch.object(app, "_store_cached_embedding", fake_store):
            original_rag = app.rag_enabled
            original_key = app.openai_api_key
            try:
                app.rag_enabled = True
                app.openai_api_key = "test-key"
                FakeOpenAI.calls["count"] = 0
                app.embed_texts_openai(["hello"], db=DummyDB())
                app.embed_texts_openai(["hello"], db=DummyDB())
                self.assertEqual(FakeOpenAI.calls["count"], 1)
            finally:
                app.rag_enabled = original_rag
                app.openai_api_key = original_key


if __name__ == "__main__":
    unittest.main()

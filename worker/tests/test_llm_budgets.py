import unittest

from _test_utils import load_worker_app

app = load_worker_app()


class LlmBudgetTests(unittest.TestCase):
    def test_budgets_truncate(self):
        original_findings = app.llm_max_findings
        original_evidence = app.llm_max_total_evidence_chars
        original_rag = app.llm_max_total_rag_chars
        original_rag_chunks = app.rag_max_chunks_per_finding
        try:
            app.llm_max_findings = 1
            app.llm_max_total_evidence_chars = 10
            app.llm_max_total_rag_chars = 12
            app.rag_max_chunks_per_finding = 1
            findings = [
                {
                    "check_id": "a",
                    "title": "First",
                    "severity": "high",
                    "evidence_quotes": [{"quote": "x" * 50}],
                    "rag": {"chunks": [{"chunk_index": 1, "content": "y" * 50, "meta_json": {}}]},
                },
                {
                    "check_id": "b",
                    "title": "Second",
                    "severity": "low",
                    "evidence_quotes": [{"quote": "z" * 50}],
                    "rag": {"chunks": [{"chunk_index": 2, "content": "w" * 50, "meta_json": {}}]},
                },
            ]
            payload, stats = app.build_llm_payload({}, findings)
            self.assertEqual(len(payload["risk_table"]), 1)
            self.assertLessEqual(stats["total_evidence_chars"], 10)
            self.assertLessEqual(stats["total_rag_chars"], 12)
        finally:
            app.llm_max_findings = original_findings
            app.llm_max_total_evidence_chars = original_evidence
            app.llm_max_total_rag_chars = original_rag
            app.rag_max_chunks_per_finding = original_rag_chunks


if __name__ == "__main__":
    unittest.main()

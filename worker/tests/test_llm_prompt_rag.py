import json
import unittest

from _test_utils import load_worker_app

app = load_worker_app()
build_llm_prompt = app.build_llm_prompt


class LlmPromptRagTests(unittest.TestCase):
    def _extract_payload(self, prompt: str) -> dict:
        marker = "INPUT JSON:\n"
        start = prompt.find(marker)
        self.assertNotEqual(start, -1)
        payload = prompt[start + len(marker) :]
        return json.loads(payload)

    def test_per_finding_rag_included_and_truncated(self):
        findings = [
            {
                "check_id": "a",
                "title": "A",
                "severity": "high",
                "recommendation": "rec a",
                "evidence_quotes": [],
                "rag": {
                    "chunks": [
                        {"chunk_index": 1, "content": "x" * 800, "meta_json": {"h": "A"}}
                    ]
                },
            },
            {
                "check_id": "b",
                "title": "B",
                "severity": "low",
                "recommendation": "rec b",
                "evidence_quotes": [],
                "rag": {
                    "chunks": [
                        {"chunk_index": 2, "content": "y" * 700, "meta_json": {"h": "B"}}
                    ]
                },
            },
        ]
        prompt, _, _ = build_llm_prompt({"region": "EU"}, findings)
        payload = self._extract_payload(prompt)
        self.assertEqual(len(payload["risk_table"]), 2)
        rag_a = payload["risk_table"][0]["rag_chunks"][0]["content"]
        rag_b = payload["risk_table"][1]["rag_chunks"][0]["content"]
        self.assertLessEqual(len(rag_a), 600)
        self.assertLessEqual(len(rag_b), 600)


if __name__ == "__main__":
    unittest.main()

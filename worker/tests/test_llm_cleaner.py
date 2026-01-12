import unittest

from app import clean_llm_output


class LlmCleanerTests(unittest.TestCase):
    def setUp(self):
        self.allowed = {"a", "b"}
        self.deterministic_exec = "deterministic summary"
        self.deterministic_pack = [
            {
                "check_id": "a",
                "title": "A",
                "severity": "high",
                "ask": "ask a",
                "fallback": "fb a",
                "rationale": "ra a",
            },
            {
                "check_id": "b",
                "title": "B",
                "severity": "low",
                "ask": "ask b",
                "fallback": "fb b",
                "rationale": "ra b",
            },
        ]

    def test_filters_extra_and_duplicates(self):
        output = {
            "exec_summary": "llm summary",
            "negotiation_pack": [
                {"check_id": "a", "ask": "a1", "fallback": "f1", "rationale": "r1"},
                {"check_id": "a", "ask": "a2", "fallback": "f2", "rationale": "r2"},
                {"check_id": "c", "ask": "c", "fallback": "c", "rationale": "c"},
            ],
        }
        cleaned = clean_llm_output(
            output, self.allowed, self.deterministic_exec, self.deterministic_pack
        )
        self.assertEqual(cleaned["exec_summary"], "llm summary")
        self.assertEqual(len(cleaned["negotiation_pack"]), 2)
        self.assertEqual(cleaned["negotiation_pack"][0]["check_id"], "a")
        self.assertEqual(cleaned["negotiation_pack"][1]["check_id"], "b")

    def test_missing_ids_filled(self):
        output = {"exec_summary": "llm summary", "negotiation_pack": []}
        cleaned = clean_llm_output(
            output, self.allowed, self.deterministic_exec, self.deterministic_pack
        )
        self.assertEqual(len(cleaned["negotiation_pack"]), 2)
        self.assertEqual(cleaned["negotiation_pack"][0]["ask"], "ask a")

    def test_long_strings_trimmed(self):
        long_text = "x" * 450
        output = {
            "exec_summary": "llm summary",
            "negotiation_pack": [
                {"check_id": "a", "ask": long_text, "fallback": long_text, "rationale": long_text}
            ],
        }
        cleaned = clean_llm_output(
            output, self.allowed, self.deterministic_exec, self.deterministic_pack
        )
        item = cleaned["negotiation_pack"][0]
        self.assertEqual(len(item["ask"]), 400)
        self.assertEqual(len(item["fallback"]), 400)
        self.assertEqual(len(item["rationale"]), 400)


if __name__ == "__main__":
    unittest.main()

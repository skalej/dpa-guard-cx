import unittest

from app.api.reviews import _build_report_pdf


class PdfExportTests(unittest.TestCase):
    def test_pdf_build_with_long_text(self):
        long_text = "Long evidence " * 200
        results = {
            "exec_summary": "Summary",
            "extraction": {"chars": 1234, "pages": 2, "chunks": 4, "doc_type_score": {"dpa": 3}},
            "risk_table": [
                {
                    "title": "Breach Notification Timeline",
                    "severity": "high",
                    "evidence_quotes": [{"quote": long_text}],
                }
            ],
            "negotiation_pack": [
                {
                    "title": "Breach Notification Timeline",
                    "severity": "high",
                    "ask": long_text,
                    "fallback": long_text,
                    "rationale": long_text,
                }
            ],
        }
        pdf_bytes = _build_report_pdf(results, "Vendor Example", "review-123")
        self.assertGreater(len(pdf_bytes), 100)


if __name__ == "__main__":
    unittest.main()

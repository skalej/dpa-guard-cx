import unittest

from app.api.reviews import (
    _build_evidence_entries,
    _build_report_pdf,
    _build_section_index,
    _clean_evidence_text,
    _dedupe_quotes,
    normalize_text,
    sanitize_text,
    smart_truncate,
)


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

    def test_smart_truncate_prefers_sentence(self):
        text = "First sentence. Second sentence is long and keeps going for a while."
        truncated = smart_truncate(text, 30)
        self.assertTrue(truncated.endswith("..."))
        self.assertIn("First sentence", truncated)
        self.assertNotIn("Second sentence is long", truncated)

    def test_smart_truncate_no_mid_word(self):
        text = "Alpha beta gamma delta"
        truncated = smart_truncate(text, 12)
        self.assertTrue(truncated.endswith("..."))
        last_token = truncated[:-3].split()[-1]
        self.assertIn(last_token, text.split())

    def test_dedupe_quotes(self):
        quotes = [
            {"quote": "Line one"},
            {"quote": "Line  one"},
            {"quote": "Line two"},
        ]
        deduped = _dedupe_quotes(quotes, max_items=2)
        self.assertEqual(len(deduped), 2)
        self.assertEqual(deduped[0]["quote"], "Line one")

    def test_clean_evidence_text_strips_fragments(self):
        text = "ing aware of a personal data breach."
        cleaned = _clean_evidence_text(text)
        self.assertTrue(cleaned.startswith("… "))
        self.assertIn("aware of a personal data breach.", cleaned)

    def test_excerpt_no_newlines_or_mid_word(self):
        extracted = "1. Intro\nAlpha beta gamma delta epsilon zeta."
        quote = {"quote": "gamma delta", "start_char": 13, "end_char": 25}
        finding = {"check_id": "breach_notification_timeline", "title": "Breach Notification Timeline"}
        sections = _build_section_index(extracted)
        entries = _build_evidence_entries(finding, [quote], extracted, sections)
        excerpt = entries[0]["excerpt"] if entries else ""
        self.assertNotIn("\n", excerpt)
        self.assertTrue(excerpt)
        last_token = excerpt[:-3].split()[-1] if excerpt.endswith("...") else excerpt.split()[-1]
        tokens = [token.strip(".") for token in normalize_text(extracted).split()]
        self.assertIn(last_token.strip("."), tokens)

    def test_dedupe_by_span(self):
        quotes = [
            {"quote": "Line one", "start_char": 10, "end_char": 20},
            {"quote": "Line one", "start_char": 10, "end_char": 20},
            {"quote": "Line two", "start_char": 30, "end_char": 40},
        ]
        deduped = _dedupe_quotes(quotes, max_items=2)
        self.assertEqual(len(deduped), 2)

    def test_sentence_excerpt_boundaries(self):
        extracted = "1. Intro\nFirst sentence. Second sentence continues. Third sentence."
        quote = {"quote": "Second sentence", "start_char": 26, "end_char": 42}
        finding = {"check_id": "breach_notification_timeline", "title": "Breach Notification Timeline"}
        sections = _build_section_index(extracted)
        entries = _build_evidence_entries(finding, [quote], extracted, sections)
        excerpt = entries[0]["excerpt"] if entries else ""
        self.assertTrue(excerpt.startswith("Second sentence"))
        self.assertIn("Second sentence continues.", excerpt)

    def test_section_clamp(self):
        extracted = "8. Security Measures.\n9. Assistance.\n10. Deletion."
        sections = _build_section_index(extracted)
        quote = {"quote": "Assistance", "start_char": 23, "end_char": 33}
        finding = {"check_id": "assistance_with_dsar_and_security", "title": "Assistance"}
        entries = _build_evidence_entries(finding, [quote], extracted, sections)
        excerpt = entries[0]["excerpt"] if entries else ""
        self.assertNotIn("10. Deletion", excerpt)

    def test_section_sentence_does_not_spill(self):
        extracted = (
            "8. Personal Data Breach\nThe Processor shall notify the Controller without undue delay."
            "\n9. Assistance & Audits\nThe Processor shall assist the Controller in responding to data subject requests."
            "\n10. Deletion\nUpon termination, the Processor shall delete or return all personal data."
        )
        sections = _build_section_index(extracted)
        quote = {"quote": "assist the Controller", "start_char": 100, "end_char": 120}
        finding = {"check_id": "assistance_with_dsar_and_security", "title": "Assistance"}
        entries = _build_evidence_entries(finding, [quote], extracted, sections)
        excerpt = entries[0]["excerpt"] if entries else ""
        self.assertIn("assist the Controller", excerpt)
        self.assertNotIn("Deletion", excerpt)

    def test_heading_detection_title_case(self):
        extracted = "Processor Obligations\nThe Processor shall process data.\n"
        sections = _build_section_index(extracted)
        self.assertEqual(sections[0]["title"], "Processor Obligations")

    def test_breach_sentence_contains_keyword(self):
        extracted = (
            "7. International Transfers\nTransfers allowed.\n"
            "8. Personal Data Breach\nThe Processor shall notify the Controller without undue delay."
        )
        sections = _build_section_index(extracted)
        quote = {"quote": "notify the Controller", "start_char": 60, "end_char": 85}
        finding = {"check_id": "breach_notification_timeline", "title": "Breach Notification Timeline"}
        entries = _build_evidence_entries(finding, [quote], extracted, sections)
        self.assertTrue(any("breach" in entry["excerpt"].lower() or "notify" in entry["excerpt"].lower() for entry in entries))
        self.assertTrue(all(entry["heading"].startswith("8.") for entry in entries))

    def test_breach_excerpt_not_too_short(self):
        extracted = (
            "7. International Transfers\nAppropriate safeguards are in place.\n"
            "8. Personal Data Breach\nThe Processor shall notify the Controller without undue delay."
        )
        sections = _build_section_index(extracted)
        quote = {"quote": "Personal Data Breach", "start_char": 70, "end_char": 90}
        finding = {"check_id": "breach_notification_timeline", "title": "Breach Notification Timeline"}
        entries = _build_evidence_entries(finding, [quote], extracted, sections)
        self.assertTrue(entries)
        self.assertGreaterEqual(len(entries[0]["excerpt"]), 60)

    def test_dsar_includes_security_measures_heading_or_text(self):
        extracted = (
            "5. Security Measures\nThe Processor shall implement appropriate technical and organizational measures.\n"
            "9. Assistance & Audits\nThe Processor shall assist the Controller in responding to data subject requests."
        )
        sec_phrase = "technical and organizational measures"
        assist_phrase = "assist the Controller"
        sections = _build_section_index(extracted)
        quotes = [
            {
                "quote": sec_phrase,
                "start_char": extracted.index(sec_phrase),
                "end_char": extracted.index(sec_phrase) + len(sec_phrase),
            },
            {
                "quote": assist_phrase,
                "start_char": extracted.index(assist_phrase),
                "end_char": extracted.index(assist_phrase) + len(assist_phrase),
            },
        ]
        finding = {
            "check_id": "assistance_with_dsar_and_security",
            "title": "Assistance with Data Subject Rights and Security",
        }
        entries = _build_evidence_entries(finding, quotes, extracted, sections)
        self.assertTrue(entries)
        self.assertTrue(
            any(
                "security measures" in entry["heading"].lower()
                or "technical and organizational measures" in entry["excerpt"].lower()
                for entry in entries
            )
        )

    def test_excerpt_preserves_sentence_punctuation(self):
        extracted = "1. Intro\nAlpha beta. Gamma delta."
        quote = {"quote": "Gamma delta", "start_char": 22, "end_char": 33}
        finding = {"check_id": "breach_notification_timeline", "title": "Breach Notification Timeline"}
        sections = _build_section_index(extracted)
        entries = _build_evidence_entries(finding, [quote], extracted, sections)
        excerpt = entries[0]["excerpt"] if entries else ""
        self.assertTrue(excerpt.endswith("."))

    def test_sanitize_text(self):
        text = "“Quoted” and ‘single’\nline"
        self.assertEqual(sanitize_text(text), "\"Quoted\" and 'single' line")


if __name__ == "__main__":
    unittest.main()

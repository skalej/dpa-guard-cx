import unittest

from _test_utils import load_worker_app

app = load_worker_app()


class PlaybookSplitterTests(unittest.TestCase):
    def test_plain_headings_detected(self):
        text = "\n".join(
            [
                "Processor Obligations",
                "Requirement: Process only on documented instructions.",
                "",
                "Confidentiality",
                "Requirement: Staff are bound by confidentiality.",
            ]
        )
        sections = app._split_playbook_sections(text)
        headings = [section["meta_json"]["heading"] for section in sections]
        self.assertIn("Processor Obligations", headings)
        self.assertIn("Confidentiality", headings)

    def test_coded_headings_detected(self):
        text = "\n".join(
            [
                "DPA-BR-01 - Breach Notification",
                "Requirement: Notify within 72 hours.",
                "",
                "DPA-DEL-01 - Deletion/Return",
                "Requirement: Delete within 30 days.",
            ]
        )
        sections = app._split_playbook_sections(text)
        headings = [section["meta_json"]["heading"] for section in sections]
        self.assertIn("DPA-BR-01 - Breach Notification", headings)
        self.assertIn("DPA-DEL-01 - Deletion/Return", headings)


if __name__ == "__main__":
    unittest.main()

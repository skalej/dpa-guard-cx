import unittest

from _test_utils import load_worker_app

app = load_worker_app()
select_rag_chunks_for_finding = app.select_rag_chunks_for_finding


class RagSelectionTests(unittest.TestCase):
    def test_breach_prefers_br_heading(self):
        finding = {"check_id": "breach_notification_timeline", "title": "Breach Notification Timeline"}
        chunks = [
            {"meta_json": {"heading": "DPA-SUB-01 – Subprocessors"}, "content": "sub"},
            {"meta_json": {"heading": "DPA-BR-01 – Breach Notification"}, "content": "breach"},
        ]
        selected = select_rag_chunks_for_finding(chunks, finding, max_chunks=2)
        self.assertEqual(selected[0]["content"], "breach")

    def test_no_heading_falls_back_to_top_rank(self):
        finding = {"check_id": "unknown_check", "title": "Unknown"}
        chunks = [
            {"meta_json": {}, "content": "first"},
            {"meta_json": {}, "content": "second"},
        ]
        selected = select_rag_chunks_for_finding(chunks, finding, max_chunks=2)
        self.assertEqual(selected[0]["content"], "first")


if __name__ == "__main__":
    unittest.main()

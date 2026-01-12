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
        self.assertEqual(selected, [])

    def test_relevant_heading_kept(self):
        finding = {"check_id": "subprocessor_authorization", "title": "Subprocessor Authorization"}
        chunks = [
            {"meta_json": {"heading": "DPA-DEL-01 – Deletion/Return"}, "content": "del"},
            {"meta_json": {"heading": "DPA-SUB-01 – Subprocessors"}, "content": "sub"},
        ]
        selected = select_rag_chunks_for_finding(chunks, finding, max_chunks=2)
        self.assertEqual(selected[0]["content"], "sub")

    def test_purpose_limitation_synonym_matches(self):
        finding = {
            "check_id": "purpose_limitation_and_instructions",
            "title": "Purpose Limitation and Documented Instructions",
        }
        chunks = [
            {"meta_json": {"heading": "Processor Obligations"}, "content": "po"},
            {"meta_json": {"heading": "DPA-SUB-01 – Subprocessors"}, "content": "sub"},
        ]
        selected = select_rag_chunks_for_finding(chunks, finding, max_chunks=2)
        self.assertEqual(selected[0]["content"], "po")

    def test_confidentiality_synonym_matches(self):
        finding = {
            "check_id": "confidentiality_of_personnel",
            "title": "Confidentiality of Personnel",
        }
        chunks = [
            {"meta_json": {"heading": "Confidentiality"}, "content": "conf"},
            {"meta_json": {"heading": "DPA-DEL-01 – Deletion/Return"}, "content": "del"},
        ]
        selected = select_rag_chunks_for_finding(chunks, finding, max_chunks=2)
        self.assertEqual(selected[0]["content"], "conf")


if __name__ == "__main__":
    unittest.main()

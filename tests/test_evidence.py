"""모델 없이 근거 선택, 답변 보류, 평가 지표의 회귀를 검증합니다."""
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import config
import rag
from eval.run_eval import is_hit, summarize
from eval import run_eval
import knowledge_db


class EvidenceTest(unittest.TestCase):
    def test_filters_cosine_not_rrf_and_keeps_order(self):
        matches = [{"score": .2, "hybrid_score": .99},
                   {"score": .6}, {"score": .8}, {"score": float("nan")}]
        self.assertEqual(rag.select_evidence(matches, .6), matches[1:3])

    def test_invalid_threshold(self):
        with self.assertRaises(ValueError):
            rag.select_evidence([], -1)

    def test_vector_dimension_mismatch(self):
        with self.assertRaisesRegex(ValueError, "차원"):
            rag.similarity([1, 0], [1])

    def test_refusal_skips_generation_and_saves_no_sources(self):
        for matches in ([], [{"score": .1, "hybrid_score": 1}]):
            with self.subTest(matches=matches), \
                    mock.patch.object(rag, "search_knowledge", return_value=matches), \
                    mock.patch.object(rag, "get_messages", return_value=[]), \
                    mock.patch.object(rag, "call_ollama") as generate, \
                    mock.patch.object(rag, "save_message") as save, \
                    mock.patch.object(config, "MIN_EVIDENCE_SCORE", .45):
                answer, sources = rag.ask_ai("질문", "session")
                self.assertIn("충분한 근거", answer)
                self.assertEqual(sources, [])
                generate.assert_not_called()
                self.assertEqual(save.call_args_list, [
                    mock.call("session", "user", "질문"),
                    mock.call("session", "assistant", answer, sources=[]),
                ])

    def test_supported_answer_uses_only_accepted_sources(self):
        good = {"score": .8, "title": "설계", "content": "승인 조건", "location": "페이지 2",
                "type": "document", "document_id": 1, "chunk_id": 2, "item_id": None}
        bad = {**good, "score": .1, "content": "무관한 자료"}
        with mock.patch.object(rag, "search_knowledge", return_value=[bad, good]) as search, \
                mock.patch.object(rag, "get_messages", return_value=[]), \
                mock.patch.object(rag, "save_message") as save, \
                mock.patch.object(rag, "call_ollama", return_value={
                    "message": {"content": "승인 조건입니다."}}) as generate, \
                mock.patch.object(config, "MIN_EVIDENCE_SCORE", .45):
            answer, sources = rag.ask_ai("승인 조건?")
            search.assert_called_once_with("승인 조건?", limit=5, min_score=.45)
            self.assertEqual(sources, ["설계 (페이지 2)"])
            self.assertNotIn("무관한 자료", generate.call_args.args[1]["messages"][0]["content"])
            self.assertEqual(len(save.call_args.kwargs["sources"]), 1)

    def test_blank_question_has_no_side_effects(self):
        with mock.patch.object(rag, "search_knowledge") as search:
            with self.assertRaises(ValueError):
                rag.ask_ai("  ")
            search.assert_not_called()


class EvaluationTest(unittest.TestCase):
    def row(self, **changes):
        return {"id": "q", "gold_slides": [2], "rank": 1, "evidence_rank": 1,
                "refused": False, "top_score": .8, "retrieval_seconds": .1, **changes}

    def test_same_page_in_wrong_document_is_not_a_hit(self):
        match = {"type": "document", "page_start": 2, "page_end": 3,
                 "file_name": "a.pdf", "title": "설계"}
        self.assertTrue(is_hit(match, [3], "a.pdf"))
        self.assertFalse(is_hit(match, [3], "b.pdf"))

    def test_empty_or_negative_only_evaluation(self):
        self.assertIsNone(summarize([], 1)["mrr"])
        result = summarize([self.row(gold_slides=[], rank=None, evidence_rank=None,
                                     refused=True, top_score=None)], 1)
        self.assertIsNone(result["positive_top_score_min"])
        self.assertEqual(result["negative_refusal_rate"], 1)
        self.assertEqual(result["negative_acceptance_rate"], 0)
        self.assertNotIn("hit@3", result)

    def test_policy_metrics_include_false_refusals_and_acceptances(self):
        rows = [self.row(), self.row(rank=2, evidence_rank=None, refused=True),
                self.row(gold_slides=[], rank=None, evidence_rank=None)]
        result = summarize(rows, 5)
        self.assertEqual(result["hit@1"], .5)
        self.assertEqual(result["mrr"], .75)
        self.assertEqual(result["positive_refusal_rate"], .5)
        self.assertEqual(result["negative_acceptance_rate"], 1)
        self.assertEqual(result["accepted_hit_rate"], .5)

    def test_runner_snapshots_wal_and_writes_empty_retrieval_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.db"
            connection = sqlite3.connect(source)
            try:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("CREATE TABLE marker (value TEXT)")
                connection.execute("INSERT INTO marker VALUES ('committed')")
                connection.commit()
                questions = root / "questions.json"
                questions.write_text(json.dumps({"questions": [
                    {"id": "N1", "question": "없는 내용", "gold_slides": []}
                ]}), encoding="utf-8")
                args = SimpleNamespace(db=str(source), questions=str(questions),
                                       min_score=.45, k=1, label="test")
                with mock.patch.dict(os.environ), \
                        mock.patch.object(run_eval, "EVAL_DIR", root), \
                        mock.patch.object(knowledge_db, "active_chunks", return_value=[]), \
                        mock.patch.object(knowledge_db, "list_documents", return_value=[]), \
                        mock.patch.object(rag, "search_knowledge", return_value=[]), \
                        mock.patch("builtins.print"):
                    run_eval.run(args, root)
                snapshot = sqlite3.connect(root / "eval.db")
                try:
                    self.assertEqual(snapshot.execute("SELECT value FROM marker").fetchone(),
                                     ("committed",))
                finally:
                    snapshot.close()
                report = json.loads(next((root / "results").glob("*.json")).read_text(encoding="utf-8"))
                self.assertEqual(report["summary"]["negative_refusal_rate"], 1)
                self.assertIsNone(report["summary"]["mrr"])
                self.assertTrue(report["rows"][0]["refused"])
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()

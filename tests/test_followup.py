"""후속 질문 변환과 세션별 검색/저장을 로컬 모델 없이 검증합니다."""
import json
import unittest
from unittest import mock

from test_phase1 import TempEnv
import config
import knowledge_db as db
import rag


HISTORY = [
    {"role": "user", "content": "로그인 비밀번호 규칙은?"},
    {"role": "assistant", "content": "비밀번호 검증 규칙입니다."},
]


def rewritten(query):
    return {"message": {"content": json.dumps({"query": query}, ensure_ascii=False)}}


class RewriteTest(unittest.TestCase):
    def test_first_question_and_disabled_mode_skip_model(self):
        with mock.patch.object(rag, "call_ollama") as call:
            self.assertEqual(rag.rewrite_search_query("첫 질문", []), "첫 질문")
            with mock.patch.object(config, "REWRITE_QUERY", False):
                self.assertEqual(rag.rewrite_search_query("그럼?", HISTORY), "그럼?")
            call.assert_not_called()

    def test_followup_receives_recent_history_and_original_question(self):
        query = "로그인 비밀번호 검증 실패 시 처리 절차는?"
        with mock.patch.object(config, "REWRITE_QUERY", True), \
                mock.patch.object(rag, "call_ollama", return_value=rewritten(query)) as call:
            self.assertEqual(rag.rewrite_search_query("그럼 실패하면?", HISTORY), query)
            payload = call.call_args.args[1]
            data = json.loads(payload["messages"][1]["content"])
            self.assertEqual(data, {"history": HISTORY, "question": "그럼 실패하면?"})
            self.assertFalse(payload["think"])

    def test_topic_change_preserves_independent_query(self):
        question = "결제 RQ-PAY-003의 재시도 금지 조건은?"
        with mock.patch.object(config, "REWRITE_QUERY", True), \
                mock.patch.object(rag, "call_ollama", return_value=rewritten(question)):
            self.assertEqual(rag.rewrite_search_query(question, HISTORY), question)

    def test_failed_or_invalid_response_falls_back_to_original(self):
        responses = [{}, {"message": {"content": "JSON 아님"}}, rewritten(" "),
                     rewritten(None), rewritten("x" * 1001),
                     {"message": {"content": "[]"}}]
        with mock.patch.object(config, "REWRITE_QUERY", True):
            for response in responses:
                with self.subTest(response=response), \
                        mock.patch.object(rag, "call_ollama", return_value=response):
                    self.assertEqual(rag.rewrite_search_query("원문", HISTORY), "원문")
            with mock.patch.object(rag, "call_ollama", side_effect=rag.OllamaError("timeout")):
                self.assertEqual(rag.rewrite_search_query("원문", HISTORY), "원문")

    def test_history_budget_and_role_filter(self):
        history = [{"role": "user", "content": str(i) + "x" * 2000} for i in range(10)]
        history.append({"role": "system", "content": "이전 지시"})
        with mock.patch.object(config, "REWRITE_QUERY", True), \
                mock.patch.object(rag, "call_ollama", return_value=rewritten("검색")) as call:
            rag.rewrite_search_query("그럼?", history)
            recent = json.loads(call.call_args.args[1]["messages"][1]["content"])["history"]
            self.assertEqual(len(recent), 6)
            self.assertTrue(recent[0]["content"].startswith("4"))
            self.assertTrue(all(len(m["content"]) <= 1000 for m in recent))


class FollowupChatTest(TempEnv):
    def setUp(self):
        super().setUp()
        db.init_db()
        self.patches.append(mock.patch.object(config, "REWRITE_QUERY", True))
        self.patches[-1].start()

    def test_search_uses_rewrite_but_answer_and_storage_use_original(self):
        for message in HISTORY:
            db.save_message("login", message["role"], message["content"])
        db.save_message("other", "user", "다른 세션의 결제 질문")
        evidence = {"score": 1, "title": "인증", "content": "검증 실패 시 거절",
                    "location": "페이지 3", "type": "document", "document_id": 1,
                    "chunk_id": 1, "item_id": None}
        query = "로그인 비밀번호 검증 실패 처리"
        with mock.patch.object(rag, "search_knowledge", return_value=[evidence]) as search, \
                mock.patch.object(rag, "call_ollama", side_effect=[
                    rewritten(query), {"message": {"content": "실패하면 거절합니다."}}
                ]) as call:
            answer, sources = rag.ask_ai("그럼 실패하면?", session_id="login")
            search.assert_called_once_with(query, limit=5, min_score=config.MIN_EVIDENCE_SCORE)
            rewrite_data = json.loads(call.call_args_list[0].args[1]["messages"][1]["content"])
            self.assertEqual(rewrite_data["history"], HISTORY)
            answer_messages = call.call_args_list[1].args[1]["messages"]
            self.assertEqual(answer_messages[-1]["content"], "/no_think\n그럼 실패하면?")
            self.assertEqual(answer_messages[1:-1], HISTORY)
        saved = db.get_messages("login")
        self.assertEqual(saved[-2]["content"], "그럼 실패하면?")
        self.assertEqual(saved[-1]["content"], answer)
        self.assertTrue(sources)
        self.assertEqual(len(db.get_messages("other")), 1)

    def test_new_session_has_no_rewrite_even_when_another_session_has_history(self):
        db.save_message("old", "user", "이전 주제")
        with mock.patch.object(rag, "search_knowledge", return_value=[]) as search, \
                mock.patch.object(rag, "call_ollama") as call:
            rag.ask_ai("새 질문", session_id="new")
            call.assert_not_called()
            self.assertEqual(search.call_args.args[0], "새 질문")

    def test_rewrite_error_still_searches_and_records_refusal(self):
        db.save_message("login", "user", "로그인 규칙")
        with mock.patch.object(rag, "call_ollama", side_effect=rag.OllamaError("실패")), \
                mock.patch.object(rag, "search_knowledge", return_value=[]) as search:
            answer, sources = rag.ask_ai("그 예외는?", session_id="login")
            self.assertEqual(search.call_args.args[0], "그 예외는?")
            self.assertEqual(sources, [])
            self.assertIn("충분한 근거", answer)
        self.assertEqual(db.get_messages("login")[-2]["content"], "그 예외는?")


if __name__ == "__main__":
    unittest.main()

"""Streamlit 화면과 세션 전환을 임시 DB에서 검증합니다."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import knowledge_db as db
from streamlit.testing.v1 import AppTest


class WebTest(unittest.TestCase):
    def test_history_evidence_and_new_session(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(db, "DB_PATH", Path(directory) / "web.db"):
            db.init_db()
            session = db.create_session("기존 대화")
            db.save_message(session, "user", "질문")
            db.save_message(session, "assistant", "답변", sources=[{
                "label": "설계.pdf (페이지 2)", "score": .8,
                "content": "답변 시점의 근거 원문",
            }])
            app = AppTest.from_file(
                str(Path(__file__).resolve().parents[1] / "web_app.py")
            ).run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.chat_message), 2)
            self.assertTrue(any(t.value == "답변 시점의 근거 원문" for t in app.text))
            next(b for b in app.button if b.label == "새 대화").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.chat_message), 0)
            self.assertEqual(len(db.list_sessions()), 2)

    def test_chat_submission_saves_and_renders_answer(self):
        def answer(question, session_id):
            db.save_message(session_id, "user", question)
            db.save_message(session_id, "assistant", "문서 기반 답변")
            return "문서 기반 답변", []

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(db, "DB_PATH", Path(directory) / "web.db"), \
                patch("rag.ask_ai", side_effect=answer) as ask:
            app = AppTest.from_file(
                str(Path(__file__).resolve().parents[1] / "web_app.py")
            ).run()
            app.chat_input[0].set_value("승인 조건은?").run()
            self.assertFalse(app.exception)
            ask.assert_called_once()
            self.assertEqual(len(app.chat_message), 2)
            self.assertEqual(db.list_sessions()[0]["title"], "승인 조건은?")

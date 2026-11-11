"""Phase 1 테스트: DB 마이그레이션, 문서 CRUD/버전, 검증, 지식/대화 CRUD.

실행: .venv\\Scripts\\python.exe -m unittest discover -s tests -v
실제 ai_data.db는 절대 수정하지 않습니다 (백업 사본/임시 DB만 사용).
"""
import glob
import hashlib
import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
import knowledge_db as db  # noqa: E402
import document_service as docs  # noqa: E402
import rag  # noqa: E402
from document_loader import (  # noqa: E402
    DocumentValidationError, chunk_segments, read_segments, safe_file_name, validate_upload,
)


def fake_embed(texts):
    """결정적 가짜 임베딩 (Ollama 없이 테스트)."""
    vectors = []
    for text in texts:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vectors.append([b / 255.0 for b in digest[:16]])
    return vectors


def table_fingerprint(path, table):
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = conn.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()
    finally:
        conn.close()
    return len(rows), hashlib.sha256(repr(rows).encode()).hexdigest()


def make_pdf(path, pages):
    """텍스트가 들어있는 최소 PDF를 직접 생성합니다."""
    objects = ["<< /Type /Catalog /Pages 2 0 R >>"]
    kids = " ".join(f"{3 + i * 2} 0 R" for i in range(len(pages)))
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>")
    font_id = 3 + len(pages) * 2
    for i, text in enumerate(pages):
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                       f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {4 + i * 2} 0 R >>")
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = b"%PDF-1.4\n"
    offsets = []
    for n, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n{obj}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    Path(path).write_bytes(out)


class TempEnv(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mobility_test_"))
        self.patches = [
            mock.patch.object(db, "DB_PATH", self.tmp / "test.db"),
            mock.patch.object(config, "DOCUMENT_STORE_DIR", self.tmp / "store"),
            mock.patch.object(config, "BACKUP_DIR", self.tmp / "backups"),
            mock.patch.object(rag, "embed", fake_embed),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
class MigrationTest(TempEnv):
    def copy_real_backup(self):
        backups = sorted(glob.glob(str(ROOT / "backups" / "ai_data_*.db")))
        if not backups:
            self.skipTest("실제 DB 백업 파일이 없습니다.")
        # 마이그레이션 전 백업(가장 오래된 v0 상태)을 사용
        source = None
        for b in backups:
            conn = sqlite3.connect(f"file:{b}?mode=ro", uri=True)
            v = conn.execute("PRAGMA user_version").fetchone()[0]
            conn.close()
            if v == 0:
                source = b
                break
        if not source:
            self.skipTest("v0 상태의 백업이 없습니다.")
        shutil.copy(source, db.DB_PATH)

    def test_migrate_real_data_copy(self):
        self.copy_real_backup()
        before_k = table_fingerprint(db.DB_PATH, "knowledge")
        before_c = table_fingerprint(db.DB_PATH, "conversations")

        applied = db.init_db()
        self.assertEqual(len(applied), 1)
        self.assertEqual(db.get_schema_version(), 1)

        # 기존 테이블은 한 글자도 바뀌지 않아야 함
        self.assertEqual(table_fingerprint(db.DB_PATH, "knowledge"), before_k)
        self.assertEqual(table_fingerprint(db.DB_PATH, "conversations"), before_c)

        # 자동 백업 생성
        self.assertEqual(len(list((self.tmp / "backups").glob("*before_migration*.db"))), 1)

        conn = sqlite3.connect(db.DB_PATH)
        legacy_docs = conn.execute("SELECT count(*) FROM knowledge WHERE kind='document'").fetchone()[0]
        legacy_other = conn.execute("SELECT count(*) FROM knowledge WHERE kind!='document'").fetchone()[0]
        legacy_msgs = conn.execute("SELECT count(*) FROM conversations").fetchone()[0]
        self.assertEqual(conn.execute("SELECT count(*) FROM document_chunks").fetchone()[0], legacy_docs)
        self.assertEqual(conn.execute("SELECT count(*) FROM knowledge_items").fetchone()[0], legacy_other)
        self.assertEqual(conn.execute("SELECT count(*) FROM chat_messages").fetchone()[0], legacy_msgs)

        # 모든 청크 내용이 원본과 동일하게 복사되었는지
        mismatch = conn.execute(
            """SELECT count(*) FROM document_chunks c JOIN knowledge k ON k.id = c.legacy_knowledge_id
               WHERE c.content != k.content""").fetchone()[0]
        self.assertEqual(mismatch, 0)
        conn.close()

        # 임베딩 변환 (JSON float64 → float32 BLOB) 정확도
        conn = sqlite3.connect(db.DB_PATH)
        blob, js = conn.execute(
            """SELECT c.embedding, k.embedding FROM document_chunks c
               JOIN knowledge k ON k.id = c.legacy_knowledge_id LIMIT 1""").fetchone()
        conn.close()
        import json
        a, b = list(db.unpack_vector(blob)), json.loads(js)
        self.assertEqual(len(a), len(b))
        self.assertGreater(rag.similarity(a, b), 0.99999)

        # PPTX 슬라이드 번호 추정
        doc = db.list_documents()[0]
        chunks = db.get_document_chunks(doc["id"])
        self.assertTrue(all(c["page_start"] is not None for c in chunks))
        self.assertEqual(chunks[0]["page_start"], 1)
        self.assertTrue(all(c["page_start"] <= c["page_end"] for c in chunks))
        starts = [c["page_start"] for c in chunks]
        self.assertEqual(starts, sorted(starts))

        # 재실행해도 아무 일도 일어나지 않음 (멱등성)
        self.assertEqual(db.init_db(), [])
        self.assertEqual(len(list((self.tmp / "backups").glob("*.db"))), 1)

    def test_fresh_db(self):
        self.assertEqual(len(db.init_db()), 1)
        stats = db.db_stats()
        self.assertEqual(stats["schema_version"], 1)
        self.assertEqual(stats["documents_total"], 0)
        self.assertFalse((self.tmp / "backups").exists())  # 빈 DB는 백업 불필요

    def test_migration_failure_rolls_back(self):
        self.copy_real_backup()
        before = table_fingerprint(db.DB_PATH, "knowledge")
        with mock.patch.object(db, "_infer_legacy_pages", side_effect=RuntimeError("실패 유도")):
            with self.assertRaises(RuntimeError):
                db.init_db()
        self.assertEqual(db.get_schema_version(), 0)
        conn = sqlite3.connect(db.DB_PATH)
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        conn.close()
        self.assertNotIn("documents", tables)
        self.assertEqual(table_fingerprint(db.DB_PATH, "knowledge"), before)

    def test_infer_legacy_pages(self):
        chunks = ["[슬라이드 1]\nA\n\n[슬라이드 2]\nB", "B 이어짐", "C\n[슬라이드 3]\nD", "[슬라이드 4]\nE"]
        self.assertEqual(db._infer_legacy_pages(chunks), [(1, 2), (2, 2), (2, 3), (4, 4)])


# ---------------------------------------------------------------------------
class DocumentTest(TempEnv):
    def setUp(self):
        super().setUp()
        db.init_db()
        self.files = self.tmp / "files"
        self.files.mkdir()

    def write(self, name, text):
        path = self.files / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_register_duplicate_version_delete(self):
        path = self.write("sample_doc.txt", "샘플 관리 기능은 항목 등록, 수정, 삭제를 다룹니다.\n" * 5)
        r1 = docs.register_file(path, embedder=fake_embed)
        self.assertEqual(r1["status"], "created")
        d1 = r1["document"]
        self.assertTrue(docs.stored_file_path(d1).is_file())

        # 같은 내용 → 중복 거부 (파일명이 달라도)
        dup = self.write("복사본.txt", path.read_text(encoding="utf-8"))
        r_dup = docs.register_file(dup, embedder=fake_embed)
        self.assertEqual(r_dup["status"], "duplicate")
        self.assertEqual(db.db_stats()["documents_total"], 1)

        # 같은 파일명, 다른 내용 → 새 버전
        path.write_text("샘플 관리 v2: 항목 변경 이력 관리가 추가되었습니다.", encoding="utf-8")
        r2 = docs.register_file(path, embedder=fake_embed)
        self.assertEqual(r2["status"], "new_version")
        d2 = r2["document"]
        self.assertEqual(d2["version"], 2)
        self.assertEqual(d2["doc_uid"], d1["doc_uid"])
        self.assertFalse(db.get_document(d1["id"])["is_active"])
        self.assertEqual([d["id"] for d in db.list_documents()], [d2["id"]])
        self.assertEqual(len(db.list_documents(include_inactive=True)), 2)

        # 검색은 활성 버전만
        self.assertTrue(all(c["document_id"] == d2["id"] for c in db.active_chunks()))

        # 이전 버전 재활성화
        self.assertTrue(db.set_active_version(d1["id"]))
        self.assertEqual([d["id"] for d in db.list_documents()], [d1["id"]])

        # 활성 버전 삭제 → 남은 최신 버전 자동 활성화, 저장 파일 제거
        stored1 = docs.stored_file_path(db.get_document(d1["id"]))
        self.assertEqual(docs.delete_document(d1["id"]), 1)
        self.assertFalse(stored1.exists())
        self.assertTrue(db.get_document(d2["id"])["is_active"])
        self.assertEqual(db.get_document_chunks(d1["id"]), [])  # CASCADE

        # 문서 전체 삭제
        path.write_text("v3 내용", encoding="utf-8")
        r3 = docs.register_file(path, embedder=fake_embed)
        self.assertEqual(r3["document"]["version"], 3)
        self.assertEqual(docs.delete_document(r3["document"]["id"], all_versions=True), 2)
        self.assertEqual(db.db_stats()["documents_total"], 0)
        self.assertEqual(db.db_stats()["chunks"], 0)
        self.assertEqual(list((self.tmp / "store").iterdir()), [])

    def test_explicit_replace_and_meta_update(self):
        a = docs.register_file(self.write("a.txt", "가 내용"), embedder=fake_embed)["document"]
        b = docs.register_file(self.write("b.md", "# 나\n내용"), embedder=fake_embed,
                               replace_document_id=a["id"])
        self.assertEqual(b["status"], "new_version")
        self.assertEqual(b["document"]["doc_uid"], a["doc_uid"])
        self.assertTrue(db.update_document_meta(b["document"]["id"], title="새 제목", category="샘플"))
        self.assertEqual(db.get_document(b["document"]["id"])["title"], "새 제목")

    def test_reindex(self):
        d = docs.register_file(self.write("r.txt", "재임베딩 테스트 " * 200), embedder=fake_embed)["document"]
        before = db.get_document_chunks(d["id"])
        result = docs.reindex_document(d["id"], embedder=fake_embed)
        self.assertEqual(result["chunks"], len(before))
        self.assertIn("원본", result["mode"])
        # 원본 파일이 없으면 저장 텍스트로 재임베딩
        docs.stored_file_path(d).unlink()
        result = docs.reindex_document(d["id"], embedder=fake_embed)
        self.assertIn("원본 파일 없음", result["mode"])

    def test_upload_bytes(self):
        r = docs.register_upload("../../evil/sample_rule.md", "# 규칙\n샘플 신청 절차".encode(),
                                 embedder=fake_embed)
        self.assertEqual(r["document"]["file_name"], "sample_rule.md")
        self.assertEqual(r["document"]["source"], "upload")

    def test_pptx_slide_numbers(self):
        pptx = ROOT / "documents" / "sample.pptx"
        if not pptx.exists():
            self.skipTest("sample.pptx 없음")
        r = docs.register_file(pptx, embedder=fake_embed)
        chunks = db.get_document_chunks(r["document"]["id"])
        self.assertGreater(len(chunks), 0)
        for c in chunks:
            self.assertTrue(1 <= c["page_start"] <= c["page_end"] <= 145)
            if c["content"].startswith("[슬라이드"):
                self.assertTrue(c["content"].startswith(f"[슬라이드 {c['page_start']}]"))
        self.assertLessEqual(max(c["char_count"] for c in chunks), config.CHUNK_SIZE + 20)

    def test_pdf_page_numbers(self):
        pdf = self.tmp / "files" / "plan.pdf"
        make_pdf(pdf, ["Alpha management page one", "Product management page two"])
        segments = read_segments(pdf)
        self.assertEqual([s["page"] for s in segments], [1, 2])
        self.assertIn("Product", segments[1]["text"])
        chunks = chunk_segments(segments, size=45, unit="페이지")
        self.assertEqual([(c["page_start"], c["page_end"]) for c in chunks], [(1, 1), (2, 2)])
        self.assertTrue(chunks[1]["content"].startswith("[페이지 2]"))
        # 긴 페이지 분할: overlap이 size보다 커도 조각 수가 폭증하지 않아야 함
        long_pieces = chunk_segments([{"page": 7, "text": "가" * 300}], size=100, overlap=500)
        self.assertLessEqual(len(long_pieces), 5)
        self.assertTrue(all(p["page_start"] == 7 for p in long_pieces))
        merged = chunk_segments(segments, size=800, unit="페이지")
        self.assertEqual([(c["page_start"], c["page_end"]) for c in merged], [(1, 2)])

    def test_docx_tables_extracted(self):
        from docx import Document
        path = self.files / "spec.docx"
        d = Document()
        d.add_paragraph("샘플 항목 관리")
        t = d.add_table(rows=1, cols=2)
        t.cell(0, 0).text = "항목번호"
        t.cell(0, 1).text = "2025-001"
        d.save(path)
        text = read_segments(path)[0]["text"]
        self.assertIn("샘플 항목 관리", text)
        self.assertIn("항목번호 | 2025-001", text)

    def test_validation(self):
        with self.assertRaises(DocumentValidationError):
            validate_upload("malware.exe", 10)
        with self.assertRaises(DocumentValidationError):
            validate_upload("big.pdf", config.MAX_UPLOAD_BYTES + 1)
        with self.assertRaises(DocumentValidationError):
            validate_upload("empty.txt", 0)
        self.assertEqual(safe_file_name("..\\..\\Windows\\a:b?.txt"), "a_b_.txt")
        with self.assertRaises(DocumentValidationError):
            docs.register_file(self.write("x.csv", "a,b"), embedder=fake_embed)
        # DB의 저장 경로가 조작되어도 저장소 밖 파일에 접근하지 않음
        with self.assertRaises(DocumentValidationError):
            docs.stored_file_path({"stored_path": "../../config.py"})


# ---------------------------------------------------------------------------
class KnowledgeAndChatTest(TempEnv):
    def setUp(self):
        super().setUp()
        db.init_db()

    def test_knowledge_crud(self):
        [kid] = rag.add_text("memory", "선호", "답변은 표 형식을 선호합니다.")
        self.assertEqual(db.get_knowledge_item(kid)["kind"], "memory")
        self.assertTrue(db.update_knowledge_item(kid, "선호", "답변은 목록 형식", fake_embed(["x"])[0]))
        self.assertEqual(db.get_knowledge_item(kid)["content"], "답변은 목록 형식")
        self.assertEqual(db.list_knowledge(), [(kid, "memory", "선호")])  # 호환 API
        self.assertTrue(db.delete_knowledge_item(kid))
        self.assertIsNone(db.get_knowledge_item(kid))
        with self.assertRaises(ValueError):
            rag.add_text("note", "빈", "   ")

    def test_sessions(self):
        sid = db.create_session("샘플 세션", mode="transport")
        db.save_message(sid, "user", "질문")
        db.save_message(sid, "assistant", "답변", sources=[{"label": "sample.pptx (슬라이드 3)"}])
        msgs = db.get_messages(sid, with_meta=True)
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant"])
        self.assertEqual(msgs[1]["sources"][0]["label"], "sample.pptx (슬라이드 3)")
        self.assertEqual(db.list_sessions(mode="transport")[0]["message_count"], 2)
        self.assertTrue(db.rename_session(sid, "이름 변경"))
        self.assertTrue(db.delete_session(sid))
        self.assertEqual(db.get_messages(sid), [])  # CASCADE

    def test_search_includes_location(self):
        docs.register_upload("규칙.txt", "샘플 신청 규칙".encode(), embedder=fake_embed)
        rag.add_text("note", "메모", "샘플 메모")
        results = rag.search_knowledge("샘플", limit=10)
        self.assertEqual({r["type"] for r in results}, {"document", "knowledge"})


# ---------------------------------------------------------------------------
def ollama_available():
    try:
        import requests
        tags = requests.get(config.OLLAMA_BASE_URL + "/api/tags", timeout=2).json()
        names = {m["name"].split(":")[0] for m in tags.get("models", [])}
        return {"bge-m3", "qwen3"} <= names
    except Exception:
        return False


@unittest.skipUnless(ollama_available(), "Ollama 또는 모델이 실행 중이 아님")
class OllamaIntegrationTest(MigrationTest):
    """실제 Ollama로 마이그레이션된 데이터(백업 사본)를 검색/답변합니다."""

    def setUp(self):
        super().setUp()
        self.patches[-1].stop()  # 실제 embed 사용
        self.patches.pop()

    def test_migrate_real_data_copy(self):  # 부모 테스트 중복 실행 방지
        pass

    def test_fresh_db(self):
        pass

    def test_migration_failure_rolls_back(self):
        pass

    def test_search_and_answer_on_migrated_copy(self):
        self.copy_real_backup()
        db.init_db()
        results = rag.search_knowledge("로그인 비밀번호 규칙", limit=3)
        self.assertEqual(len(results), 3)
        self.assertTrue(any(r["location"] and r["location"].startswith("슬라이드") for r in results))
        answer, sources = rag.ask_ai("로그인 비밀번호 규칙을 알려줘", session_id="test-session")
        self.assertTrue(answer)
        self.assertNotIn("<think>", answer)
        self.assertTrue(sources)
        self.assertEqual(len(db.get_messages("test-session")), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)

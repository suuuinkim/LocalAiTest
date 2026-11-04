"""SQLite 저장소 계층 (문서 / 청크 / 지식 / 대화).

스키마 버전 (PRAGMA user_version)
  0 : 기존 구조 (knowledge, conversations)
  1 : documents, document_chunks, knowledge_items, chat_sessions,
      chat_messages, migration_log 추가

마이그레이션 원칙
  * 마이그레이션 전 자동 백업 (backups/ 폴더)
  * 기존 테이블(knowledge, conversations)은 삭제·변경하지 않고 그대로 보존
  * 기존 데이터는 새 테이블로 '복사'만 함
  * 하나의 트랜잭션으로 실행 → 실패 시 전체 롤백
"""
import hashlib
import json
import re
import sqlite3
import uuid
from array import array
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import config

DB_PATH = config.DB_PATH  # 테스트에서 덮어쓸 수 있음
SCHEMA_VERSION = 1

# 출처 표시용 위치 단위
LOCATION_UNIT = {"pptx": "슬라이드", "pdf": "페이지"}


# ---------------------------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------------------------
def now():
    return datetime.now().isoformat(timespec="seconds")


@contextmanager
def connect():
    """커밋/롤백 후 반드시 연결을 닫는 컨텍스트 매니저."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def pack_vector(vector):
    return array("f", vector).tobytes()


def unpack_vector(blob):
    values = array("f")
    values.frombytes(blob)
    return values


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def format_location(file_type, page_start, page_end):
    """예: '슬라이드 3', '페이지 5-6'. 위치 정보가 없으면 None."""
    if page_start is None:
        return None
    unit = LOCATION_UNIT.get(file_type, "위치")
    if page_end is None or page_end == page_start:
        return f"{unit} {page_start}"
    return f"{unit} {page_start}-{page_end}"


# ---------------------------------------------------------------------------
# 스키마 & 마이그레이션
# ---------------------------------------------------------------------------
SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS documents (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_uid       TEXT    NOT NULL,              -- 버전과 무관한 문서 고유 ID
    version       INTEGER NOT NULL DEFAULT 1,
    is_active     INTEGER NOT NULL DEFAULT 1,    -- 검색 대상(최신 사용 버전)
    title         TEXT    NOT NULL,
    file_name     TEXT,
    file_type     TEXT    NOT NULL,              -- pdf/pptx/docx/txt/md
    file_size     INTEGER,
    content_hash  TEXT    NOT NULL,              -- 중복 등록 방지
    stored_path   TEXT,                          -- DOCUMENT_STORE_DIR 기준 상대 경로
    category      TEXT,
    note          TEXT,
    status        TEXT    NOT NULL DEFAULT 'ready',  -- ready / legacy / failed
    source        TEXT    NOT NULL DEFAULT 'upload', -- upload / cli / legacy_migration
    chunk_count   INTEGER NOT NULL DEFAULT 0,
    embed_model   TEXT,
    created_at    TEXT    NOT NULL,
    updated_at    TEXT    NOT NULL,
    UNIQUE (doc_uid, version)
);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash);
CREATE INDEX IF NOT EXISTS idx_documents_uid ON documents(doc_uid);

CREATE TABLE IF NOT EXISTS document_chunks (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id         INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index         INTEGER NOT NULL,
    content             TEXT    NOT NULL,
    page_start          INTEGER,        -- PDF 페이지 / PPTX 슬라이드 번호
    page_end            INTEGER,
    char_count          INTEGER NOT NULL,
    embedding           BLOB    NOT NULL, -- float32 배열
    legacy_knowledge_id INTEGER,          -- 기존 knowledge.id (마이그레이션 추적)
    created_at          TEXT    NOT NULL,
    UNIQUE (document_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON document_chunks(document_id);

CREATE TABLE IF NOT EXISTS knowledge_items (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    kind                TEXT    NOT NULL,   -- note / memory / term / rule
    title               TEXT    NOT NULL,
    content             TEXT    NOT NULL,
    tags                TEXT,
    embedding           BLOB    NOT NULL,
    source              TEXT    NOT NULL DEFAULT 'user', -- user / conversation / legacy
    source_message_id   INTEGER,
    legacy_knowledge_id INTEGER,
    created_at          TEXT    NOT NULL,
    updated_at          TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    mode        TEXT NOT NULL DEFAULT 'general', -- general/transport/planning/springboot
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role        TEXT    NOT NULL,   -- user / assistant
    content     TEXT    NOT NULL,
    sources     TEXT,               -- JSON: 답변 근거 목록
    created_at  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON chat_messages(session_id, id);

CREATE TABLE IF NOT EXISTS migration_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    version     INTEGER NOT NULL,
    description TEXT    NOT NULL,
    backup_path TEXT,
    applied_at  TEXT    NOT NULL
);
"""


def _table_exists(conn, name):
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _infer_legacy_pages(chunks):
    """기존 700자 청크의 '[슬라이드 N]' 표식으로 슬라이드 범위를 추정합니다."""
    marker = re.compile(r"\[슬라이드 (\d+)\]")
    current = None
    result = []
    for content in chunks:
        numbers = [int(n) for n in marker.findall(content)]
        starts_with_marker = content.lstrip().startswith("[슬라이드")
        start = numbers[0] if numbers and (starts_with_marker or current is None) else current
        end = numbers[-1] if numbers else current
        if start is None:
            start = end
        result.append((start, end))
        if numbers:
            current = numbers[-1]
    return result


def _create_schema_v1(conn):
    # executescript()는 진행 중인 트랜잭션을 암묵적으로 COMMIT하므로 사용하지 않음
    for statement in SCHEMA_V1.split(";"):
        if statement.strip():
            conn.execute(statement)


def _migrate_v0_to_v1(conn, backup_path):
    _create_schema_v1(conn)
    ts = now()
    copied_docs = copied_items = copied_msgs = 0

    if _table_exists(conn, "knowledge"):
        rows = conn.execute(
            "SELECT id, kind, title, content, embedding FROM knowledge ORDER BY id"
        ).fetchall()

        # 1) 문서 조각 → documents + document_chunks (제목 'xxx #N' 기준 그룹화)
        groups = {}
        for row in rows:
            if row["kind"] != "document":
                continue
            base = re.sub(r" #\d+$", "", row["title"])
            groups.setdefault(base, []).append(row)

        for base, parts in groups.items():
            def order(r):
                m = re.search(r" #(\d+)$", r["title"])
                return int(m.group(1)) if m else 0
            parts.sort(key=order)
            suffix = Path(base).suffix.lower().lstrip(".") or "txt"
            joined = "".join(p["content"] for p in parts)
            cur = conn.execute(
                """INSERT INTO documents (doc_uid, version, is_active, title, file_name,
                       file_type, content_hash, status, source, chunk_count,
                       embed_model, note, created_at, updated_at)
                   VALUES (?, 1, 1, ?, ?, ?, ?, 'legacy', 'legacy_migration', ?, ?, ?, ?, ?)""",
                (uuid.uuid4().hex, base, base, suffix,
                 "legacy-text:" + sha256_text(joined), len(parts), config.EMBED_MODEL,
                 "기존 knowledge 테이블에서 이전된 문서 (재등록 시 정확한 페이지/슬라이드 정보로 갱신 권장)",
                 ts, ts),
            )
            doc_id = cur.lastrowid
            pages = _infer_legacy_pages([p["content"] for p in parts]) if suffix == "pptx" \
                else [(None, None)] * len(parts)
            for index, (part, (p_start, p_end)) in enumerate(zip(parts, pages)):
                conn.execute(
                    """INSERT INTO document_chunks (document_id, chunk_index, content,
                           page_start, page_end, char_count, embedding,
                           legacy_knowledge_id, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (doc_id, index, part["content"], p_start, p_end,
                     len(part["content"]), pack_vector(json.loads(part["embedding"])),
                     part["id"], ts),
                )
            copied_docs += 1

        # 2) note / memory 등 → knowledge_items
        for row in rows:
            if row["kind"] == "document":
                continue
            conn.execute(
                """INSERT INTO knowledge_items (kind, title, content, embedding, source,
                       legacy_knowledge_id, created_at, updated_at)
                   VALUES (?, ?, ?, ?, 'legacy', ?, ?, ?)""",
                (row["kind"], row["title"], row["content"],
                 pack_vector(json.loads(row["embedding"])), row["id"], ts, ts),
            )
            copied_items += 1

    # 3) 기존 conversations → chat_sessions + chat_messages
    if _table_exists(conn, "conversations"):
        rows = conn.execute(
            "SELECT id, session_id, role, content FROM conversations ORDER BY id"
        ).fetchall()
        seen = set()
        for row in rows:
            sid = row["session_id"]
            if sid not in seen:
                first_user = next(
                    (r["content"] for r in rows if r["session_id"] == sid and r["role"] == "user"),
                    sid,
                )
                conn.execute(
                    """INSERT OR IGNORE INTO chat_sessions (id, title, mode, created_at, updated_at)
                       VALUES (?, ?, 'general', ?, ?)""",
                    (sid, f"[이전] {first_user[:30]}", ts, ts),
                )
                seen.add(sid)
            conn.execute(
                """INSERT INTO chat_messages (session_id, role, content, created_at)
                   VALUES (?, ?, ?, ?)""",
                (sid, row["role"], row["content"], ts),
            )
            copied_msgs += 1

    description = (
        f"v1 스키마 생성 및 기존 데이터 복사 (문서 {copied_docs}건, "
        f"지식 {copied_items}건, 메시지 {copied_msgs}건). 기존 테이블은 보존."
    )
    conn.execute(
        "INSERT INTO migration_log (version, description, backup_path, applied_at) VALUES (?, ?, ?, ?)",
        (1, description, str(backup_path) if backup_path else None, ts),
    )
    return description


def get_schema_version():
    with connect() as conn:
        return conn.execute("PRAGMA user_version").fetchone()[0]


def init_db(backup=True):
    """스키마를 최신 버전으로 맞춥니다. 필요할 때만 백업 후 마이그레이션합니다.

    반환값: 실행된 마이그레이션 설명 목록 (이미 최신이면 빈 목록)
    """
    from backup_db import backup_database

    db_exists = Path(DB_PATH).exists()
    with connect() as conn:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        has_data = conn.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table'"
        ).fetchone()[0] > 0

    if version >= SCHEMA_VERSION:
        return []

    backup_path = None
    if backup and db_exists and has_data:
        backup_path = backup_database(DB_PATH, label=f"before_migration_v{SCHEMA_VERSION}")

    applied = []
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.isolation_level = None  # 트랜잭션 직접 제어
        conn.execute("BEGIN IMMEDIATE")
        # 다른 프로세스가 먼저 마이그레이션했는지 재확인
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version < 1:
            applied.append(_migrate_v0_to_v1(conn, backup_path))
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    return applied


# ---------------------------------------------------------------------------
# 문서 (documents / document_chunks)
# ---------------------------------------------------------------------------
def find_document_by_hash(content_hash):
    """동일한 내용의 문서(모든 버전 포함)를 찾습니다."""
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE content_hash=? ORDER BY is_active DESC, id DESC LIMIT 1",
            (content_hash,),
        ).fetchone()
        return dict(row) if row else None


def find_latest_by_file_name(file_name):
    """같은 파일명의 최신 문서(버전 계보 판단용)."""
    with connect() as conn:
        row = conn.execute(
            """SELECT * FROM documents WHERE file_name=?
               ORDER BY is_active DESC, version DESC, id DESC LIMIT 1""",
            (file_name,),
        ).fetchone()
        return dict(row) if row else None


def next_version(doc_uid):
    with connect() as conn:
        return conn.execute(
            "SELECT COALESCE(MAX(version), 0) + 1 FROM documents WHERE doc_uid=?", (doc_uid,)
        ).fetchone()[0]


def _insert_chunks(conn, document_id, chunks, ts):
    for index, chunk in enumerate(chunks):
        conn.execute(
            """INSERT INTO document_chunks (document_id, chunk_index, content, page_start,
                   page_end, char_count, embedding, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (document_id, index, chunk["content"], chunk.get("page_start"),
             chunk.get("page_end"), len(chunk["content"]), pack_vector(chunk["vector"]), ts),
        )


def create_document(meta, chunks):
    """문서와 청크를 한 트랜잭션으로 저장합니다.

    meta: doc_uid, version, title, file_name, file_type, file_size, content_hash,
          stored_path, category, note, source, embed_model
    chunks: [{"content", "page_start", "page_end", "vector"}]
    같은 doc_uid의 이전 버전은 비활성화됩니다(삭제하지 않음).
    """
    ts = now()
    with connect() as conn:
        conn.execute("UPDATE documents SET is_active=0, updated_at=? WHERE doc_uid=?",
                     (ts, meta["doc_uid"]))
        cur = conn.execute(
            """INSERT INTO documents (doc_uid, version, is_active, title, file_name, file_type,
                   file_size, content_hash, stored_path, category, note, status, source,
                   chunk_count, embed_model, created_at, updated_at)
               VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, 'ready', ?, ?, ?, ?, ?)""",
            (meta["doc_uid"], meta["version"], meta["title"], meta.get("file_name"),
             meta["file_type"], meta.get("file_size"), meta["content_hash"],
             meta.get("stored_path"), meta.get("category"), meta.get("note"),
             meta.get("source", "upload"), len(chunks), meta.get("embed_model"), ts, ts),
        )
        doc_id = cur.lastrowid
        _insert_chunks(conn, doc_id, chunks, ts)
        return doc_id


def replace_chunks(document_id, chunks, embed_model=None, status="ready"):
    """재임베딩: 해당 문서 버전의 청크를 원자적으로 교체합니다."""
    ts = now()
    with connect() as conn:
        conn.execute("DELETE FROM document_chunks WHERE document_id=?", (document_id,))
        _insert_chunks(conn, document_id, chunks, ts)
        conn.execute(
            """UPDATE documents SET chunk_count=?, embed_model=COALESCE(?, embed_model),
                   status=?, updated_at=? WHERE id=?""",
            (len(chunks), embed_model, status, ts, document_id),
        )


def list_documents(include_inactive=False):
    sql = "SELECT * FROM documents"
    if not include_inactive:
        sql += " WHERE is_active=1"
    sql += " ORDER BY updated_at DESC, id DESC"
    with connect() as conn:
        return [dict(r) for r in conn.execute(sql).fetchall()]


def get_document(document_id):
    with connect() as conn:
        row = conn.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
        return dict(row) if row else None


def list_document_versions(doc_uid):
    with connect() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM documents WHERE doc_uid=? ORDER BY version DESC", (doc_uid,)
        ).fetchall()]


def get_document_chunks(document_id, with_vectors=False):
    cols = "id, chunk_index, content, page_start, page_end, char_count"
    if with_vectors:
        cols += ", embedding"
    with connect() as conn:
        rows = conn.execute(
            f"SELECT {cols} FROM document_chunks WHERE document_id=? ORDER BY chunk_index",
            (document_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def update_document_meta(document_id, title=None, category=None, note=None):
    with connect() as conn:
        cur = conn.execute(
            """UPDATE documents SET title=COALESCE(?, title), category=COALESCE(?, category),
                   note=COALESCE(?, note), updated_at=? WHERE id=?""",
            (title, category, note, now(), document_id),
        )
        return cur.rowcount > 0


def set_active_version(document_id):
    """특정 버전을 검색 대상(활성)으로 지정하고 같은 문서의 다른 버전은 비활성화합니다."""
    with connect() as conn:
        row = conn.execute("SELECT doc_uid FROM documents WHERE id=?", (document_id,)).fetchone()
        if not row:
            return False
        ts = now()
        conn.execute("UPDATE documents SET is_active=0, updated_at=? WHERE doc_uid=?",
                     (ts, row["doc_uid"]))
        conn.execute("UPDATE documents SET is_active=1, updated_at=? WHERE id=?", (ts, document_id))
        return True


def delete_document_version(document_id):
    """문서의 한 버전을 삭제합니다. 활성 버전이었다면 남은 최신 버전을 활성화합니다.

    반환: 삭제된 문서 dict (없으면 None)
    """
    with connect() as conn:
        row = conn.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
        if not row:
            return None
        conn.execute("DELETE FROM documents WHERE id=?", (document_id,))
        if row["is_active"]:
            latest = conn.execute(
                "SELECT id FROM documents WHERE doc_uid=? ORDER BY version DESC LIMIT 1",
                (row["doc_uid"],),
            ).fetchone()
            if latest:
                conn.execute("UPDATE documents SET is_active=1, updated_at=? WHERE id=?",
                             (now(), latest["id"]))
        return dict(row)


def delete_document_all_versions(doc_uid):
    """문서 전체(모든 버전과 청크)를 삭제합니다. 반환: 삭제된 문서 행 목록."""
    with connect() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM documents WHERE doc_uid=?", (doc_uid,)
        ).fetchall()]
        conn.execute("DELETE FROM documents WHERE doc_uid=?", (doc_uid,))
        return rows


def active_chunks():
    """검색 대상 청크 (활성 문서만)."""
    with connect() as conn:
        rows = conn.execute(
            """SELECT c.id AS chunk_id, c.document_id, c.chunk_index, c.content,
                      c.page_start, c.page_end, c.embedding,
                      d.title, d.file_type, d.version, d.category
               FROM document_chunks c JOIN documents d ON d.id = c.document_id
               WHERE d.is_active = 1"""
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# 지식 (knowledge_items)
# ---------------------------------------------------------------------------
def add_knowledge_item(kind, title, content, vector, tags=None, source="user",
                       source_message_id=None):
    ts = now()
    with connect() as conn:
        cur = conn.execute(
            """INSERT INTO knowledge_items (kind, title, content, tags, embedding, source,
                   source_message_id, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (kind, title, content, tags, pack_vector(vector), source, source_message_id, ts, ts),
        )
        return cur.lastrowid


def list_knowledge_items(kind=None):
    sql = "SELECT id, kind, title, content, tags, source, created_at, updated_at FROM knowledge_items"
    params = ()
    if kind:
        sql += " WHERE kind=?"
        params = (kind,)
    sql += " ORDER BY id"
    with connect() as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def get_knowledge_item(item_id):
    with connect() as conn:
        row = conn.execute(
            "SELECT id, kind, title, content, tags, source, created_at, updated_at "
            "FROM knowledge_items WHERE id=?", (item_id,)
        ).fetchone()
        return dict(row) if row else None


def update_knowledge_item(item_id, title, content, vector, kind=None, tags=None):
    with connect() as conn:
        cur = conn.execute(
            """UPDATE knowledge_items SET title=?, content=?, embedding=?,
                   kind=COALESCE(?, kind), tags=COALESCE(?, tags), updated_at=?
               WHERE id=?""",
            (title, content, pack_vector(vector), kind, tags, now(), item_id),
        )
        return cur.rowcount > 0


def delete_knowledge_item(item_id):
    with connect() as conn:
        return conn.execute("DELETE FROM knowledge_items WHERE id=?", (item_id,)).rowcount > 0


def knowledge_items_with_vectors():
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, kind, title, content, embedding FROM knowledge_items"
        ).fetchall()
    return [dict(r) for r in rows]


# 이전 코드 호환용 이름 (knowledge 테이블 → knowledge_items)
def add_knowledge(kind, title, content, vector):
    return add_knowledge_item(kind, title, content, vector)


def list_knowledge():
    return [(i["id"], i["kind"], i["title"]) for i in list_knowledge_items()]


def update_knowledge(item_id, title, content, vector):
    return update_knowledge_item(item_id, title, content, vector)


def delete_knowledge(item_id):
    return delete_knowledge_item(item_id)

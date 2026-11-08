"""문서 등록 / 중복 방지 / 버전 관리 / 재임베딩 / 삭제.

등록 규칙
  1. 파일 내용(SHA-256)이 이미 등록된 문서와 같으면 → 중복으로 거부
  2. 같은 파일명의 문서가 있으면 → 그 문서의 새 버전으로 등록 (이전 버전은 비활성 보관)
  3. 그 외 → 새 문서로 등록
원본 파일 사본은 storage/documents/ 에 보관되어 재임베딩에 사용됩니다.
"""
import shutil
import tempfile
import uuid
from pathlib import Path

import config
import knowledge_db as db
from document_loader import (
    DocumentValidationError,
    chunk_segments,
    file_sha256,
    read_segments,
    validate_upload,
)


def _default_embedder(texts):
    from rag import embed
    return embed(texts)


def _store_dir():
    path = Path(config.DOCUMENT_STORE_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def stored_file_path(document):
    """DB에 기록된 상대 경로를 저장소 내부 경로로만 해석합니다 (경로 이탈 차단)."""
    if not document or not document.get("stored_path"):
        return None
    store = _store_dir()
    path = (store / document["stored_path"]).resolve()
    if store not in path.parents:
        raise DocumentValidationError("허용되지 않은 문서 저장 경로입니다.")
    return path if path.is_file() else None


def _build_chunks(path, title, file_type, embedder):
    unit = db.LOCATION_UNIT.get(file_type)
    chunks = chunk_segments(read_segments(path), unit=unit)
    if not chunks:
        raise DocumentValidationError(
            "문서에서 텍스트를 추출하지 못했습니다. (스캔 이미지 PDF는 지원하지 않습니다)"
        )
    vectors = embedder([f"{title}\n{c['content']}" for c in chunks])
    if len(vectors) != len(chunks):
        raise RuntimeError("임베딩 개수가 청크 개수와 다릅니다.")
    for chunk, vector in zip(chunks, vectors):
        chunk["vector"] = vector
    return chunks


def register_file(file_path, title=None, category=None, note=None, source="cli",
                  original_name=None, replace_document_id=None, embedder=None):
    """문서 파일을 등록합니다.

    replace_document_id: 지정 시 해당 문서의 새 버전으로 등록 (파일명이 달라도)
    반환: {"status": "created" | "new_version" | "duplicate", "document": dict, "message": str}
    """
    embedder = embedder or _default_embedder
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"파일이 없습니다: {path.name}")

    file_name, suffix = validate_upload(original_name or path.name, path.stat().st_size)
    file_type = suffix.lstrip(".")
    content_hash = file_sha256(path)

    existing = db.find_document_by_hash(content_hash)
    if existing:
        state = "사용 중" if existing["is_active"] else "비활성 이전 버전"
        return {
            "status": "duplicate",
            "document": existing,
            "message": f"이미 등록된 문서입니다: [{existing['id']}] {existing['title']} "
                       f"v{existing['version']} ({state})",
        }

    if replace_document_id is not None:
        base = db.get_document(replace_document_id)
        if not base:
            raise ValueError(f"문서 ID {replace_document_id}를 찾을 수 없습니다.")
    else:
        base = db.find_latest_by_file_name(file_name)

    if base:
        doc_uid = base["doc_uid"]
        version = db.next_version(doc_uid)
        title = title or base["title"]
        category = category or base.get("category")
        status = "new_version"
    else:
        doc_uid = uuid.uuid4().hex
        version = 1
        title = title or file_name
        status = "created"

    chunks = _build_chunks(path, title, file_type, embedder)

    stored_name = f"{doc_uid}_v{version}{suffix}"
    stored = _store_dir() / stored_name
    shutil.copy2(path, stored)
    try:
        doc_id = db.create_document(
            {
                "doc_uid": doc_uid, "version": version, "title": title,
                "file_name": file_name, "file_type": file_type,
                "file_size": path.stat().st_size, "content_hash": content_hash,
                "stored_path": stored_name, "category": category, "note": note,
                "source": source, "embed_model": config.EMBED_MODEL,
            },
            chunks,
        )
    except Exception:
        stored.unlink(missing_ok=True)
        raise

    document = db.get_document(doc_id)
    message = (
        f"새 문서 등록 완료: [{doc_id}] {title} (청크 {len(chunks)}개)"
        if status == "created"
        else f"새 버전 등록 완료: [{doc_id}] {title} v{version} (청크 {len(chunks)}개, 이전 버전은 비활성 보관)"
    )
    return {"status": status, "document": document, "message": message}


def register_upload(file_name, data, title=None, category=None, note=None,
                    replace_document_id=None, embedder=None):
    """웹 업로드(bytes)를 검증 후 등록합니다."""
    safe_name, suffix = validate_upload(file_name, len(data))
    tmp_dir = Path(tempfile.mkdtemp(prefix="mobility_upload_"))
    try:
        tmp_path = tmp_dir / f"upload{suffix}"
        tmp_path.write_bytes(data)
        return register_file(
            tmp_path, title=title, category=category, note=note, source="upload",
            original_name=safe_name, replace_document_id=replace_document_id,
            embedder=embedder,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def reindex_document(document_id, embedder=None):
    """문서 버전을 다시 청크 분할·임베딩합니다.

    - 원본 사본이 있으면: 원본에서 다시 추출 (페이지/슬라이드 정보 갱신)
    - 원본이 없으면(이전 데이터): 저장된 청크 텍스트만 다시 임베딩
    """
    embedder = embedder or _default_embedder
    document = db.get_document(document_id)
    if not document:
        raise ValueError(f"문서 ID {document_id}를 찾을 수 없습니다.")

    source_path = stored_file_path(document)
    if source_path:
        chunks = _build_chunks(source_path, document["title"], document["file_type"], embedder)
        status = "ready"
        mode = "원본 파일에서 재추출"
    else:
        old = db.get_document_chunks(document_id)
        if not old:
            raise DocumentValidationError("재임베딩할 내용이 없습니다.")
        vectors = embedder([f"{document['title']}\n{c['content']}" for c in old])
        chunks = [
            {"content": c["content"], "page_start": c["page_start"],
             "page_end": c["page_end"], "vector": v}
            for c, v in zip(old, vectors)
        ]
        status = document["status"]
        mode = "저장된 텍스트 재임베딩 (원본 파일 없음)"

    db.replace_chunks(document_id, chunks, embed_model=config.EMBED_MODEL, status=status)
    return {"document": db.get_document(document_id), "chunks": len(chunks), "mode": mode}


def _remove_stored_files(rows):
    for row in rows:
        try:
            path = stored_file_path(row)
        except DocumentValidationError:
            continue
        if path:
            path.unlink(missing_ok=True)


def delete_document(document_id, all_versions=False):
    """문서 버전(또는 전체 버전)을 삭제합니다. 반환: 삭제된 버전 수."""
    document = db.get_document(document_id)
    if not document:
        return 0
    if all_versions:
        rows = db.delete_document_all_versions(document["doc_uid"])
    else:
        row = db.delete_document_version(document_id)
        rows = [row] if row else []
    _remove_stored_files(rows)
    return len(rows)


def document_summary(document):
    """CLI/웹 표시용 한 줄 요약."""
    active = "사용중" if document["is_active"] else "비활성"
    return (f"[{document['id']}] {document['title']} v{document['version']} "
            f"({document['file_type']}, 청크 {document['chunk_count']}개, {active}, {document['status']})")

"""문서 텍스트 추출, 업로드 검증, 위치(페이지/슬라이드) 정보를 보존한 청크 분할."""
import hashlib
import re
from pathlib import Path

from pypdf import PdfReader
from docx import Document
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

import config


class DocumentValidationError(ValueError):
    """업로드 문서가 형식/크기 규칙을 위반한 경우."""


# ---------------------------------------------------------------------------
# 검증
# ---------------------------------------------------------------------------
_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_file_name(name):
    """경로 성분을 제거하고 Windows에서 안전한 파일명으로 정리합니다."""
    name = Path(str(name).replace("\\", "/")).name  # 디렉터리 경로 제거 (../ 차단)
    name = _UNSAFE_CHARS.sub("_", name).strip(" .")
    if not name:
        raise DocumentValidationError("파일명이 올바르지 않습니다.")
    return name[:150]


def validate_upload(file_name, size):
    """확장자와 크기를 검사하고 (안전한 파일명, 확장자)를 반환합니다."""
    name = safe_file_name(file_name)
    suffix = Path(name).suffix.lower()
    if suffix not in config.ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(config.ALLOWED_EXTENSIONS))
        raise DocumentValidationError(f"지원하지 않는 파일 형식입니다: {suffix or '(없음)'} (허용: {allowed})")
    if size <= 0:
        raise DocumentValidationError("빈 파일은 등록할 수 없습니다.")
    if size > config.MAX_UPLOAD_BYTES:
        raise DocumentValidationError(
            f"파일이 너무 큽니다: {size / 1024 / 1024:.1f}MB "
            f"(최대 {config.MAX_UPLOAD_BYTES // 1024 // 1024}MB)"
        )
    return name, suffix


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# 추출
# ---------------------------------------------------------------------------
def extract_pptx_shape(shape):
    """PPTX의 텍스트, 표, 그룹 도형에서 내용을 추출합니다."""
    texts = []

    if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
        for child in shape.shapes:
            texts.extend(extract_pptx_shape(child))
        return texts

    if shape.has_text_frame:
        text = shape.text.strip()
        if text:
            texts.append(text)

    if getattr(shape, "has_table", False) and shape.has_table:
        for row in shape.table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):  # 빈 행( | | | ) 제외
                texts.append(" | ".join(cells))

    return texts


def _clean(text):
    text = (text or "").replace("\x0b", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_segments(file_path):
    """문서를 위치 단위 조각 목록으로 추출합니다.

    반환: [{"page": int|None, "text": str}]
      - PDF: 페이지 번호, PPTX: 슬라이드 번호, DOCX/TXT/MD: None
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"파일이 없습니다: {path.name}")

    suffix = path.suffix.lower()

    if suffix in (".txt", ".md"):
        raw = path.read_bytes()
        for encoding in ("utf-8-sig", "cp949"):
            try:
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise DocumentValidationError("텍스트 인코딩을 인식할 수 없습니다 (UTF-8 또는 CP949 필요).")
        return [{"page": None, "text": _clean(text)}]

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return [
            {"page": number, "text": _clean(page.extract_text())}
            for number, page in enumerate(reader.pages, start=1)
        ]

    if suffix == ".docx":
        doc = Document(str(path))
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells]
                if any(cells):
                    parts.append(" | ".join(cells))
        return [{"page": None, "text": _clean("\n".join(parts))}]

    if suffix == ".pptx":
        presentation = Presentation(str(path))
        segments = []
        for number, slide in enumerate(presentation.slides, start=1):
            texts = []
            for shape in slide.shapes:
                texts.extend(extract_pptx_shape(shape))
            if slide.has_notes_slide:
                notes = slide.notes_slide.notes_text_frame.text.strip()
                if notes:
                    texts.append(f"[발표자 노트] {notes}")
            segments.append({"page": number, "text": _clean("\n".join(texts))})
        return segments

    raise DocumentValidationError(f"지원하지 않는 파일 형식: {suffix}")


def read_document(file_path):
    """(호환용) 문서 전체 텍스트를 하나의 문자열로 반환합니다."""
    path = Path(file_path)
    segments = read_segments(path)
    if path.suffix.lower() == ".pptx":
        return "\n\n".join(
            f"[슬라이드 {s['page']}]\n{s['text']}" for s in segments if s["text"]
        )
    if path.suffix.lower() == ".pdf":
        return "\n".join(s["text"] for s in segments)
    return "\n".join(s["text"] for s in segments)


# ---------------------------------------------------------------------------
# 청크 분할
# ---------------------------------------------------------------------------
def _split_long_text(text, size, overlap):
    """긴 텍스트를 문단/줄 경계를 우선해 size 이하 조각으로 나눕니다."""
    if len(text) <= size:
        return [text]
    overlap = min(overlap, size // 4)  # overlap이 크면 같은 조각이 반복 생성됨
    pieces = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            cut = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind(". "))
            if cut > size // 2:
                end = start + cut + 1
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [p for p in pieces if p]


def chunk_segments(segments, size=None, overlap=None, unit=None):
    """위치 조각을 검색용 청크로 변환합니다.

    - 짧은 페이지/슬라이드는 이웃과 합쳐 size 이내로 묶음 (page_start~page_end 기록)
    - 긴 페이지는 size 단위로 분할 (같은 페이지 번호 유지)
    - unit(예: '슬라이드')을 주면 각 조각 앞에 '[슬라이드 N]' 표식을 붙임
    반환: [{"content", "page_start", "page_end"}]
    """
    size = size or config.CHUNK_SIZE
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap
    chunks = []
    buffer, buf_start, buf_end = [], None, None

    def flush():
        nonlocal buffer, buf_start, buf_end
        if buffer:
            chunks.append({"content": "\n\n".join(buffer), "page_start": buf_start, "page_end": buf_end})
        buffer, buf_start, buf_end = [], None, None

    for segment in segments:
        text = segment["text"].strip()
        if not text:
            continue
        page = segment["page"]
        if unit and page is not None:
            text = f"[{unit} {page}]\n{text}"
        if len(text) > size:
            flush()
            for piece in _split_long_text(text, size, overlap):
                chunks.append({"content": piece, "page_start": page, "page_end": page})
            continue
        current_len = sum(len(b) + 2 for b in buffer)
        if buffer and current_len + len(text) > size:
            flush()
        if not buffer:
            buf_start = page
        buffer.append(text)
        buf_end = page
    flush()
    return chunks

"""Mobility Dev AI 공통 설정.

환경 변수로 일부 값을 덮어쓸 수 있습니다 (테스트용 DB 경로 등).
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# --- 데이터 저장 위치 (모두 로컬) ---
DB_PATH = Path(os.environ.get("MOBILITY_AI_DB", BASE_DIR / "ai_data.db"))
BACKUP_DIR = BASE_DIR / "backups"
# 업로드/등록된 원본 문서 사본 저장 위치 (재임베딩 시 사용)
DOCUMENT_STORE_DIR = Path(
    os.environ.get("MOBILITY_AI_DOC_STORE", BASE_DIR / "storage" / "documents")
)

# --- Ollama (로컬 전용) ---
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
CHAT_MODEL = os.environ.get("MOBILITY_AI_CHAT_MODEL", "qwen3:4b")
EMBED_MODEL = os.environ.get("MOBILITY_AI_EMBED_MODEL", "bge-m3")
OLLAMA_TIMEOUT = 300
EMBED_BATCH_SIZE = 16

# --- 검색 ---
# 벡터 검색과 BM25 키워드 검색을 RRF로 결합합니다. "0"이면 벡터 검색만 사용합니다.
HYBRID_SEARCH = os.environ.get("MOBILITY_AI_HYBRID", "1") != "0"
RRF_K = 60
# 대화가 있는 경우 로컬 모델로 독립적인 검색 질문을 만듭니다.
REWRITE_QUERY = os.environ.get("MOBILITY_AI_REWRITE_QUERY", "1") != "0"
# 답변에 사용할 최소 코사인 유사도. 초기값이며 실제 평가 데이터로 보정하세요.
MIN_EVIDENCE_SCORE = float(os.environ.get("MOBILITY_AI_MIN_EVIDENCE_SCORE", "0.45"))
if not 0 <= MIN_EVIDENCE_SCORE <= 1:
    raise ValueError("MOBILITY_AI_MIN_EVIDENCE_SCORE는 0~1 사이여야 합니다.")

# --- 문서 처리 ---
ALLOWED_EXTENSIONS = {".pdf", ".pptx", ".docx", ".txt", ".md"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50MB
CHUNK_SIZE = 800  # 문자 수 기준
CHUNK_OVERLAP = 120

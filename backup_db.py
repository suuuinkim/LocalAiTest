"""SQLite DB 안전 백업.

SQLite online backup API를 사용하므로 앱이 실행 중이어도 일관된 사본을 만듭니다.
원본 DB는 읽기만 하며 절대 수정하지 않습니다.
"""
import sqlite3
from datetime import datetime
from pathlib import Path

import config


def backup_database(db_path=None, label="manual", backup_dir=None):
    """DB를 백업하고 백업 파일 경로를 반환합니다. DB가 없으면 None."""
    db_path = Path(db_path or config.DB_PATH)
    backup_dir = Path(backup_dir or config.BACKUP_DIR)

    if not db_path.exists():
        return None

    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup_path = backup_dir / f"{db_path.stem}_{timestamp}_{label}.db"

    source = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    target = sqlite3.connect(backup_path)
    try:
        source.backup(target)
        result = target.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        target.close()
        source.close()

    if result != "ok":
        raise RuntimeError(f"백업 파일 무결성 검사 실패: {backup_path}")

    return backup_path


if __name__ == "__main__":
    path = backup_database()
    if path is None:
        print("백업할 DB 파일이 없습니다.")
    else:
        print("DB 백업 완료!")
        print("저장 위치:", path)

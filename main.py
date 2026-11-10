"""Mobility Dev AI - 터미널 채팅."""
from pathlib import Path

import knowledge_db as db
import document_service as docs
from rag import add_text, ask_ai, embed

HELP = """사용 가능한 명령어:
  [지식]  /add 제목 | 내용        /memory 기억할 내용
          /list                   /edit ID | 새 제목 | 새 내용     /delete ID
  [문서]  /file 파일경로          /docs (활성 문서)   /docs all (모든 버전)
          /doc ID (상세)          /reindex ID         /deldoc ID   /deldoc ID all
          /activate ID (해당 버전을 사용 버전으로 지정)
  [대화]  /new                    /sessions           /use 세션ID
  [기타]  /stats  /help  /exit"""


def confirm(prompt):
    return input(f"{prompt} (y/N): ").strip().lower() == "y"


def handle_command(command, state, base_path):
    if command.startswith("/add "):
        title, content = command[5:].split("|", 1)
        print("지식 저장 완료! ID:", add_text("note", title.strip(), content.strip()))

    elif command.startswith("/memory "):
        print("장기 기억 저장 완료! ID:", add_text("memory", "사용자가 알려준 정보", command[8:].strip()))

    elif command == "/list":
        items = db.list_knowledge_items()
        if not items:
            print("등록된 지식이 없습니다.")
        for item in items:
            print(f"{item['id']} | {item['kind']} | {item['title']} | {item['content'][:40]}")

    elif command.startswith("/edit "):
        id_text, title, content = command[6:].split("|", 2)
        title, content = title.strip(), content.strip()
        vector = embed([f"{title}\n{content}"])[0]
        ok = db.update_knowledge_item(int(id_text.strip()), title, content, vector)
        print("수정 완료" if ok else "해당 ID가 없습니다.")

    elif command.startswith("/delete "):
        item_id = int(command[8:].strip())
        item = db.get_knowledge_item(item_id)
        if not item:
            print("해당 ID가 없습니다.")
        elif confirm(f"지식 [{item_id}] '{item['title']}'을(를) 삭제할까요?"):
            db.delete_knowledge_item(item_id)
            print("삭제 완료")

    elif command.startswith("/file "):
        file_path = Path(command[6:].strip().strip('"')).expanduser()
        if not file_path.is_absolute():
            file_path = base_path / file_path
        print("문서를 분석하고 임베딩하는 중입니다... (문서 크기에 따라 수 분 걸릴 수 있습니다)")
        result = docs.register_file(file_path, source="cli")
        print(result["message"])

    elif command in ("/docs", "/docs all"):
        documents = db.list_documents(include_inactive=command.endswith("all"))
        if not documents:
            print("등록된 문서가 없습니다.")
        for d in documents:
            print(docs.document_summary(d))

    elif command.startswith("/doc "):
        d = db.get_document(int(command[5:].strip()))
        if not d:
            print("해당 문서가 없습니다.")
            return
        print(docs.document_summary(d))
        print(f"  파일명: {d['file_name']} / 크기: {d['file_size'] or '-'} / 등록: {d['created_at']}")
        if d["note"]:
            print(f"  메모: {d['note']}")
        for v in db.list_document_versions(d["doc_uid"]):
            print("  버전:", docs.document_summary(v))
        for c in db.get_document_chunks(d["id"])[:5]:
            loc = db.format_location(d["file_type"], c["page_start"], c["page_end"]) or "-"
            print(f"  #{c['chunk_index']} [{loc}] {c['content'][:60]!r}")

    elif command.startswith("/reindex "):
        print("재임베딩 중입니다...")
        result = docs.reindex_document(int(command[9:].strip()))
        print(f"재임베딩 완료: 청크 {result['chunks']}개 ({result['mode']})")

    elif command.startswith("/deldoc "):
        args = command[8:].split()
        d = db.get_document(int(args[0]))
        all_versions = len(args) > 1 and args[1] == "all"
        if not d:
            print("해당 문서가 없습니다.")
        elif confirm(f"{docs.document_summary(d)} {'의 모든 버전' if all_versions else ''}을(를) 삭제할까요?"):
            print(f"삭제 완료: {docs.delete_document(d['id'], all_versions=all_versions)}개 버전")

    elif command.startswith("/activate "):
        ok = db.set_active_version(int(command[10:].strip()))
        print("사용 버전 변경 완료" if ok else "해당 문서가 없습니다.")

    elif command == "/new":
        state["session_id"] = db.create_session("터미널 대화")
        print("새 대화를 시작합니다:", state["session_id"])

    elif command == "/sessions":
        for s in db.list_sessions():
            mark = "*" if s["id"] == state["session_id"] else " "
            print(f"{mark} {s['id']} | {s['title']} | 메시지 {s['message_count']}개 | {s['updated_at']}")

    elif command.startswith("/use "):
        sid = command[5:].strip()
        if db.get_session(sid):
            state["session_id"] = sid
            print("대화 전환 완료")
        else:
            print("해당 대화가 없습니다.")

    elif command == "/stats":
        for k, v in db.db_stats().items():
            print(f"  {k}: {v}")

    elif command == "/help":
        print(HELP)

    else:
        print("알 수 없는 명령어입니다. /help 를 입력해 보세요.")


def main():
    for message in db.init_db():
        print("DB 마이그레이션:", message)
    base_path = Path(__file__).resolve().parent
    state = {"session_id": "clean-chat-v1"}
    db.ensure_session(state["session_id"], title="터미널 대화")

    print("Mobility Dev AI (터미널) 실행 완료!\n")
    print(HELP)

    while True:
        try:
            command = input("\n나: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n프로그램을 종료합니다.")
            break

        if command == "/exit":
            print("프로그램을 종료합니다.")
            break
        if not command:
            continue

        try:
            if command.startswith("/"):
                handle_command(command, state, base_path)
            else:
                answer, sources = ask_ai(command, session_id=state["session_id"])
                print("\nAI:", answer)
                if sources:
                    print("\n참고 자료:")
                    for s in sources:
                        print("  -", s)
        except Exception as error:
            print("오류가 발생했습니다:", error)


if __name__ == "__main__":
    main()

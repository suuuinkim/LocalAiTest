"""로컬 문서 채팅 UI. 실행: python -m streamlit run web_app.py"""
import streamlit as st

import config
import knowledge_db as db
from document_service import register_upload
from rag import ask_ai


def render_message(message):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        sources = message.get("sources", [])
        if sources:
            with st.expander(f"검색 근거 원문 · {len(sources)}개"):
                st.caption("답변 생성에 제공한 자료입니다. 유사도는 답변의 정확도 확률이 아닙니다.")
                for source in sources:
                    st.markdown(f"**{source['label']}**")
                    st.caption(f"코사인 유사도 {source['score']:.3f}")
                    if "content" in source:
                        st.text(source["content"])
                    else:
                        st.caption("이전 대화에는 근거 원문이 저장되어 있지 않습니다.")


def sidebar():
    with st.sidebar:
        st.header("내 문서")
        with st.form("upload", clear_on_submit=True):
            uploaded = st.file_uploader(
                "문서 선택 · 최대 50MB",
                type=sorted(ext.lstrip(".") for ext in config.ALLOWED_EXTENSIONS),
            )
            submitted = st.form_submit_button("문서 등록", use_container_width=True)
        if submitted:
            if uploaded is None:
                st.warning("등록할 문서를 선택해 주세요.")
            elif uploaded.size > config.MAX_UPLOAD_BYTES:
                st.error("50MB 이하의 문서를 선택해 주세요.")
            else:
                try:
                    with st.spinner("문서 분석 및 임베딩 중…"):
                        result = register_upload(uploaded.name, uploaded.getvalue())
                    st.success(result["message"])
                except Exception as error:
                    st.error(f"문서 등록 실패: {error}")

        documents = db.list_documents()
        st.caption(f"사용 중인 문서 {len(documents)}개")
        for document in documents:
            st.text(f"{document['title']} · v{document['version']}")
        if not documents:
            st.info("문서를 등록하고 질문해 보세요.")

        st.divider()
        st.header("대화")
        if st.button("새 대화", use_container_width=True):
            st.session_state.session_id = db.create_session("웹 대화")
        sessions = db.list_sessions()
        ids = [session["id"] for session in sessions]
        if st.session_state.get("session_id") not in ids:
            st.session_state.session_id = ids[0] if ids else db.create_session("웹 대화")
            sessions = db.list_sessions()
            ids = [session["id"] for session in sessions]
        labels = {s["id"]: f"{s['title']} · {s['id'][:8]}" for s in sessions}
        st.selectbox("대화 선택", ids, format_func=labels.get, key="session_id")
        st.caption("문서와 대화는 이 PC의 로컬 저장소에 보관됩니다.")


def main():
    st.set_page_config(page_title="Mobility Dev AI", page_icon="📚", layout="wide")
    st.title("Mobility Dev AI")
    st.caption("내 문서에 질문하고, 답변의 검색 근거를 확인하세요.")
    try:
        db.init_db()
        sidebar()
        session_id = st.session_state.session_id
        messages = db.get_messages(session_id, limit=None, with_meta=True)
    except Exception as error:
        st.error(f"저장소를 열지 못했습니다: {error}")
        st.stop()

    for message in messages:
        render_message(message)
    question = st.chat_input("문서에 대해 질문해 주세요")
    if question and question.strip():
        try:
            with st.spinner("관련 자료 검색 및 답변 생성 중…"):
                ask_ai(question, session_id=session_id)
            if not messages:
                db.rename_session(session_id, question.strip()[:40])
        except Exception as error:
            st.error(f"답변 생성 실패: {error}")
        else:
            st.rerun()


if __name__ == "__main__":
    main()

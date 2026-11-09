"""임베딩, 검색, 답변 생성 (Ollama 로컬 전용).

Phase 1: 새 스키마(document_chunks, knowledge_items) 기반 검색 + 출처(페이지/슬라이드) 표시.
Phase 2에서 검색 정확도/응답 정책을 고도화할 예정입니다.
"""
import math
import re

import requests

import config
from knowledge_db import (
    active_chunks,
    add_knowledge_item,
    format_location,
    get_messages,
    knowledge_items_with_vectors,
    save_message,
    unpack_vector,
)

# 이전 코드 호환용 상수
BASE_URL = config.OLLAMA_BASE_URL
CHAT_MODEL = config.CHAT_MODEL
EMBED_MODEL = config.EMBED_MODEL


class OllamaError(RuntimeError):
    pass


def call_ollama(endpoint, payload):
    try:
        response = requests.post(
            config.OLLAMA_BASE_URL + endpoint, json=payload, timeout=config.OLLAMA_TIMEOUT
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError as error:
        raise OllamaError(
            "Ollama 서버에 연결할 수 없습니다. 'ollama serve'가 실행 중인지 확인해 주세요."
        ) from error
    except requests.exceptions.RequestException as error:
        raise OllamaError(f"Ollama 요청 실패: {error}") from error
    return response.json()


def embed(texts):
    """텍스트 목록을 임베딩합니다 (배치 단위 호출)."""
    vectors = []
    for i in range(0, len(texts), config.EMBED_BATCH_SIZE):
        batch = texts[i:i + config.EMBED_BATCH_SIZE]
        result = call_ollama("/api/embed", {"model": config.EMBED_MODEL, "input": batch})
        vectors.extend(result["embeddings"])
    return vectors


def similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b + 1e-10)


# ---------------------------------------------------------------------------
# 지식 등록 (호환)
# ---------------------------------------------------------------------------
def add_text(kind, title, content):
    """(호환용) 짧은 지식/메모를 knowledge_items에 저장합니다. 반환: [id]

    문서 파일 등록은 document_service.register_file()을 사용하세요.
    """
    content = content.strip()
    if not content:
        raise ValueError("빈 내용은 저장할 수 없습니다")
    vector = embed([f"{title}\n{content}"])[0]
    return [add_knowledge_item(kind, title.strip(), content, vector)]


# ---------------------------------------------------------------------------
# 검색
# ---------------------------------------------------------------------------
def search_knowledge(question, limit=5, min_score=0.0):
    """문서 청크 + 지식 항목을 함께 검색합니다.

    반환: [{"score", "type"(document/knowledge), "title", "content", "location",
            "document_id", "chunk_id", "item_id"}] (점수 내림차순)
    """
    question_vector = embed([question])[0]
    results = []

    for chunk in active_chunks():
        score = similarity(question_vector, unpack_vector(chunk["embedding"]))
        results.append({
            "score": score,
            "type": "document",
            "title": chunk["title"],
            "content": chunk["content"],
            "location": format_location(chunk["file_type"], chunk["page_start"], chunk["page_end"]),
            "document_id": chunk["document_id"],
            "chunk_id": chunk["chunk_id"],
            "item_id": None,
        })

    for item in knowledge_items_with_vectors():
        score = similarity(question_vector, unpack_vector(item["embedding"]))
        results.append({
            "score": score,
            "type": "knowledge",
            "title": item["title"],
            "content": item["content"],
            "location": f"지식 #{item['id']} ({item['kind']})",
            "document_id": None,
            "chunk_id": None,
            "item_id": item["id"],
        })

    results = [r for r in results if r["score"] >= min_score]
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:limit]


def source_label(match):
    return f"{match['title']} ({match['location']})" if match["location"] else match["title"]


# ---------------------------------------------------------------------------
# 답변
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """당신은 사용자의 개인용 교통시스템 개발 전문 AI(Mobility Dev AI)입니다.

규칙:
1. [검색된 자료]를 우선하여 답변하세요.
2. 자료에서 확인되는 내용은 출처(문서명, 페이지/슬라이드)를 함께 적으세요.
3. 자료에 없는 일반적인 설명을 덧붙일 때는 '[일반 설명]'이라고 구분해 표시하세요.
4. 자료로 확인할 수 없는 사실은 지어내지 말고 '제공된 자료에서 확인되지 않습니다'라고 답하세요.
5. 항상 한국어 존댓말로 답변하세요.
6. 이전 대화와 검색된 자료가 충돌하면 검색된 자료를 우선하세요.

[검색된 자료]
{context}
"""


def clean_answer(answer):
    if "</think>" in answer:
        answer = answer.split("</think>", 1)[1]
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
    if "<think>" in answer or not answer:
        return "AI가 최종 답변을 생성하지 못했습니다. 다시 질문해 주세요."
    return answer


def ask_ai(question, session_id="default", limit=5):
    """질문에 답하고 대화를 저장합니다. 반환: (answer, sources 라벨 목록)"""
    matches = search_knowledge(question, limit=limit)

    context = "\n\n".join(
        f"[자료 {i}] {source_label(m)}\n{m['content']}" for i, m in enumerate(matches, start=1)
    ) or "관련 자료 없음"

    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
    messages += get_messages(session_id)
    messages.append({"role": "user", "content": f"/no_think\n{question}"})

    result = call_ollama(
        "/api/chat",
        {"model": config.CHAT_MODEL, "messages": messages, "stream": False, "think": False},
    )
    answer = clean_answer(result["message"].get("content", ""))

    sources = [
        {"label": source_label(m), "score": round(m["score"], 4), "type": m["type"],
         "document_id": m["document_id"], "chunk_id": m["chunk_id"], "item_id": m["item_id"]}
        for m in matches
    ]
    save_message(session_id, "user", question)
    save_message(session_id, "assistant", answer, sources=sources)

    return answer, [s["label"] for s in sources]

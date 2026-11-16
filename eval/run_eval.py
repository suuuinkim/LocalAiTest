"""RAG 검색 정확도 평가 (Phase 2).

실행 예:
  .venv\\Scripts\\python.exe eval\\run_eval.py                 # ai_data.db 사본으로 평가
  .venv\\Scripts\\python.exe eval\\run_eval.py --db 다른.db     # 다른 DB(예: v2 재등록 사본)와 비교
  .venv\\Scripts\\python.exe eval\\run_eval.py --label 점수하한   # 결과 파일 이름표

원본 DB는 절대 수정하지 않습니다: 지정한 DB를 임시 폴더로 복사한 뒤 그 사본을 검색합니다.
측정 항목 (gold_slides가 있는 질문만):
  - Hit@1/3/5: 정답 슬라이드를 포함한 청크가 상위 k개 안에 있는 비율
  - MRR: 첫 정답 순위의 역수 평균
문서별 정답과 근거 없는 질문을 구분하여 근거 선택 정책의 통과율/보류율도 측정합니다.
생성 답변 자체의 사실성은 측정하지 않습니다.
"""
import argparse
import json
import os
import sqlite3
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = Path(__file__).resolve().parent


def is_hit(match, gold, document=None):
    if document and document not in (match.get("file_name"), match.get("title")):
        return False
    if match["type"] != "document" or match.get("page_start") is None:
        return False
    start = match["page_start"]
    end = match["page_end"] if match.get("page_end") is not None else start
    return any(start <= g <= end for g in gold)


def summarize(rows, k):
    """검색 및 근거 선택 정책을 평가합니다. 생성 답변의 사실성 평가는 아닙니다."""
    positives = [r for r in rows if r["gold_slides"]]
    negatives = [r for r in rows if not r["gold_slides"]]

    def rate(count, total):
        return round(count / total, 3) if total else None

    scores = [r["top_score"] for r in positives if r["top_score"] is not None]
    return {
        "questions": len(positives), "negative_questions": len(negatives),
        **{f"hit@{cutoff}": rate(sum(bool(r["rank"] and r["rank"] <= cutoff)
                                    for r in positives), len(positives))
           for cutoff in sorted({1, 3, k}) if cutoff <= k},
        "mrr": rate(sum(1 / r["rank"] for r in positives if r["rank"]), len(positives)),
        "positive_refusal_rate": rate(sum(r["refused"] for r in positives), len(positives)),
        "negative_refusal_rate": rate(sum(r["refused"] for r in negatives), len(negatives)),
        "negative_acceptance_rate": rate(sum(not r["refused"] for r in negatives), len(negatives)),
        "accepted_hit_rate": rate(sum(bool(r["evidence_rank"]) for r in positives), len(positives)),
        "positive_top_score_min": min(scores) if scores else None,
        "positive_top_score_avg": round(sum(scores) / len(scores), 4) if scores else None,
        "negative_top_scores": {r["id"]: r["top_score"] for r in negatives},
        "retrieval_seconds_avg": round(sum(r["retrieval_seconds"] for r in rows) / len(rows), 3)
        if rows else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(ROOT / "ai_data.db"))
    parser.add_argument("--questions", default=str(EVAL_DIR / "questions.json"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--min-score", type=float, help="근거 선택 코사인 유사도 하한 (0~1)")
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k는 1 이상이어야 합니다.")
    if args.min_score is not None and not 0 <= args.min_score <= 1:
        parser.error("--min-score는 0~1 사이여야 합니다.")
    if Path(args.label).name != args.label or any(c in args.label for c in '/\\:*?"<>|'):
        parser.error("--label에는 파일명에 사용할 수 있는 이름을 입력하세요.")

    # 예외 발생 시에도 임시 DB를 제거합니다.
    with tempfile.TemporaryDirectory(prefix="rag_eval_") as tmp:
        run(args, Path(tmp))


def run(args, tmp_dir):
    db_copy = tmp_dir / "eval.db"
    # WAL에 남은 커밋도 포함한 일관된 읽기 전용 스냅샷입니다.
    source_uri = Path(args.db).resolve().as_uri() + "?mode=ro"
    source = sqlite3.connect(source_uri, uri=True)
    target = sqlite3.connect(db_copy)
    try:
        source.backup(target)
    finally:
        source.close()
        target.close()
    os.environ["MOBILITY_AI_DB"] = str(db_copy)

    sys.path.insert(0, str(ROOT))
    import knowledge_db  # noqa: E402  (환경변수 설정 후 import)
    import rag  # noqa: E402
    threshold = rag.config.MIN_EVIDENCE_SCORE if args.min_score is None else args.min_score

    # search_knowledge 결과에 page_start/page_end를 붙이기 위해 chunk_id → 범위 맵 생성
    pages = {c["chunk_id"]: (c["page_start"], c["page_end"]) for c in knowledge_db.active_chunks()}
    documents = {d["id"]: d["file_name"] for d in knowledge_db.list_documents()}

    data = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    rows = []

    for q in data["questions"]:
        started = time.perf_counter()
        # 임베딩은 한 번만 수행하고, 하한 적용 전/후의 순위를 각각 평가합니다.
        all_matches = rag.search_knowledge(q["question"], limit=None)
        matches = all_matches[:args.k]
        evidence = rag.select_evidence(all_matches, threshold)
        evidence.sort(key=lambda m: m["score"], reverse=True)
        if rag.config.HYBRID_SEARCH:
            rag.apply_hybrid_ranking(q["question"], evidence)
        evidence = evidence[:args.k]
        elapsed = time.perf_counter() - started
        for m in matches:
            m["page_start"], m["page_end"] = pages.get(m["chunk_id"], (None, None))
            m["file_name"] = documents.get(m["document_id"])
        for m in evidence:
            m["page_start"], m["page_end"] = pages.get(m["chunk_id"], (None, None))
            m["file_name"] = documents.get(m["document_id"])
        document = q.get("document", data.get("document"))
        rank = next((i for i, m in enumerate(matches, 1)
                     if is_hit(m, q["gold_slides"], document)), None)
        evidence_rank = next((i for i, m in enumerate(evidence, 1)
                              if is_hit(m, q["gold_slides"], document)), None)
        row = {
            "id": q["id"],
            "question": q["question"],
            "gold_slides": q["gold_slides"],
            "rank": rank,
            "document": document,
            "refused": not evidence,
            "evidence_rank": evidence_rank,
            "retrieval_seconds": round(elapsed, 4),
            "top_score": round(matches[0]["score"], 4) if matches else None,
            "top": [
                {"score": round(m["score"], 4), "location": m["location"], "type": m["type"],
                 "preview": m["content"][:60].replace("\n", " ")}
                for m in matches
            ],
        }
        rows.append(row)

        mark = "-" if not q["gold_slides"] else ("O" if rank else "X")
        locs = ", ".join(f"{t['location'] or t['type']}({t['score']:.3f})" for t in row["top"][:3])
        print(f"[{mark}] {q['id']} 순위={rank or '-'} 정답={q['gold_slides']} | 상위3: {locs}")

    summary = {
        "label": args.label,
        "db": args.db,
        "min_evidence_score": threshold,
        **summarize(rows, args.k),
    }

    print("\n=== 요약 ===")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    out_dir = EVAL_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{args.label}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n결과 저장: {out}")


if __name__ == "__main__":
    main()

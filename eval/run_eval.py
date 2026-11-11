"""RAG 검색 정확도 평가 (Phase 2).

실행 예:
  .venv\\Scripts\\python.exe eval\\run_eval.py                 # ai_data.db 사본으로 평가
  .venv\\Scripts\\python.exe eval\\run_eval.py --db 다른.db     # 다른 DB(예: v2 재등록 사본)와 비교
  .venv\\Scripts\\python.exe eval\\run_eval.py --label 점수하한   # 결과 파일 이름표

원본 DB는 절대 수정하지 않습니다: 지정한 DB를 임시 폴더로 복사한 뒤 그 사본을 검색합니다.
측정 항목 (gold_slides가 있는 질문만):
  - Hit@1/3/5: 정답 슬라이드를 포함한 청크가 상위 k개 안에 있는 비율
  - MRR: 첫 정답 순위의 역수 평균
문서에 없는 질문(N*)은 최고 점수를 기록해, 이후 점수 하한(min_score) 결정에 씁니다.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = Path(__file__).resolve().parent


def is_hit(match, gold):
    if match["type"] != "document" or match.get("page_start") is None:
        return False
    start = match["page_start"]
    end = match["page_end"] if match.get("page_end") is not None else start
    return any(start <= g <= end for g in gold)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(ROOT / "ai_data.db"))
    parser.add_argument("--questions", default=str(EVAL_DIR / "questions.json"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--label", default="baseline")
    args = parser.parse_args()

    tmp_dir = Path(tempfile.mkdtemp(prefix="rag_eval_"))
    db_copy = tmp_dir / "eval.db"
    shutil.copy2(args.db, db_copy)
    os.environ["MOBILITY_AI_DB"] = str(db_copy)

    sys.path.insert(0, str(ROOT))
    import knowledge_db  # noqa: E402  (환경변수 설정 후 import)
    import rag  # noqa: E402

    # search_knowledge 결과에 page_start/page_end를 붙이기 위해 chunk_id → 범위 맵 생성
    pages = {c["chunk_id"]: (c["page_start"], c["page_end"]) for c in knowledge_db.active_chunks()}

    data = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    rows, positives, negatives = [], [], []

    for q in data["questions"]:
        matches = rag.search_knowledge(q["question"], limit=args.k)
        for m in matches:
            m["page_start"], m["page_end"] = pages.get(m["chunk_id"], (None, None))
        rank = next((i for i, m in enumerate(matches, 1) if is_hit(m, q["gold_slides"])), None)
        row = {
            "id": q["id"],
            "question": q["question"],
            "gold_slides": q["gold_slides"],
            "rank": rank,
            "top_score": round(matches[0]["score"], 4) if matches else None,
            "top": [
                {"score": round(m["score"], 4), "location": m["location"], "type": m["type"],
                 "preview": m["content"][:60].replace("\n", " ")}
                for m in matches
            ],
        }
        rows.append(row)
        (positives if q["gold_slides"] else negatives).append(row)

        mark = "-" if not q["gold_slides"] else ("O" if rank else "X")
        locs = ", ".join(f"{t['location'] or t['type']}({t['score']:.3f})" for t in row["top"][:3])
        print(f"[{mark}] {q['id']} 순위={rank or '-'} 정답={q['gold_slides']} | 상위3: {locs}")

    n = len(positives)
    summary = {
        "label": args.label,
        "db": args.db,
        "questions": n,
        **{f"hit@{k}": round(sum(1 for r in positives if r["rank"] and r["rank"] <= k) / n, 3)
           for k in (1, 3, args.k)},
        "mrr": round(sum(1 / r["rank"] for r in positives if r["rank"]) / n, 3),
        "positive_top_score_min": min(r["top_score"] for r in positives),
        "positive_top_score_avg": round(sum(r["top_score"] for r in positives) / n, 4),
        "negative_top_scores": {r["id"]: r["top_score"] for r in negatives},
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
    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

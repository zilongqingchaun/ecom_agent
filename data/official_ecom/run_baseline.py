"""Compare retrieval on all labeled questions and record answers on behavior cases."""

from functools import lru_cache
import json
import os
import sys

from prepare import PROJECT, ROOT, write_json


def main():
    platform = sys.argv[1]
    if platform not in {"jd", "pdd"}:
        raise ValueError("platform must be jd or pdd")
    folder = ROOT / "exports" / platform
    os.environ.update(json.loads((folder / "index_config.json").read_text(encoding="utf-8")))
    sys.path.insert(0, str(PROJECT))

    from openai import OpenAI
    from app.rag.embedding import embedding
    from app.rag_eval.metric import retrieval_metrics
    from app.rag_eval.rag_eval import RagEvaluator
    from app.settings.settings import settings

    # Both strategies see the same query vector.
    embedding.embed_queries = lru_cache(maxsize=None)(embedding.embed_queries)
    client = OpenAI(api_key=settings.model_api_key, base_url=settings.base_url, timeout=60)
    cases = json.loads((folder / "cases_all.json").read_text(encoding="utf-8"))
    behaviors = json.loads((folder / "behavior_cases.json").read_text(encoding="utf-8"))
    behavior_ids = {case["id"] for case in behaviors}
    results = []
    report_path = folder / "baseline_report.json"
    for mode in ("CHROMA_HNSW", "CHROMA_BM25"):
        evaluator = RagEvaluator(client=client, model=settings.model_name, top_k=5,
                                 search_mode=mode, use_judge=False)
        for case in cases:
            row = {"id": case["id"], "mode": mode, "query": case["query"],
                   "type": case["type"], "split": case["split"],
                   "reference_answer": case["reference_answer"],
                   "relevant_chunk_ids": case["relevant_chunk_ids"]}
            try:
                chunks = evaluator._retrieve(case["query"])
                ids = [chunk["chunk_id"] for chunk in chunks]
                row["retrieved_chunk_ids"] = ids
                if case["id"] in behavior_ids:
                    row["answer"], row["answer_tokens"] = evaluator._generate_answer(
                        case["query"], evaluator._format_contexts(chunks))
                else:
                    row["metrics"] = {
                        str(k): retrieval_metrics(ids, case["relevant_chunk_ids"], k)
                        for k in (1, 3, 5)
                    }
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            results.append(row)
            write_json(report_path, {"platform": platform, "collection": evaluator.collection.name,
                                    "chunk_count": evaluator.collection.count(),
                                    "model": settings.model_name, "results": results})
            if len(results) % 10 == 0:
                print(f"{platform}: {len(results)}/{2 * len(cases)}", flush=True)
    print(report_path)


if __name__ == "__main__":
    main()

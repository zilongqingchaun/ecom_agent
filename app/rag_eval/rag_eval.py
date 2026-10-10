"""运行 RAG 检索和回答质量评估。

示例：
    python -m app.rag_eval.rag_eval
    python -m app.rag_eval.rag_eval --top-k 3 --output app/rag_eval/report.json
    python -m app.rag_eval.rag_eval --retrieval-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from openai import OpenAI

from app.rag.backends.chroma_db import ecom_chroma_backend
from app.rag_eval.dataset import RagEvalCase, load_rag_dataset
from app.rag_eval.metric import judge_rag_answer, retrieval_metrics
from app.rag_eval.prompt import RAG_ANSWER_SYSTEM_PROMPT, RAG_ANSWER_USER_PROMPT
from app.settings.settings import PROJECT_ROOT, settings


class RagEvaluator:
    def __init__(
        self,
        client: OpenAI | None,
        model: str,
        top_k: int = 5,
        search_mode: str = "CHROMA_HNSW",
        retrieval_only: bool = False,
        use_judge: bool = True,
    ):
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        if not retrieval_only and client is None:
            raise ValueError("生成回答时必须提供模型客户端")
        if search_mode not in {"CHROMA_HNSW", "CHROMA_BM25"}:
            raise ValueError("search_mode 只能是 CHROMA_HNSW 或 CHROMA_BM25")
        self.client = client
        self.model = model
        self.top_k = top_k
        self.search_mode = search_mode
        self.retrieval_only = retrieval_only
        self.use_judge = use_judge

        # self.collection = ecom_chroma_backend.chroma
        self.backend=ecom_chroma_backend
        self.collection = ecom_chroma_backend.chroma
        if self.collection is None:
            raise RuntimeError("Chroma collection 尚未初始化")
        if self.collection.count() == 0:
            raise RuntimeError("Chroma collection 为空，请先构建知识库")

    def _retrieve(self, query: str) -> list[dict[str, Any]]:
        top_k = min(self.top_k, self.collection.count())
        if self.search_mode == "CHROMA_HNSW":
            result=self.backend.search(query=query,topk=min(self.top_k, self.collection.count()))
        else:
            result=self.backend.sim_and_bm25_search(query=query, topk=top_k)

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        rrf_scores = (result.get("rrf_scores") or [[]])[0]

        chunks: list[dict[str, Any]] = []
        for index, chunk_id in enumerate(ids):
            metadata = metadatas[index] or {}
            chunks.append(
                {
                    "rank": index + 1,
                    "chunk_id": str(chunk_id),
                    "distance": distances[index] if index < len(distances) else None,
                    "rrf_score": rrf_scores[index] if index < len(rrf_scores) else None,
                    "docs": metadata.get("docs"),
                    "section": metadata.get("section"),
                    "content": documents[index],
                }
            )
        return chunks

    @staticmethod
    def _format_contexts(chunks: list[dict[str, Any]]) -> str:
        if not chunks:
            return "（没有检索到上下文）"
        return "\n\n".join(
            (
                f"[Chunk {chunk['chunk_id']}]\n"
                f"来源：{chunk['docs']}\n"
                f"章节：{chunk['section']}\n"
                f"内容：{chunk['content']}"
            )
            for chunk in chunks
        )

    def _generate_answer(self, query: str, contexts: str) -> tuple[str, int]:
        assert self.client is not None
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": RAG_ANSWER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": RAG_ANSWER_USER_PROMPT.format(
                        query=query,
                        contexts=contexts,
                    ),
                },
            ],
        )
        usage = getattr(response, "usage", None)
        tokens = int(getattr(usage, "total_tokens", 0) or 0)
        return response.choices[0].message.content or "", tokens

    def run_case(self, case: RagEvalCase) -> dict[str, Any]:
        result: dict[str, Any] = {
            "case_id": case.id,
            "query": case.query,
            "relevant_chunk_ids": case.relevant_chunk_ids,
            "reference_answer": case.reference_answer,
            "retrieval": None,
            "generation": None,
            "error": None,
        }
        try:
            chunks = self._retrieve(case.query)
            retrieved_ids = [chunk["chunk_id"] for chunk in chunks]
            result["retrieval"] = {
                "retrieved_chunk_ids": retrieved_ids,
                "metrics": retrieval_metrics(
                    retrieved_ids,
                    case.relevant_chunk_ids,
                    self.top_k,
                ),
                "chunks": chunks,
            }
            if self.retrieval_only:
                return result

            contexts = self._format_contexts(chunks)
            answer, answer_tokens = self._generate_answer(case.query, contexts)
            generation: dict[str, Any] = {
                "answer": answer,
                "metrics": None,
                "answer_tokens": answer_tokens,
                "judge_tokens": 0,
                "judge_error": None,
            }
            result["generation"] = generation

            if self.use_judge:
                try:
                    assert self.client is not None
                    scores, judge_tokens = judge_rag_answer(
                        client=self.client,
                        model=self.model,
                        query=case.query,
                        contexts=contexts,
                        answer=answer,
                        reference_answer=case.reference_answer,
                    )
                    generation["metrics"] = scores
                    generation["judge_tokens"] = judge_tokens
                except Exception as exc:
                    generation["judge_error"] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    def run_all(self, cases: list[RagEvalCase]) -> dict[str, Any]:
        results = [self.run_case(case) for case in cases]
        return {
            "config": {
                "collection": self.collection.name,
                "collection_count": self.collection.count(),
                "top_k": self.top_k,
                "search_mode": self.search_mode,
                "model": self.model,
                "retrieval_only": self.retrieval_only,
                "use_judge": self.use_judge,
            },
            "summary": self._summarize(results),
            "cases": results,
        }

    def _summarize(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        hit_key = f"hit@{self.top_k}"
        recall_key = f"recall@{self.top_k}"
        precision_key = f"precision@{self.top_k}"

        def retrieval_values(name: str) -> list[float]:
            return [
                item["retrieval"]["metrics"][name]
                for item in results
                if item.get("retrieval") is not None
            ]

        def generation_values(name: str) -> list[float]:
            values: list[float] = []
            for item in results:
                generation = item.get("generation")
                metrics = generation.get("metrics") if generation else None
                if metrics and name in metrics:
                    values.append(metrics[name]["score"])
            return values

        def average(values: list[float]) -> float | None:
            return mean(values) if values else None

        answer_tokens = sum(
            (item.get("generation") or {}).get("answer_tokens", 0) for item in results
        )
        judge_tokens = sum(
            (item.get("generation") or {}).get("judge_tokens", 0) for item in results
        )
        return {
            "total_cases": len(results),
            "successful_retrievals": sum(item.get("retrieval") is not None for item in results),
            "errors": sum(item.get("error") is not None for item in results),
            hit_key: average(retrieval_values(hit_key)),
            recall_key: average(retrieval_values(recall_key)),
            precision_key: average(retrieval_values(precision_key)),
            "mrr": average(retrieval_values("reciprocal_rank")),
            "faithfulness": average(generation_values("faithfulness")),
            "answer_relevance": average(generation_values("answer_relevance")),
            "answer_correctness": average(generation_values("answer_correctness")),
            "context_relevance": average(generation_values("context_relevance")),
            "answer_tokens": answer_tokens,
            "judge_tokens": judge_tokens,
            "total_tokens": answer_tokens + judge_tokens,
        }


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def _print_report(report: dict[str, Any]) -> None:
    top_k = report["config"]["top_k"]
    print("\nRAG 评估结果")
    for case in report["cases"]:
        if case["error"]:
            print(f"- {case['case_id']}: 异常 - {case['error']}")
            continue
        metrics = case["retrieval"]["metrics"]
        line = (
            f"- {case['case_id']}: Hit@{top_k}={metrics[f'hit@{top_k}']}, "
            f"Recall@{top_k}={_percent(metrics[f'recall@{top_k}'])}, "
            f"Precision@{top_k}={_percent(metrics[f'precision@{top_k}'])}, "
            f"RR={metrics['reciprocal_rank']:.3f}"
        )
        generation = case.get("generation")
        if generation and generation.get("metrics"):
            generated = generation["metrics"]
            line += (
                f", Faithfulness={_percent(generated['faithfulness']['score'])}, "
                f"Correctness={_percent(generated['answer_correctness']['score'])}"
            )
        print(line)

    summary = report["summary"]
    print("\n平均指标")
    print(f"Hit@{top_k}: {_percent(summary[f'hit@{top_k}'])}")
    print(f"Recall@{top_k}: {_percent(summary[f'recall@{top_k}'])}")
    print(f"Precision@{top_k}: {_percent(summary[f'precision@{top_k}'])}")
    print(f"MRR: {_percent(summary['mrr'])}")
    print(f"Faithfulness: {_percent(summary['faithfulness'])}")
    print(f"Answer Relevance: {_percent(summary['answer_relevance'])}")
    print(f"Answer Correctness: {_percent(summary['answer_correctness'])}")
    print(f"Context Relevance: {_percent(summary['context_relevance'])}")
    print(f"总 tokens: {summary['total_tokens']}")


def main() -> None:
    default_dataset = Path(__file__).with_name("rag_data.json")
    default_output = Path(__file__).with_name("rag_eval_report.json")
    parser = argparse.ArgumentParser(description="运行 RAG 检索与回答质量评估")
    parser.add_argument("--dataset", type=Path, default=default_dataset)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--top-k", type=int, default=settings.rag_top_k)
    parser.add_argument(
        "--search-mode",
        choices=["CHROMA_HNSW", "CHROMA_BM25"],
        default="CHROMA_HNSW",
    )
    parser.add_argument("--model", default=settings.model_name)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--no-judge", action="store_true")
    args = parser.parse_args()

    dataset_path = args.dataset if args.dataset.is_absolute() else PROJECT_ROOT / args.dataset
    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    cases = load_rag_dataset(dataset_path)
    client = None
    if not args.retrieval_only:
        client = OpenAI(api_key=settings.model_api_key, base_url=settings.base_url)

    evaluator = RagEvaluator(
        client=client,
        model=args.model,
        top_k=args.top_k,
        search_mode=args.search_mode,
        retrieval_only=args.retrieval_only,
        use_judge=not args.no_judge,
    )
    report = evaluator.run_all(cases)
    _print_report(report)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n报告已写入：{output_path}")


if __name__ == "__main__":
    main()

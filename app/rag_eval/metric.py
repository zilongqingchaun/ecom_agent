"""RAG 检索指标与 LLM-as-judge 评分。"""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from app.rag_eval.prompt import RAG_JUDGE_PROMPT


def hit_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> bool:
    """Top-K 中至少出现一个相关 chunk 时返回 True，否则返回 False。"""

    relevant = set(relevant_ids)
    return any(chunk_id in relevant for chunk_id in retrieved_ids[:k])


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Top-K 找回的相关 chunk 数 / 全部相关 chunk 数。"""

    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("relevant_ids 不能为空")
    found = relevant.intersection(retrieved_ids[:k])
    return len(found) / len(relevant)


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Top-K 中相关 chunk 数 / K。"""

    if k <= 0:
        raise ValueError("k 必须大于 0")
    relevant = set(relevant_ids)
    found = relevant.intersection(retrieved_ids[:k])
    return len(found) / k


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str], k: int | None = None) -> float:
    """第一个相关 chunk 排名的倒数；Top-K 未命中时为 0。"""

    relevant = set(relevant_ids)
    ranked_ids = retrieved_ids if k is None else retrieved_ids[:k]
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def retrieval_metrics(
    retrieved_ids: list[str], relevant_ids: list[str], k: int
) -> dict[str, float | bool]:
    return {
        f"hit@{k}": hit_at_k(retrieved_ids, relevant_ids, k),
        f"recall@{k}": recall_at_k(retrieved_ids, relevant_ids, k),
        f"precision@{k}": precision_at_k(retrieved_ids, relevant_ids, k),
        "reciprocal_rank": reciprocal_rank(retrieved_ids, relevant_ids, k),
    }


def _parse_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Judge 返回值必须是 JSON 对象")
    return parsed


def _normalized_dimension(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Judge 返回值缺少 {name}")
    score = max(0.0, min(1.0, float(value.get("score"))))
    return {"score": score, "reason": str(value.get("reason", ""))}


def judge_rag_answer(
    client: OpenAI,
    model: str,
    query: str,
    contexts: str,
    answer: str,
    reference_answer: str,
) -> tuple[dict[str, dict[str, Any]], int]:
    """一次模型调用同时评估四个生成指标，降低评估成本。"""

    prompt = RAG_JUDGE_PROMPT.format(
        query=query,
        contexts=contexts,
        answer=answer,
        reference_answer=reference_answer,
    )
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    parsed = _parse_json(response.choices[0].message.content or "")
    dimensions = {
        name: _normalized_dimension(parsed, name)
        for name in (
            "faithfulness",
            "answer_relevance",
            "answer_correctness",
            "context_relevance",
        )
    }
    usage = getattr(response, "usage", None)
    return dimensions, int(getattr(usage, "total_tokens", 0) or 0)

"""RAG 评估数据集的数据结构与加载。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RagEvalCase:
    id: str
    query: str
    relevant_chunk_ids: list[str]
    reference_answer: str


def load_rag_dataset(path: str | Path | None = None) -> list[RagEvalCase]:
    """加载 RAG 测试集，并检查必要字段。"""

    dataset_path = Path(path) if path is not None else Path(__file__).with_name("rag_data.json")
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("RAG 数据集顶层必须是列表")

    cases: list[RagEvalCase] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"第 {index} 条用例必须是 JSON 对象")
        case = RagEvalCase(
            id=str(item.get("id", "")).strip(),
            query=str(item.get("query", "")).strip(),
            relevant_chunk_ids=[str(value) for value in item.get("relevant_chunk_ids", [])],
            reference_answer=str(item.get("reference_answer", "")).strip(),
        )
        if not case.id or not case.query:
            raise ValueError(f"第 {index} 条用例缺少 id 或 query")
        if case.id in seen_ids:
            raise ValueError(f"用例 id 重复：{case.id}")
        if not case.relevant_chunk_ids:
            raise ValueError(f"用例 {case.id} 没有 relevant_chunk_ids")
        if not case.reference_answer:
            raise ValueError(f"用例 {case.id} 没有 reference_answer")
        seen_ids.add(case.id)
        cases.append(case)
    return cases

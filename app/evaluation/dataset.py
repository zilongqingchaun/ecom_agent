"""评估数据集的数据结构与加载。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalCase:
    """单条评估用例；turns 会按顺序交给同一个 Agent 实例。"""

    id: str
    description: str
    turns: list[str]

    # 结果期望
    expected_intent: str | None = None
    expected_keywords: list[str] = field(default_factory=list)
    expected_requires_human: bool | None = None

    # 过程期望
    expected_tools: list[str] = field(default_factory=list)
    min_tool_calls: int | None = None
    max_tokens: int | None = None
    expected_route: str | None = None


def load_dataset(path: str | Path) -> list[EvalCase]:
    """从 ``{"cases": [...]}`` 格式的 JSON 文件加载评估用例。"""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = data.get("cases")
    if not isinstance(cases, list):
        raise ValueError("评估数据集必须包含 cases 列表")
    return [EvalCase(**item) for item in cases]

"""Agent 评估过程中采集的模型调用、工具调用与最终结果。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.utils.response_format import CustomerServiceResponse


@dataclass
class LLMCallRecord:
    purpose: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    tool_calls: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass
class ToolObservation:
    name: str
    arguments: dict
    result: str


@dataclass
class RunTrace:
    case_id: str
    turns: list[str]
    final_response: "CustomerServiceResponse | None" = None
    route: str | None = None
    llm_calls: list[LLMCallRecord] = field(default_factory=list)
    tool_observations: list[ToolObservation] = field(default_factory=list)
    error: str | None = None

    @property
    def total_tokens(self) -> int:
        return sum(call.total_tokens for call in self.llm_calls)

    @property
    def num_llm_calls(self) -> int:
        return len(self.llm_calls)

    @property
    def num_tool_calls(self) -> int:
        return len(self.tool_observations)

    @property
    def tool_call_names(self) -> list[str]:
        return [observation.name for observation in self.tool_observations]

    def to_dict(self) -> dict:
        response = self.final_response
        return {
            "case_id": self.case_id,
            "turns": self.turns,
            "route": self.route,
            "reply": response.reply if response else None,
            "intent": response.intent.value if response else None,
            "requires_human": response.requires_human if response else None,
            "total_tokens": self.total_tokens,
            "num_llm_calls": self.num_llm_calls,
            "num_tool_calls": self.num_tool_calls,
            "tool_calls": self.tool_call_names,
            "error": self.error,
        }

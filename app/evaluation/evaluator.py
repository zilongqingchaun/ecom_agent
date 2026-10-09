"""编排沙箱执行、指标评分和报告聚合。"""

from __future__ import annotations

from dataclasses import dataclass, field

from openai import OpenAI

from app.evaluation import metrics
from app.evaluation.dataset import EvalCase
from app.evaluation.sandbox import Sandbox


@dataclass
class EvalResult:
    case_id: str
    description: str

    tool_accuracy: float | None = None
    tool_efficiency: float | None = None
    token_cost: int = 0
    token_pass: bool | None = None
    process_soundness: float | None = None
    route_match: float | None = None

    intent_match: float | None = None
    keyword_coverage: float | None = None
    requires_human_match: float | None = None
    answer_quality: float | None = None
    faithfulness: float | None = None

    process_score: float | None = None
    result_score: float | None = None
    passed: bool = False
    trace: dict = field(default_factory=dict)
    judge_reasons: dict = field(default_factory=dict)
    error: str | None = None


def _avg(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


class Evaluator:
    def __init__(
        self,
        sandbox: Sandbox,
        client: OpenAI,
        model: str,
        use_judge: bool = True,
        pass_threshold: float = 0.6,
    ):
        self.sandbox = sandbox
        self.client = client
        self.model = model
        self.use_judge = use_judge
        self.pass_threshold = pass_threshold

    def run_case(self, case: EvalCase) -> EvalResult:
        trace = self.sandbox.run(case)
        result = EvalResult(
            case_id=case.id,
            description=case.description,
            trace=trace.to_dict(),
        )
        if trace.error:
            result.error = trace.error
            return result

        try:
            response = trace.final_response
            reply = response.reply if response else ""
            actual_intent = response.intent.value if response else ""
            actual_human = response.requires_human if response else False
            called_tools = trace.tool_call_names

            result.token_cost = trace.total_tokens
            result.tool_accuracy = metrics.tool_accuracy(
                case.expected_tools, called_tools
            )
            result.tool_efficiency = metrics.tool_efficiency(
                case.min_tool_calls, trace.num_tool_calls
            )
            result.token_pass = metrics.token_cost_pass(
                trace.total_tokens, case.max_tokens
            )
            result.route_match = metrics.route_match(
                case.expected_route, trace.route
            )

            result.intent_match = metrics.intent_match(
                case.expected_intent, actual_intent
            )
            result.keyword_coverage = metrics.keyword_coverage(
                case.expected_keywords, reply
            )
            result.requires_human_match = metrics.requires_human_match(
                case.expected_requires_human, actual_human
            )

            if self.use_judge:
                quality, quality_reason = metrics.judge_answer_quality(
                    self.client,
                    self.model,
                    case.turns[-1],
                    reply,
                    case.expected_keywords,
                )
                result.answer_quality = quality / 5.0
                result.judge_reasons["answer_quality"] = quality_reason

                faithful, faithful_reason = metrics.judge_faithfulness(
                    self.client,
                    self.model,
                    reply,
                    trace.tool_observations,
                )
                result.faithfulness = faithful
                result.judge_reasons["faithfulness"] = faithful_reason

                process, process_reason = metrics.judge_process_soundness(
                    self.client,
                    self.model,
                    case.turns[-1],
                    called_tools,
                )
                result.process_soundness = process / 5.0
                result.judge_reasons["process_soundness"] = process_reason

            result.process_score = _avg(
                [
                    result.tool_accuracy,
                    result.tool_efficiency,
                    result.process_soundness,
                    result.route_match,
                ]
            )
            result.result_score = _avg(
                [
                    result.intent_match,
                    result.keyword_coverage,
                    result.requires_human_match,
                    result.answer_quality,
                    result.faithfulness,
                ]
            )
            result.passed = self._decide_pass(result)
        except Exception as exc:
            result.error = f"评分异常 {type(exc).__name__}: {exc}"

        return result

    def run_all(self, cases: list[EvalCase]) -> dict:
        return self._aggregate([self.run_case(case) for case in cases])

    def _decide_pass(self, result: EvalResult) -> bool:
        if result.error or result.token_pass is False:
            return False
        dimensions = [
            result.tool_accuracy,
            result.tool_efficiency,
            result.process_soundness,
            result.route_match,
            result.intent_match,
            result.keyword_coverage,
            result.requires_human_match,
            result.answer_quality,
            result.faithfulness,
        ]
        present = [value for value in dimensions if value is not None]
        return (
            all(value >= self.pass_threshold for value in present)
            if present
            else True
        )

    def _aggregate(self, results: list[EvalResult]) -> dict:
        total = len(results)
        passed = sum(1 for result in results if result.passed)
        total_tokens = sum(result.token_cost for result in results)
        return {
            "summary": {
                "total": total,
                "passed": passed,
                "pass_rate": passed / total if total else 0.0,
                "avg_process_score": _avg(
                    [result.process_score for result in results]
                ),
                "avg_result_score": _avg(
                    [result.result_score for result in results]
                ),
                "total_tokens": total_tokens,
                "avg_tokens_per_case": total_tokens / total if total else 0,
            },
            "cases": [self._result_to_dict(result) for result in results],
        }

    @staticmethod
    def _result_to_dict(result: EvalResult) -> dict:
        return {
            "case_id": result.case_id,
            "description": result.description,
            "passed": result.passed,
            "process": {
                "tool_accuracy": result.tool_accuracy,
                "tool_efficiency": result.tool_efficiency,
                "token_cost": result.token_cost,
                "token_pass": result.token_pass,
                "process_soundness": result.process_soundness,
                "route_match": result.route_match,
                "process_score": result.process_score,
            },
            "result": {
                "intent_match": result.intent_match,
                "keyword_coverage": result.keyword_coverage,
                "requires_human_match": result.requires_human_match,
                "answer_quality": result.answer_quality,
                "faithfulness": result.faithfulness,
                "result_score": result.result_score,
            },
            "judge_reasons": result.judge_reasons,
            "trace": result.trace,
            "error": result.error,
        }

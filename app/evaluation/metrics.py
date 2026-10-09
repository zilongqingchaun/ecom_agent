"""评估指标：代码规则与可选的 LLM judge。"""

from __future__ import annotations

import json

from openai import OpenAI

from app.evaluation.prompts import (
    ANSWER_QUALITY_PROMPT,
    HALLUCINATION_PROMPT,
    PROCESS_SOUNDNESS_PROMPT,
)
from app.evaluation.trace import ToolObservation


def tool_accuracy(expected: list[str], called: list[str]) -> float | None:
    """期望工具中实际被调用的比例。"""

    if not expected:
        return None
    called_set = set(called)
    return sum(1 for tool in expected if tool in called_set) / len(expected)


def tool_efficiency(min_calls: int | None, actual: int) -> float | None:
    """理论最少调用次数与实际调用次数之比。"""

    if min_calls is None:
        return None
    if actual <= 0:
        return 1.0 if min_calls == 0 else 0.0
    return min(1.0, min_calls / actual)


def token_cost_pass(total_tokens: int, budget: int | None) -> bool | None:
    if budget is None:
        return None
    return total_tokens <= budget


def route_match(expected: str | None, actual: str | None) -> float | None:
    if expected is None:
        return None
    return 1.0 if expected == actual else 0.0


def intent_match(expected: str | None, actual: str) -> float | None:
    if expected is None:
        return None
    return 1.0 if expected == actual else 0.0


def keyword_coverage(expected: list[str], reply: str) -> float | None:
    if not expected:
        return None
    return sum(1 for keyword in expected if keyword in reply) / len(expected)


def requires_human_match(expected: bool | None, actual: bool) -> float | None:
    if expected is None:
        return None
    return 1.0 if expected == actual else 0.0


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return json.loads(raw)


def _judge(client: OpenAI, model: str, prompt: str) -> dict:
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return _parse_json(response.choices[0].message.content or "")


def judge_answer_quality(
    client: OpenAI,
    model: str,
    user_input: str,
    reply: str,
    reference: list[str] | None = None,
) -> tuple[float, str]:
    prompt = ANSWER_QUALITY_PROMPT.format(
        user_input=user_input,
        reply=reply,
        reference="、".join(reference) if reference else "（无）",
    )
    try:
        data = _judge(client, model, prompt)
        return float(data["score"]), data.get("reason", "")
    except Exception as exc:  # 单项 judge 失败不应中断整条用例
        return 0.0, f"质量评分失败: {type(exc).__name__}: {exc}"


def judge_faithfulness(
    client: OpenAI,
    model: str,
    reply: str,
    observations: list[ToolObservation],
) -> tuple[float, str]:
    observation_text = (
        "\n".join(
            f"- {item.name}({item.arguments}) → {item.result}" for item in observations
        )
        if observations
        else "（本次会话未调用工具）"
    )
    prompt = HALLUCINATION_PROMPT.format(
        reply=reply,
        observations=observation_text,
    )
    try:
        data = _judge(client, model, prompt)
        faithful = bool(data["faithful"])
        return (1.0 if faithful else 0.0), data.get("reason", "")
    except Exception as exc:
        return 0.0, f"幻觉检测失败: {type(exc).__name__}: {exc}"


def judge_process_soundness(
    client: OpenAI,
    model: str,
    user_input: str,
    tool_sequence: list[str],
) -> tuple[float, str]:
    prompt = PROCESS_SOUNDNESS_PROMPT.format(
        user_input=user_input,
        tool_sequence=" → ".join(tool_sequence) if tool_sequence else "（未调用工具）",
    )
    try:
        data = _judge(client, model, prompt)
        return float(data["score"]), data.get("reason", "")
    except Exception as exc:
        return 0.0, f"过程评分失败: {type(exc).__name__}: {exc}"

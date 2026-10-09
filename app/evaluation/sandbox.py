"""隔离运行评估用例，并采集模型与工具调用轨迹。"""
#_instrument里面调用_wrap_create和_wrap_call_tool，在原本调用api的基础上包了一层记录
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from uuid import uuid4

from app.evaluation.dataset import EvalCase
from app.evaluation.trace import LLMCallRecord, RunTrace, ToolObservation
from app.settings.settings import settings
from app.utils.response_format import CustomerServiceResponse


class Sandbox:
    """当前项目单 Agent 模式的评估沙箱。"""

    def __init__(self, mode: str = "single", tmp_root: str | Path | None = None):
        if mode != "single":
            raise ValueError("当前项目的评估沙箱暂只支持 single 模式")
        self.mode = mode
        self._temporary_directory = None
        if tmp_root is None:
            self._temporary_directory = tempfile.TemporaryDirectory(prefix="eval_sandbox_")
            self.tmp_root = Path(self._temporary_directory.name)
        else:
            self.tmp_root = Path(tmp_root)
            self.tmp_root.mkdir(parents=True, exist_ok=True)

    def session_path_for(self, case_id: str) -> Path:
        """每次执行都使用全新的会话目录，防止历史记录污染评估。"""

        path = self.tmp_root / f"{case_id}_{uuid4().hex}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _build_agent():
        from app.agent.ecom_single_agent import ecom_single_agent

        return ecom_single_agent(
            model=settings.model_name,
            temperature=settings.model_temperature,
        )

    def run(self, case: EvalCase) -> RunTrace:
        """执行一条用例；异常只记录到该用例，不中断整批评估。"""

        trace = RunTrace(case_id=case.id, turns=list(case.turns))
        patches: list[tuple[object, str, object]] = []
        original_settings = {
            "memory_enable": settings.memory_enable,
            "enable_mcp": settings.enable_mcp,
            "session_path": settings.session_path,
            "user_id": settings.user_id,
        }

        try:
            settings.memory_enable = False
            settings.enable_mcp = False
            settings.session_path = self.session_path_for(case.id)
            settings.user_id = case.id

            agent = self._build_agent()
            self._instrument(agent, trace, patches)

            result = None
            for turn in case.turns:
                result = agent.chat(turn)
            if result is not None:
                trace.final_response = CustomerServiceResponse.model_validate_json(result)
        except Exception as exc:
            trace.error = f"{type(exc).__name__}: {exc}"
        finally:
            for obj, attribute, original in reversed(patches):
                setattr(obj, attribute, original)
            for name, value in original_settings.items():
                setattr(settings, name, value)
            # 不调用 agent.close()，避免评估额外触发长期记忆提取。

        return trace

    def _instrument(
        self,
        agent,
        trace: RunTrace,
        patches: list[tuple[object, str, object]],
    ) -> None:
        completions = agent.client.chat.completions
        patches.append((completions, "create", completions.create))
        completions.create = self._wrap_create(completions.create, trace)

        manager = agent.tool_manager
        patches.append((manager, "call_tool", manager.call_tool))
        manager.call_tool = self._wrap_call_tool(manager.call_tool, trace)

    def _wrap_create(self, original, trace: RunTrace):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            response = original(*args, **kwargs)
            latency_ms = (time.perf_counter() - start) * 1000
            self._record_llm_call(
                trace,
                response,
                latency_ms,
                purpose=self._guess_purpose(kwargs),
            )
            return response

        return wrapper

    @staticmethod
    def _wrap_call_tool(original, trace: RunTrace):
        def wrapper(tool_name: str, **kwargs):
            try:
                result = original(tool_name, **kwargs)
            except Exception as exc:
                trace.tool_observations.append(
                    ToolObservation(
                        name=tool_name,
                        arguments=dict(kwargs),
                        result=json.dumps(
                            {"error": str(exc)}, ensure_ascii=False, default=str
                        ),
                    )
                )
                raise

            trace.tool_observations.append(
                ToolObservation(
                    name=tool_name,
                    arguments=dict(kwargs),
                    result=json.dumps(result, ensure_ascii=False, default=str),
                )
            )
            return result

        return wrapper

    @staticmethod
    def _guess_purpose(kwargs: dict) -> str:
        if kwargs.get("response_format"):
            return "extract"
        if kwargs.get("tools"):
            return "react"
        return "chat"

    @staticmethod
    def _record_llm_call(
        trace: RunTrace,
        response,
        latency_ms: float,
        purpose: str,
    ) -> None:
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or 0

        tool_calls = []
        try:
            message = response.choices[0].message
            for tool_call in getattr(message, "tool_calls", None) or []:
                tool_calls.append(
                    {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    }
                )
        except (AttributeError, IndexError):
            pass

        trace.llm_calls.append(
            LLMCallRecord(
                purpose=purpose,
                model=getattr(response, "model", "") or "",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                tool_calls=tool_calls,
                latency_ms=latency_ms,
            )
        )

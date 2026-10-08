import json
from pathlib import Path
from app.settings.settings import settings
from openai import OpenAI
from app.tool_manager.tool_manager import tool_manager
from app.prompt.customer_service import SYSTEM_PROMPT
from app.utils.response_format import CustomerServiceResponse
from app.memory_manager.memory_manager import MemoryManager
class ecom_single_agent:
    # 需要配置 session_path、model、temperature 等参数
    def __init__(self, model: str, temperature: float):

        self.client=OpenAI(api_key=settings.model_api_key, base_url=settings.base_url)

        self.model = model
        self.temperature = temperature

        # session_path 指向会话 JSON 文件所在目录；文件名为 <user_id>.json
        self.session_path=settings.session_path
        self.user_id=settings.user_id
        self.raw_history: list[dict] = []
        self.summary: str = ""
        self.memory: MemoryManager | None = None
        
        self.load_session()
        self._memory_saved_until = len(self.raw_history)
        if settings.memory_enable:
            self.memory = MemoryManager(settings.memory_path, self.user_id)
        
        self.tool_manager= tool_manager  # 这里可以初始化工具管理器，用于处理工具调用




    #编写chat方法，管理输入输出
    def chat(self, input_text: str):
        #把input包装成字典加入历史记录
        user_message = {"role": "user", "content": input_text}
        self.raw_history.append(user_message)
        if settings.memory_enable and self.memory is not None:
            self.memory.update_short_term_memory([user_message])
        #让模型根据系统提示词回复，明确可处理的问题和可调用的工具
        response=self.react_loop()
        #转换为格式化的输出
        #判断意图、置信度（ai自己判断）、回复、是否需要人工、追问的问题。
        #需要再调用一次模型
        structured_response=self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": (
                          "请将客服回复转换为 JSON，reply 必须原样保留。只输出一个 JSON 对象，格式为："
                          '{"intent":"greeting","confidence":0.95,"reply":"原回复",'
                          '"requires_human":false,"follow_up_question":null}。'
                          "intent 只能是 order_query、return_request、product_consult、complaint、"
                          "after_sale、promotion、account、greeting、other 之一。"
                      )},
                      {"role": "assistant", "content": response}],
            temperature=settings.structured_response_temperature,
            response_format={"type": "json_object"},
        )
        result = structured_response.choices[0].message.content or "{}"
        validated_result = CustomerServiceResponse.model_validate_json(result)
        self.save_session()
        return validated_result.model_dump_json()

    def close(self):
        """保存会话记录，并在启用记忆时提取新增长期记忆。"""
        self.save_session()
        if not settings.memory_enable or self.memory is None:
            return
        end = len(self.raw_history)
        recent_history = self.raw_history[self._memory_saved_until:end]
        if not recent_history:
            return
        self.memory.update_long_term_memory(recent_history)
        self._memory_saved_until = end

    def save_session(self):
        """仅将尚未保存的对话消息和变化后的摘要追加到 JSONL 日志。"""
        if not self.session_path:
            raise ValueError("请先设置 session_path（会话文件所在目录）")
        if not self.user_id:
            raise ValueError("请先设置 user_id")

        session_file = Path(self.session_path) / f"{self.user_id}.jsonl"
        session_file.parent.mkdir(parents=True, exist_ok=True)
        saved_until = getattr(self, "_session_saved_until", 0)
        saved_summary = getattr(self, "_session_saved_summary", "")
        new_messages = self.raw_history[saved_until:]
        summary_changed = self.summary != saved_summary
        if not new_messages and not summary_changed:
            return

        event = {"type": "update", "messages": new_messages}
        if summary_changed:
            event["summary"] = self.summary
        # 单次更新写成一行；只序列化新增消息，避免重写整段历史。
        with session_file.open("a", encoding="utf-8", newline="\n") as file:
            file.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            file.flush()

        self._session_saved_until = len(self.raw_history)
        self._session_saved_summary = self.summary

    def react_loop(self):
        #加载模型并调用模型进行推理，返回结果
#openai：message基本格式
# {
#     'role': 'assistant',
#     'content': None,
#     'tool_calls': [
#         {
#             'id': 'call_abc',
#             'type': 'function',
#             'function': {
#                 'name': 'get_weather',
#                 'arguments': '{"city":"北京"}'
#             }
#         }
#     ]
# }
        
        last_content = None
        for i in range(settings.react_step):
            prompted_messages=self.build_messages()
            response=self.client.chat.completions.create(
                model=self.model,
                messages=prompted_messages,
                temperature=self.temperature,
                tools=self.tool_manager.load_tools()
            )
            # 输出包含 content 和 tool_calls；没有工具调用时直接结束
            message = response.choices[0].message
            
            tool_calls = message.tool_calls
            #无工具调用直接结束并返回结果
            if not tool_calls:
                content = message.content or ""

                self.raw_history.append({
                    "role": "assistant",
                    "content": content
                })

                return content

            # 以 Chat Completions 接受的格式记录助手工具调用消息。
            self.raw_history.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in tool_calls
                ],
            })

            #得到工具调用 工具的输出需要另外实现
            
            for tc in tool_calls:
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments)

                try:
                    tool_output = self.tool_manager.call_tool(
                        tool_name,
                        **tool_args
                    )
                except Exception as e:
                    tool_output = {
                        "error": str(e)
                    }

                # 每一个 tool_call_id 都必须对应一个 tool message
                self.raw_history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(
                        tool_output,
                        ensure_ascii=False,
                        default=str
                    )
                })

        # 注意：这里不要 return
        # 下一轮继续让 LLM 根据工具结果生成回复

        return "抱歉，处理步骤超过最大限制。"

        # 达到最大迭代次数时返回最后一段可用内容。
        # return last_content or "工具调用次数已达到上限，请稍后重试。"
            
    def build_messages(self):
        messages: list[dict] = []
        # 把记忆上下文附加到系统提示词；尚未配置 memory 时只使用基础提示词。
        system_prompt = SYSTEM_PROMPT
        if settings.memory_enable and self.memory is not None:
            memory_prompt = self.memory.build_memory_prompt()
            if memory_prompt:
                system_prompt = f"{system_prompt}\n\n## 用户长短期记忆\n{memory_prompt}"
        messages.append({"role": "system", "content": system_prompt})
        # 添加历史记录
        messages.extend(self.raw_history)
        return messages



            

    def load_session(self):
        """优先从 JSONL 追加日志恢复会话，并兼容读取旧版 JSON 快照。"""
        if not self.session_path:
            raise ValueError("请先设置 session_path（会话文件所在目录）")
        if not self.user_id:
            raise ValueError("请先设置 user_id")

        directory = Path(self.session_path)
        log_file = directory / f"{self.user_id}.jsonl"
        legacy_file = directory / f"{self.user_id}.json"
        self.raw_history = []
        self.summary = ""

        if log_file.exists():
            lines = log_file.read_text(encoding="utf-8").splitlines()
            for index, line in enumerate(lines):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    # 进程在追加时异常退出可能留下不完整的最后一行；忽略该行即可恢复此前记录。
                    if index == len(lines) - 1:
                        break
                    raise ValueError(f"会话日志第 {index + 1} 行不是有效 JSON：{log_file}") from exc
                if not isinstance(event, dict):
                    raise ValueError(f"会话日志事件必须是 JSON 对象：{log_file}")
                if event.get("type") in {"messages", "update"}:
                    messages = event.get("messages")
                    if not isinstance(messages, list) or any(not isinstance(item, dict) for item in messages):
                        raise ValueError(f"会话日志中的 messages 必须是对象列表：{log_file}")
                    self.raw_history.extend(messages)
                if event.get("type") in {"summary", "update"} and "summary" in event:
                    summary = event.get("summary")
                    if not isinstance(summary, str):
                        raise ValueError(f"会话日志中的 summary 必须是字符串：{log_file}")
                    self.summary = summary
        elif legacy_file.exists():
            try:
                with legacy_file.open("r", encoding="utf-8") as file:
                    session_data = json.load(file)
            except json.JSONDecodeError as exc:
                raise ValueError(f"会话文件不是有效的 JSON：{legacy_file}") from exc
            if not isinstance(session_data, dict):
                raise ValueError(f"会话文件顶层内容必须是 JSON 对象：{legacy_file}")
            history = session_data.get("raw_history", [])
            summary = session_data.get("summary", "")
            if not isinstance(history, list) or any(not isinstance(item, dict) for item in history):
                raise ValueError(f"会话文件中的 raw_history 必须是对象列表：{legacy_file}")
            if not isinstance(summary, str):
                raise ValueError(f"会话文件中的 summary 必须是字符串：{legacy_file}")
            self.raw_history = history
            self.summary = summary

        # 旧 JSON 记录会在下次保存时迁移到 JSONL；JSONL 已存在时以日志为准。
        self._session_saved_until = 0 if not log_file.exists() else len(self.raw_history)
        self._session_saved_summary = "" if not log_file.exists() else self.summary
        return self.raw_history, self.summary

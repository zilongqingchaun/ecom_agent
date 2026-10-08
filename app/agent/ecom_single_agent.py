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
        self.session_path: str | Path | None = None
        self.user_id: str | None = None
        self.raw_history: list[dict] = []
        self.summary: str = ""
        self.memory: MemoryManager | None = None
        
        self.load_session()
        
        self.tool_manager= tool_manager  # 这里可以初始化工具管理器，用于处理工具调用




    #编写chat方法，管理输入输出
    def chat(self, input_text: str):
        #把input包装成字典加入历史记录
        self.raw_history.append({"role": "user", "content": input_text})
        #让模型根据系统提示词回复，明确可处理的问题和可调用的工具
        response=self.react_loop()
        #转换为格式化的输出
        #判断意图、置信度（ai自己判断）、回复、是否需要人工、追问的问题。
        #需要再调用一次模型
        structured_response=self.client.chat.completions.parse(
            model=self.model,
            messages=[{"role": "system", "content": "请将以下内容转换为结构化 JSON 格式，包含意图、置信度、回复、是否需要人工、追问的问题。reply直接粘贴原文，不要改动"},
                      {"role": "assistant", "content": response}],
            temperature=0,
            response_format=CustomerServiceResponse
        )
        result=structured_response.choices[0].message.content
        return result
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
        prompted_messages=self.build_messages()
        for i in range(settings.react_step):
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
                if message.content is not None:
                    self.raw_history.append({"role": "assistant", "content": message.content})
                return message.content

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
            
            for tool_call in tool_calls:
                too_id = tool_call.id
                tool_name = tool_call.function.name
                tool_args = tool_call.function.arguments
                #在这里调用工具函数，获取工具输出
                tool_output = self.tool_manager.call_tool(tool_name, **json.loads(tool_args))
                self.raw_history.append({
                    "role": "tool",
                    "tool_call_id": too_id,
                    "content": json.dumps(tool_output, ensure_ascii=False, default=str),
                })


            #如果超过最大迭代次数也没有得到结果，直接返回最后的结果
            return response.choices[0].message.content
            
    def build_messages(self):
        messages: list[dict] = []
        # 把记忆上下文附加到系统提示词；尚未配置 memory 时只使用基础提示词。
        system_prompt = SYSTEM_PROMPT
        if self.memory is not None:
            memory_prompt = self.memory.build_memory_prompt()
            if memory_prompt:
                system_prompt = f"{system_prompt}\n\n## 用户长短期记忆\n{memory_prompt}"
        messages.append({"role": "system", "content": system_prompt})
        # 添加历史记录
        messages.extend(self.raw_history)
        return messages



            

    def load_session(self):
        """根据 session_path 和 user_id 加载用户会话 JSON。

        文件格式为 {"raw_history": [...], "summary": "..."}。
        文件不存在时视为新会话，初始化为空历史和空摘要。
        """
        if not self.session_path:
            raise ValueError("请先设置 session_path（会话文件所在目录）")
        if not self.user_id:
            raise ValueError("请先设置 user_id")

        session_file = Path(self.session_path) / f"{self.user_id}.json"
        if not session_file.exists():
            self.raw_history = []
            self.summary = ""
            return self.raw_history, self.summary

        try:
            with session_file.open("r", encoding="utf-8") as file:
                session_data = json.load(file)
        except json.JSONDecodeError as exc:
            raise ValueError(f"会话文件不是有效的 JSON：{session_file}") from exc

        if not isinstance(session_data, dict):
            raise ValueError(f"会话文件顶层内容必须是 JSON 对象：{session_file}")

        history = session_data.get("raw_history", [])
        summary = session_data.get("summary", "")
        if not isinstance(history, list) or any(not isinstance(item, dict) for item in history):
            raise ValueError(f"会话文件中的 raw_history 必须是对象列表：{session_file}")
        if not isinstance(summary, str):
            raise ValueError(f"会话文件中的 summary 必须是字符串：{session_file}")

        self.raw_history = history
        self.summary = summary
        return self.raw_history, self.summary

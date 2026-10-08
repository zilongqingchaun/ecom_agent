from pathlib import Path

from app.memory_manager.long_term_memory import LongTermMemory
from app.memory_manager.shrot_term_memroy import ShortTermMemory


class MemoryManager:
    """统一管理短期记忆和长期记忆的对外接口。"""

    def __init__(self, memory_path: str | Path, user_id: str):
        self.short_term_memory = ShortTermMemory()
        self.long_term_memory = LongTermMemory(
            memory_path=memory_path,
            user_id=user_id,
        )

    def update_short_term_memory(self, recent_raw_history: list[dict]) -> list[str]:
        """根据最近的对话更新短期事实，并返回当前短期事实列表。"""
        self.short_term_memory.load_facts(recent_raw_history)
        return list(self.short_term_memory.short_term_facts)

    def get_short_term_memory(self) -> list[str]:
        """获取当前会话的短期事实。"""
        return list(self.short_term_memory.short_term_facts)

    def update_long_term_memory(self, raw_history: list[dict]) -> list[dict]:
        """从对话中抽取并保存长期事实，返回更新后的长期记忆。"""
        self.long_term_memory.save_memory(raw_history)
        return list(self.long_term_memory.long_term_memory)

    def get_long_term_memory(self) -> list[dict]:
        """获取当前用户已加载的长期记忆。"""
        return list(self.long_term_memory.long_term_memory)

    def build_memory_prompt(self) -> str:
        """将长短期记忆格式化为可附加到系统提示词的上下文。"""
        short_facts = self.get_short_term_memory()
        long_facts = self.get_long_term_memory()

        short_text = "\n".join(f"- {fact}" for fact in short_facts) or "- 暂无"
        long_text = "\n".join(
            f"- [{fact.get('category', 'other')}] {fact.get('content', '')}"
            for fact in long_facts
            if isinstance(fact, dict) and fact.get("content")
        ) or "- 暂无"

        return (
            "以下是当前用户的记忆，仅用于个性化回复；若与用户当前明确提供的信息冲突，以当前信息为准。\n"
            f"短期记忆：\n{short_text}\n"
            f"长期记忆：\n{long_text}"
        )

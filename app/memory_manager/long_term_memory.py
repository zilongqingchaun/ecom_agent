from pathlib import Path
import json
from app.settings.settings import settings
from openai import OpenAI
from app.prompt.memory import LTM_EXTRACTION_PROMPT


class LongTermMemory:
    def __init__(self, memory_path: str | Path, user_id: str):
        self.memory_path = Path(memory_path)
        self.user_id = user_id
        self.long_term_memory: list[dict] = []
        self.client = OpenAI(
            api_key=settings.model_api_key,
            base_url=settings.base_url,
        )
        if settings.memory_enable:
            self.load_memory()

    def _file(self) -> Path:
        return self.memory_path / f"{self.user_id}.json"

    def load_memory(self):
        file = self._file()
        if file.exists():
            with open(file, "r", encoding="utf-8") as f:
                self.long_term_memory = json.load(f)
        else:
            self.long_term_memory = []

    def _persist(self):
        file = self._file()
        file.parent.mkdir(parents=True, exist_ok=True)
        with open(file, "w", encoding="utf-8") as f:
            json.dump(self.long_term_memory, f, ensure_ascii=False, indent=2)

    def save_memory(self, raw_history: list[dict]):
        existing = (
            json.dumps(self.long_term_memory, ensure_ascii=False, indent=2)
            if self.long_term_memory else "（暂无）"
        )

        # 拷贝一份，别污染调用方
        messages = [
            {
                "role": "system",
                "content": LTM_EXTRACTION_PROMPT.format(existing_ltm=existing),
            },
            *raw_history,
        ]

        response = self.client.chat.completions.create(
            model=settings.model_name,
            messages=messages,
            temperature=0,                              # 抽取任务用 0
            response_format={"type": "json_object"},    # 强制 JSON
        )

        raw = response.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return ""   # demo 阶段先忽略

        new_facts = data.get("facts", [])
        if new_facts:
            self.long_term_memory.extend(new_facts)     # 只 extend facts
            self._persist()                             # 写回文件

        
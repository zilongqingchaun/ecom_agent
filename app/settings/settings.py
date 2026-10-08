from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """普通配置在此维护，密钥从 .env 读取；相对路径以项目根目录为基准。"""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8-sig",
        validate_default=True,  # 默认的相对路径也经过路径解析。
    )

    # 对话模型；API Key 在项目根目录 .env 中填写。
    base_url: str = "https://api.deepseek.com"
    model_name: str = "deepseek-v4-flash"
    model_api_key: str = Field(default="", repr=False)
    model_temperature: float = 0.7
    structured_response_temperature: float = 0
    model_mode: Literal["single", "multiple"] = "single"
    react_step: int = Field(default=5, ge=1)
    enable_mcp: bool = False

    # 用户、会话和长期记忆。
    user_id: str = "default_user"
    session_path: Path = Path("sessions")
    memory_path: Path = Path("memories")
    memory_enable: bool = True
    memory_temperature: float = 0

    # RAG 数据目录、集合和检索数量。
    chroma_collection_name: str = "ecom_collection"
    chroma_preset_path: Path = Path("app/rag/backends/chroma_db")
    knowledge_rag_path: Path = Path("app/rag/knowledge")
    rag_top_k: int = Field(default=5, ge=1)

    # 向量模型。
    embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_model_name: str = "text-embedding-v4"
    embedding_model_api_key: str = Field(default="", repr=False)
    embedding_dimensions: int = Field(default=1024, ge=1)
    embedding_batch_size: int = Field(default=10, ge=1)

    @field_validator(
        "session_path", "memory_path", "chroma_preset_path", "knowledge_rag_path",
        mode="before",
    )
    @classmethod
    def resolve_project_path(cls, value: str | Path) -> Path:
        if isinstance(value, str) and not value.strip():
            raise ValueError("路径不能为空")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()


settings = Settings()

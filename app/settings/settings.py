#模型配置需求，先写这么多，后面的再补充

from pydantic_settings import BaseSettings
from pathlib import Path
class Settings(BaseSettings):
    base_url: str = "http://localhost:8000"
    model_name: str = "gpt-4"
    model_api_key: str = ""
    model_temperature: float = 0.7
    model_embedding: str = "text-embedding-3-large"
    model_mode: str = "single"
    react_step: int = 5
    enable_mcp: bool = False

    #rag相关配置
    chroma_collection_name: str = "ecom_collection"
    chroma_preset_path: Path = Path(Path(r"D:\code\project\my_agent\my_agent_ecom\app\rag\backends\chroma_db"))
    knowledge_rag_path: Path = Path(Path(r"D:\code\project\my_agent\my_agent_ecom\app\rag\knowledge"))
    #嵌入模型配置
    embedding_model_name: str = "text-embedding-3-large"
    embedding_model_api_key: str = ""

    memory_enable: bool = True

settings = Settings()
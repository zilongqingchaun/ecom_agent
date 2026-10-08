from app.settings.settings import settings
from app.rag.chunk import Chunk
from openai import OpenAI
class Embedding:
    def __init__(self):
        self.embedding_model_name = settings.embedding_model_name
        self.client = OpenAI(
            api_key=settings.embedding_model_api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def embed(self, chunks: list[Chunk]) -> list[list[float]]:
        texts = [chunk.content for chunk in chunks]
        resp = self.client.embeddings.create(
            model=self.embedding_model_name,
            input=texts,
            dimensions=1024,
        )
        return [item.embedding for item in resp.data]

    def embed_queries(self, query: str) -> list[float]:
        resp = self.client.embeddings.create(
            model=self.embedding_model_name,
            input=[query],
            dimensions=1024,
        )
        return resp.data[0].embedding


embedding=Embedding()

import httpx
from app.settings.settings import settings
from app.rag.chunk import Chunk
from openai import OpenAI
class Embedding:
    def __init__(self):
        self.embedding_model_name = settings.embedding_model_name
        self.client = OpenAI(
            api_key=settings.embedding_model_api_key,
            base_url=settings.embedding_base_url,
            http_client=httpx.Client(trust_env=False),
        )

    def embed(self, chunks: list[Chunk]) -> list[list[float]]:
        texts = [chunk.content for chunk in chunks]
        embeddings: list[list[float]] = []
        batch_size = settings.embedding_batch_size
        for start in range(0, len(texts), batch_size):
            resp = self.client.embeddings.create(
                model=self.embedding_model_name,
                input=texts[start:start + batch_size],
                dimensions=settings.embedding_dimensions,
                encoding_format="float",
            )
            embeddings.extend(item.embedding for item in sorted(resp.data, key=lambda item: item.index))
        return embeddings

    def embed_queries(self, query: str) -> list[float]:
        resp = self.client.embeddings.create(
            model=self.embedding_model_name,
            input=[query],
            dimensions=settings.embedding_dimensions,
            encoding_format="float",
        )
        return resp.data[0].embedding


embedding=Embedding()

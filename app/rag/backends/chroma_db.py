from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from typing import Optional
import chromadb
from pathlib import Path
from app.rag.chunk import make_chunk
from app.rag.embedding import embedding
from app.settings.settings import settings
# {
#     "id": "chunk_001",

#     "document": "退款通常会在审核通过后的3到5个工作日到账。",

#     "metadata": {
#         "docs": "退款政策",
#         "section": "退款时间"
#     },

#     "embedding": [...]
# }



class chroma_backend:
    def __init__(self):
        self.collection_name:str=settings.chroma_collection_name
        self.preset_path:Path=settings.chroma_preset_path
        self.chroma_client: Optional[ClientAPI] = None
        self.chroma: Optional[Collection] = None
        #加载数据库
        self.load_chroma()

    def upsert_collection(self):
        assert self.chroma is not None, "chroma 未初始化，请先调用 load_chroma()"
        chunks=make_chunk.chunk_by_h2()
        embeddings=embedding.embed(chunks)
        self.chroma.upsert(ids=[chunk.chunk_id for chunk in chunks],
                           metadatas=[{"docs":chunk.docs,"section":chunk.section} for chunk in chunks],
                           documents=[chunk.content for chunk in chunks],embeddings=embeddings)
        



    def load_chroma(self):
        #加载chroma数据库
        if not self.preset_path.exists():
            self.preset_path.mkdir(parents=True, exist_ok=True)
        #设立永久化客户都
        self.chroma_client=chromadb.PersistentClient(path=self.preset_path)
        # 读取或新建表
        self.chroma=self.chroma_client.get_or_create_collection(name=self.collection_name)
        
    def search(self,query:str):
        #在chroma数据库中进行检索 并返回检索结果
        query_embedding=embedding.embed_queries(query)
        results=self.chroma.query(query_embeddings=query_embedding,n_results=5)
        return results

ecom_chroma_backend=chroma_backend()
    
#返回格式
# {
#     "ids": [["chunk_001", "chunk_002", "chunk_003", ...]],
#     "documents": [["退款通常会在审核通过后的3到5个工作日到账。", "...", "..."]],
#     "metadatas": [[{"docs": "退款政策", "section": "退款时间"}, {...}, {...}]],
#     "distances": [[0.12, 0.34, 0.56, ...]],
#     "embeddings": None,   # 默认不返回
#     "uris": None,
#     "data": None,
# }
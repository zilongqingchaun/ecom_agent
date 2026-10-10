from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from typing import Optional
import chromadb
import json
from pathlib import Path
from app.rag.chunk import make_chunk
from app.rag.embedding import embedding
from app.settings.settings import settings
from rank_bm25 import BM25Okapi
import jieba


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
        self.bm25_cache_path:Path=self.preset_path / "bm25_tokens.json"
        self.chroma_client: Optional[ClientAPI] = None
        self.chroma: Optional[Collection] = None
        self.bm25_index: Optional[BM25Okapi] = None
        self.bm25_ids: list[str] = []
        #加载数据库
        self.load_chroma()

    def upsert_collection(self):
        assert self.chroma is not None, "chroma 未初始化，请先调用 load_chroma()"
        chunks=make_chunk.chunk_by_h2()
        embeddings=embedding.embed(chunks)
        self.chroma.upsert(ids=[chunk.chunk_id for chunk in chunks],
                           metadatas=[{"docs":chunk.docs,"section":chunk.section} for chunk in chunks],
                           documents=[chunk.content for chunk in chunks],embeddings=embeddings)
        # 知识库内容变化后同步重建 BM25 分词缓存。
        self._build_and_save_bm25_cache()
        



    def load_chroma(self):
        #加载chroma数据库
        if not self.preset_path.exists():
            self.preset_path.mkdir(parents=True, exist_ok=True)
        #设立永久化客户都
        self.chroma_client=chromadb.PersistentClient(path=str(self.preset_path))
        # 读取或新建表
        self.chroma=self.chroma_client.get_or_create_collection(name=self.collection_name)
        self._load_or_build_bm25()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """统一文档与查询的中文分词规则。"""

        return [token.strip() for token in jieba.cut(text) if token.strip()]

    def _load_or_build_bm25(self):
        """加载本地分词缓存；缓存缺失或失效时从 Chroma 重建。"""

        assert self.chroma is not None, "chroma 未初始化"
        if self.chroma.count() == 0:
            self.bm25_index = None
            self.bm25_ids = []
            return

        current_ids = [str(item) for item in self.chroma.get(include=[]).get("ids", [])]
        if self.bm25_cache_path.exists():
            try:
                cache = json.loads(self.bm25_cache_path.read_text(encoding="utf-8"))
                cached_ids = [str(item) for item in cache.get("ids", [])]
                tokenized_documents = cache.get("tokenized_documents", [])
                cache_valid = (
                    cache.get("collection_name") == self.collection_name
                    and len(cached_ids) == len(tokenized_documents)
                    and set(cached_ids) == set(current_ids)
                    and all(isinstance(tokens, list) for tokens in tokenized_documents)
                )
                if cache_valid:
                    self.bm25_ids = cached_ids
                    self.bm25_index = BM25Okapi(tokenized_documents)
                    return
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                # 缓存损坏时直接根据 Chroma 内容重建。
                pass

        self._build_and_save_bm25_cache()

    def _build_and_save_bm25_cache(self):
        """从 Chroma 文档生成分词结果，持久化后加载到内存。"""

        assert self.chroma is not None, "chroma 未初始化"
        if self.chroma.count() == 0:
            self.bm25_index = None
            self.bm25_ids = []
            return

        data = self.chroma.get(include=["documents"])
        ids = [str(item) for item in data.get("ids", [])]
        documents = data.get("documents") or []
        tokenized_documents = [self._tokenize(document) for document in documents]
        if not ids or len(ids) != len(tokenized_documents):
            raise RuntimeError("Chroma 文档与 ID 数量不一致，无法构建 BM25 索引")

        cache = {
            "version": 1,
            "collection_name": self.collection_name,
            "ids": ids,
            "tokenized_documents": tokenized_documents,
        }
        temporary_path = self.bm25_cache_path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(cache, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary_path.replace(self.bm25_cache_path)

        self.bm25_ids = ids
        self.bm25_index = BM25Okapi(tokenized_documents)
        
    def search(self,query:str,topk:int=settings.rag_top_k):
        #在chroma数据库中进行检索 并返回检索结果
        #增加检索模式，最后实现，包括BM25，向量检索，混合检索等
        assert self.chroma is not None, "chroma 未初始化"
        if self.chroma.count() == 0:
            raise RuntimeError("Chroma collection 为空")
        query_embedding=embedding.embed_queries(query)
        results=self.chroma.query(
            query_embeddings=[query_embedding],
            n_results=min(topk, self.chroma.count()),
        )
        return results

    def sim_and_bm25_search(self,query:str,topk:int=settings.rag_top_k):
        assert self.chroma is not None, "chroma 未初始化"
        if self.chroma.count() == 0:
            raise RuntimeError("Chroma collection 为空")
        if self.bm25_index is None:
            self._load_or_build_bm25()
        if self.bm25_index is None:
            raise RuntimeError("BM25 索引未初始化")

        actual_topk = min(topk, self.chroma.count())
        #使用chroma自带的向量检索
        query_embedding=embedding.embed_queries(query)
        results=self.chroma.query(query_embeddings=[query_embedding],n_results=actual_topk)
        vector_ids = results["ids"][0]
        vector_ranks = list(range(1, len(vector_ids) + 1))


        #BM25检索
        #计算query分数
        query_tokens=self._tokenize(query)
        bm25_scores=self.bm25_index.get_scores(query_tokens)

        # 1. 配对
        pairs = list(zip(self.bm25_ids, bm25_scores))

        # 2. 按分数降序排
        pairs.sort(key=lambda x: x[1], reverse=True)

        # 3. 取前 topk，并把分数换成排名（排名从 1 开始）
        top_pairs = pairs[:actual_topk]
        bm25_results = {
            "ids": [doc_id for doc_id, _ in top_pairs],
            "ranks": list(range(1, len(top_pairs) + 1)),
        }

        #rrf：逆序排名融合
        
        k = 60
        rrf_scores = {}

        # 1. 计算向量检索的 RRF 分数
        for doc_id, rank in zip(vector_ids, vector_ranks):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)

        # 2. 计算 BM25 检索的 RRF 分数
        for doc_id, rank in zip(bm25_results["ids"], bm25_results["ranks"]):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)

        # 3. 按融合分数降序排列
        sorted_ids = sorted(
            rrf_scores,
            key=lambda doc_id: rrf_scores[doc_id],
            reverse=True
        )[:actual_topk]

        # 4. 根据 ID 从 Chroma 获取原始文档
        final_data = self.chroma.get(
            ids=sorted_ids,
            include=["documents", "metadatas"]
        )

        # 5. 建立 ID 到文档的映射，保证返回顺序与 RRF 排名一致
        doc_map = {
            doc_id: (document, metadata)
            for doc_id, document, metadata in zip(
                final_data["ids"],
                final_data["documents"],
                final_data["metadatas"]
            )
        }

        # 6. 返回与 Chroma query 相同的主要字段结构
        return {
            "ids": [sorted_ids],
            "documents": [[doc_map[i][0] for i in sorted_ids]],
            "metadatas": [[doc_map[i][1] for i in sorted_ids]],
            "distances": None,
            "rrf_scores": [[rrf_scores[i] for i in sorted_ids]]
        }

       



        


            

ecom_chroma_backend=chroma_backend()
    
#返回的可能格式
# {
#     "ids": [["chunk_001", "chunk_002", "chunk_003", ...]],
#     "documents": [["退款通常会在审核通过后的3到5个工作日到账。", "...", "..."]],
#     "metadatas": [[{"docs": "退款政策", "section": "退款时间"}, {...}, {...}]],
#     "distances": [[0.12, 0.34, 0.56, ...]],
#     "embeddings": None,   # 默认不返回
#     "uris": None,
#     "data": None,
# }


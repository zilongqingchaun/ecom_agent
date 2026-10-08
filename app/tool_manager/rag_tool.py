#rag检索工具

from app.rag.backends.chroma_db import ecom_chroma_backend


def rag_search(query: str)->dict:
    if not isinstance(query, str) or not query.strip():
        return {"success": False, "messages": "请输入查询内容。"}
    if ecom_chroma_backend.chroma is None:
        return {"success": False, "messages": "Chroma 数据库未初始化，请检查配置。"}
    if ecom_chroma_backend.chroma.count() == 0:
        #加载知识库
        ecom_chroma_backend.upsert_collection()
    search_results = ecom_chroma_backend.search(query)

    return {"success": True, "messages": search_results}
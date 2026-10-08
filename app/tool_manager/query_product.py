from app.tool_manager.mock_data import PRODUCTS

#分为两种：精确查询和模糊查询。
#前者只要输入商品的完整名称或ID，就能返回该商品的详细信息。
#后者需要关键词查询，"name"、"description"、"specs"、三个字段拼接，用关键词检索

def query_product(keyword:str)->dict:
    """按商品 ID、完整名称或关键词查询商品信息。

    ID 或完整名称精确匹配时返回单个商品；否则在名称、描述、分类和
    规格中进行不区分大小写的包含匹配，并返回所有匹配商品。
    """
    if not isinstance(keyword, str) or not keyword.strip():
        return {"success": False, "messages": "请输入商品名称、商品 ID 或搜索关键词。"}

    query = keyword.strip().casefold()

    # 先按商品 ID 和完整名称匹配，便于明确查询时优先返回单个商品。
    for product_id, product in PRODUCTS.items():
        if query in (product_id.casefold(), product["name"].casefold()):
            return {"success": True, "messages": product}

    matches = []
    for product in PRODUCTS.values():
        searchable_fields = [
            product.get("name", ""),
            product.get("description", ""),
            product.get("category", ""),
            *product.get("specs", {}).values(),
        ]
        if any(query in str(field).casefold() for field in searchable_fields):
            matches.append(product)

    if not matches:
        return {"success": False, "messages": f"未找到与“{keyword.strip()}”相关的商品。"}

    return {"success": True, "messages": matches}

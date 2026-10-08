from app.tool_manager.apply_refund import apply_refund
from app.tool_manager.list_user_orders import list_user_orders
from app.tool_manager.query_logistics import query_logistics
from app.tool_manager.query_order import query_order
from app.tool_manager.query_product import query_product
from app.tool_manager.rag_tool import rag_search

# 工具名称到实际 Python 函数的映射，供 Tool_Manager.call_tool 使用。
tool_mapping = {
    "query_order": query_order,
    "list_user_orders": list_user_orders,
    "query_product": query_product,
    "query_logistics": query_logistics,
    "apply_refund": apply_refund,
    "rag_search": rag_search,
}


# OpenAI Chat Completions 的 tools 参数格式。
tool_definition: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "query_order",
            "description": "根据订单号查询订单详情，包括状态、商品、金额和物流信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "订单编号"},
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_user_orders",
            "description": "根据用户标识查询该用户的订单概要列表。",
            "parameters": {
                "type": "object",
                "properties": {
                    "user": {"type": "string", "description": "用户标识或用户名"},
                },
                "required": ["user"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_product",
            "description": "根据商品 ID、完整名称或关键词查询商品信息，包括价格、库存和规格。",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "商品 ID、名称或搜索关键词"},
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_logistics",
            "description": "根据物流单号查询物流状态和轨迹。",
            "parameters": {
                "type": "object",
                "properties": {
                    "tracking_number": {"type": "string", "description": "物流单号"},
                },
                "required": ["tracking_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_refund",
            "description": "根据订单号和退款原因提交退款申请或查询退款处理提示。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "订单编号"},
                    "reason": {"type": "string", "description": "退款原因"},
                },
                "required": ["order_id", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "检索并夕夕的退换货政策、配送说明、会员权益和常见问题知识库。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "需要检索的政策、规则或帮助问题"},
                },
                "required": ["query"],
            },
        },
    },
]

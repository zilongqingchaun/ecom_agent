from app.tool_manager.mock_data import ORDERS


def query_order(order_id:str)->dict:
    """
    根据订单号查询订单详情
    :param order_id: 订单号
    :return: 订单详情字典，如果订单不存在则返回 None
    """
    order=ORDERS.get(order_id)
    if not order:
        return {"success": False, "messages": f"订单号 {order_id} 不存在。"}
    return {"success": True, "messages": order}
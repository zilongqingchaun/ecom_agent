#根据订单号申请退款
#必须出具订单号和退款原因

from app.tool_manager.mock_data import ORDERS
def apply_refund(order_id: str, reason: str) -> dict:
    # 尚未发货，直接退款。已经发货，回复已经上交客服，3-6个工作日给回复。没有查询到订单，直接回复没有查询到订单。已经在走退款流程，告诉用户已经在处理
    #请耐心等候
    order = ORDERS.get(order_id)
    if not order:
        return {"success": False, "messages": f"订单 {order_id} 不存在。请重新核实。"}
    if order.get("status") == "pending":
        return {"success": True, "messages": f"订单 {order_id} 尚未发货，将直接退款。"}
    if order.get("status") == "refund_processing":
        return {"success": True, "messages": f"订单 {order_id} 已在退款处理中。请耐心等候。"}

    return {"success": True, "messages": f"订单 {order_id} 的退款申请已提交，原因：{reason},客服将在3-6个工作日内回复,请耐心等待。"}
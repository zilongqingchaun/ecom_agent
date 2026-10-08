from app.tool_manager.mock_data import ORDERS

#根据用户id查询用户的所有订单概要列表,c查询该用户的所有订单，然后返回概要信息

def list_user_orders(user:str)->dict:
    user_all_orders=[]
    for order_id, order in ORDERS.items():
        if order["user"] == user:
            user_all_orders.append({
                "order_id": order["order_id"],
                "status": order["status"],
                "total": order["total"],
                "created_at": order["created_at"]
                ,"items": order["items"]
            })
    
    if not user_all_orders:
        return {"success": False, "messages": f"用户 {user} 没有订单记录。"}
    return {"success": True, "messages": user_all_orders}
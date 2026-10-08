#根据tracking_number查询物流轨迹
from app.tool_manager.mock_data import LOGISTICS




def query_logistics(tracking_number: str) -> dict:
    """
    根据物流编号查询物流轨迹
    :param tracking_number: 物流编号
    :return: 物流轨迹字典，如果订单不存在则返回 None
    """
    



    logistics_info = LOGISTICS.get(tracking_number)
    if not logistics_info:
        return {"success": False, "messages": f"物流编号 {tracking_number} 不存在或无物流信息。"}
    
    return {"success": True, "messages": logistics_info}
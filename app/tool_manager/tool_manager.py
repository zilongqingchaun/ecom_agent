## 工具使用指南
# 你现在可以调用以下工具来查询真实数据，请在需要时主动使用：
# - **query_order**：根据订单号查询订单详情（状态、商品、金额等）
# - **list_user_orders**：查询当前用户的所有订单概要列表
# - **query_product**：根据关键词搜索商品信息（价格、库存、规格等）
# - **query_logistics**：根据订单号查询物流轨迹
# - **apply_refund**：为指定订单申请退款（需要订单号和退款原因）
# - **search_knowledge**：检索并夕夕的政策与帮助文档（退换货政策、配送说明、会员权益、常见问题FAQ）
# - **recall_user_memory**：查询当前用户的记忆信息（偏好、历史问题、会员信息等）
# - **load_skill**：加载指定技能的详细操作指令（技能列表见下方「可用技能」部分）

#先简单实现query_order、list_user_orders、query_product、query_logistics、apply_refund。
#search_knowledge、recall_user_memory可以先不实现，后续再补充。

from app.tool_manager.registry import tool_definition,tool_mapping
from app.settings.settings import settings


#做一个函数名字到函数的映射



#负责初始化工具管理器，加载工具，并提供调用接口
class Tool_Manager:
    def __init__(self):
        # 初始化工具管理器
        self.tools =[] 
        #离线和在线工具
        if settings.enable_mcp:
            # 如果启用了MCP模式，加载在线工具
            #d但暂时不实现mcp
            pass
        else:
            # 否则加载离线工具
            self.tools.extend(tool_definition)
        

    def load_tools(self):
        # 返回符合 OpenAI Chat Completions tools 参数格式的函数定义。
        return self.tools

    def call_tool(self, tool_name,**kwargs):
        # 调用指定的工具并返回结果
        call_tool=tool_mapping.get(tool_name) or None
        if not call_tool:
            return f"工具 {tool_name} 不存在。"
        return call_tool(**kwargs)
        



tool_manager = Tool_Manager()

(ecom_agent) PS D:\code\project\my_agent\my_agent_ecom> python -m app.evaluation.run_eval --output evaluation_report.json

评估结果
- order_query_basic: 通过 | 工具=100% | 意图=100% | 完整性=100% | tokens=6569
- logistics_track: 通过 | 工具=100% | 意图=100% | 完整性=100% | tokens=5604
- logistics_no_tracking: 异常 — ValidationError: 3 validation errors for CustomerServiceResponse
intent
  Field required [type=missing, input_value={}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing
confidence
  Field required [type=missing, input_value={}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing
reply
  Field required [type=missing, input_value={}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing
- product_out_of_stock: 未通过 | 工具=100% | 意图=100% | 完整性=100% | tokens=8208
- return_request: 未通过 | 工具=- | 意图=0% | 完整性=100% | tokens=6855
- knowledge_policy: 通过 | 工具=100% | 意图=100% | 完整性=100% | tokens=7614
- member_benefits: 未通过 | 工具=100% | 意图=100% | 完整性=100% | tokens=6877
- complaint_to_human: 未通过 | 工具=- | 意图=100% | 完整性=- | tokens=2898
- greeting_no_tool: 通过 | 工具=- | 意图=100% | 完整性=- | tokens=2590
- multi_turn_followup: 通过 | 工具=100% | 意图=100% | 完整性=100% | tokens=8905
- list_orders: 通过 | 工具=100% | 意图=100% | 完整性=100% | tokens=6786

通过率：6/11 (55%)
总 token：62906
报告已写入：D:\code\project\my_agent\my_agent_ecom\evaluation_report.json
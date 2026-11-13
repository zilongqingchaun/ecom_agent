# E-commerce Customer Service Agent

一个面向电商客服场景的 Stateful Tool-Using Agent，支持 ReAct 工具调用、会话持久化、长短期记忆、RAG 检索以及独立的 Agent / RAG Evaluation。

项目覆盖政策咨询、订单查询、物流查询、商品咨询与退款等典型电商客服场景，并实现 Dense Retrieval 与 BM25 + RRF 混合检索的对比评测。

## Features

- 基于 DeepSeek API + OpenAI SDK 实现 ReAct 式 Agent Loop
- 支持订单、物流、商品、退款、RAG 等业务工具调用
- 使用 Pydantic 对最终客服响应进行结构化约束
- 支持人工介入判断
- JSONL 增量保存与恢复多轮会话
- STM / LTM 长短期用户记忆
- ChromaDB Dense Retrieval
- BM25 + Reciprocal Rank Fusion（RRF）混合检索
- 独立 Agent Evaluation
- 独立 RAG Retrieval / Generation Evaluation
- 基于京东、拼多多公开政策整理的小型 RAG Benchmark

## Architecture

```text
                        User
                          │
                          ▼
                  E-commerce Agent
                          │
                    ReAct Loop
                          │
             ┌────────────┼────────────┐
             │            │            │
             ▼            ▼            ▼
        Business Tools    RAG        Memory
             │            │            │
        ┌────┼────┐       │       ┌────┴────┐
        │    │    │       │       │         │
      Order 物流 商品     │      STM       LTM
        │                 │
      Refund              │
                          ▼
                  Retrieval Pipeline
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
       Dense Retrieval             BM25
        (ChromaDB)                    │
              └───────────┬───────────┘
                          ▼
                         RRF
                          │
                          ▼
                        Top-K
                          │
                          ▼
                         LLM
                          │
                          ▼
                    Final Response
```

## Agent Runtime

Agent 使用 Chat Completions Tool Calling 实现 ReAct 式循环。

基本执行流程：

```text
User Query
    │
    ▼
Build Messages
    │
    ▼
LLM Reasoning
    │
    ├── No Tool Call ───────► Final Response
    │
    └── Tool Call
            │
            ▼
       Tool Manager
            │
            ▼
       Tool Result
            │
            └──────────────► Next ReAct Step
```

当前业务工具包括：

- `query_order`：查询订单信息
- `list_user_orders`：查询用户订单列表
- `query_product`：查询商品信息
- `query_logistics`：查询物流信息
- `apply_refund`：申请退款
- `rag_search`：检索客服知识库

最终回复通过 Pydantic 进行结构化约束，包括：

```text
intent
confidence
reply
requires_human
follow_up_question
```

## Memory & Session

项目实现了独立的 `MemoryManager` 管理用户长短期信息。

### Short-Term Memory

从当前用户消息中提取：

- 用户身份信息
- 商品偏好
- 关注商品
- 用户情绪
- 特殊需求

### Long-Term Memory

会话结束时从本次交互中提取值得跨会话保留的用户事实，并持久化存储。

下一次请求中：

```text
STM
 +
LTM
 ↓
Memory Prompt
 ↓
System Prompt
 ↓
Agent
```

### Session Persistence

对话历史使用 JSONL 增量写入：

```text
sessions/
└── <user_id>.jsonl
```

启动 Agent 时可以恢复已有历史记录，避免每次覆盖完整会话文件。

## RAG

项目包含两套不同用途的知识数据。

### Runtime Knowledge Base

`app/rag/knowledge/`

包含用于客服 Agent 功能验证的模拟电商政策知识，例如：

- 会员权益
- 常见问题 FAQ
- 退换货政策
- 配送说明

这部分主要用于 Agent Runtime 的功能测试。

### Official E-commerce Benchmark

`data/official_ecom/`

基于京东、拼多多公开页面整理独立的政策摘要型知识集，用于检索实验和 RAG Evaluation。

该数据集不会自动替换 Agent Runtime 中的模拟知识库。

## Retrieval Pipeline

基础检索：

```text
Query
  │
  ▼
Embedding
  │
  ▼
ChromaDB
  │
  ▼
Dense Top-K
```

同时实现了 BM25 + RRF Hybrid Retrieval：

```text
                Query
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
 Dense Retrieval           BM25
        │                   │
        └─────────┬─────────┘
                  ▼
        Reciprocal Rank Fusion
                  │
                  ▼
                Top-K
```

RRF：

```text
RRF(d) = Σ 1 / (k + rank(d))
```

当前实现使用：

```text
k = 60
```

RRF 仅基于不同 Retriever 的排名进行融合，不直接混合 Dense Distance 与 BM25 Score。

## RAG Benchmark

`data/official_ecom/` 当前包含：

| Platform | Documents | Chunks | Candidate Queries |
|---|---:|---:|---:|
| Pinduoduo | 3 | 9 | 24 |
| JD | 9 | 34 | 76 |
| Total | **12** | **43** | **100** |

100 个候选问题包括：

- 86 个单规则不同问法
- 6 个跨片段问题
- 8 个拒答 / 澄清 / 工具依赖 / 平台边界类问题

其中 **92 个具有明确 evidence label 的问题**用于 Retrieval Evaluation。

需要注意：

- 候选问题为自动构造的评估数据
- 评测样本经过人工一致性检查，主要核对问题、证据片段与参考答案之间的对应关系；
- 同一规则的不同问法不能视为不同知识
- 当前规模属于小型 benchmark，不代表真实线上客服分布
- 不使用该结果宣称真实客服正确率

## Retrieval Evaluation

检索指标包括：

```text
Hit@K
Recall@K
Precision@K
MRR
```

在相同 92 个 evidence-labeled cases 上，对比：

```text
Dense Retrieval
vs
Dense + BM25 + RRF
```

当前 Recall@3：

| Retrieval | Recall@3 |
|---|---:|
| Dense / Chroma HNSW | **99.46%** |
| Dense + BM25 + RRF | **98.37%** |

两种方案在当前小规模知识库中均具有较高召回率。

Hybrid Retrieval 并未在所有问题上优于 Dense Retrieval，这也是本项目进行 Bad Case Analysis 的重点。

## Bad Case Analysis

例如：

```text
Query:
京东一张东券买了几件，只退一件就能拿回整张券吗？
```

Gold Evidence：

```text
jd_coupon.return
```

Dense Retrieval：

```text
Rank 1  jd_coupon.cancel
Rank 2  jd_coupon.return      ← Gold
Rank 3  jd_coupon.freight
```

Hybrid Retrieval：

```text
Rank 1  jd_coupon.freight
Rank 2  jd_coupon.cancel
Rank 3  jd_coupon.scope
Rank 4  jd_coupon.return      ← Gold
```

在该样本中，BM25 对“券、退货”等高词面相关规则赋予较高排名，经 RRF 融合后，正确证据由第 2 位下降至第 4 位。

该现象说明：

```text
Dense Retrieval
更擅长整体语义匹配

BM25
更强调词面匹配

Hybrid Retrieval
并不天然优于单一 Retriever
```

在规模较小且政策条款高度相似的知识库中，Sparse Retrieval 可能引入额外的相似条款干扰。

## Evaluation

项目包含两层 Evaluation。

### Agent Evaluation

```text
Eval Case
   │
   ▼
Fresh Session
   │
   ▼
Agent
   │
   ├── LLM Calls
   ├── Tool Calls
   └── Final Response
          │
          ▼
       Evaluation
```

评测过程中：

- 每个 case 使用独立 session
- 禁用用户 Memory
- 禁用 MCP
- 记录 LLM 调用
- 记录 Tool 调用
- 统计 Token
- 避免不同 case 之间的历史污染

Agent 指标包括：

```text
Tool Accuracy
Tool Efficiency
Intent Match
Keyword Coverage
Human Handoff
Faithfulness
Answer Quality
Process Soundness
Token Cost
```

### RAG Evaluation

```text
Query
  │
  ▼
Retriever
  │
  ▼
Top-K Chunks
  │
  ├── Retrieval Metrics
  │
  └── RAG Answer
         │
         ▼
      LLM Judge
```

生成侧指标包括：

```text
Faithfulness
Answer Relevance
Answer Correctness
Context Relevance
```

## Project Structure

```text
ecom_agent/
│
├── app/
│   ├── agent/
│   │   └── ecom_single_agent.py
│   │
│   ├── evaluation/
│   │   ├── sandbox.py
│   │   ├── evaluator.py
│   │   ├── metrics.py
│   │   ├── trace.py
│   │   └── cases.json
│   │
│   ├── memory_manager/
│   │   ├── memory_manager.py
│   │   ├── short_term_memory.py
│   │   └── long_term_memory.py
│   │
│   ├── prompt/
│   │
│   ├── rag/
│   │   ├── backends/
│   │   │   └── chroma_db.py
│   │   ├── knowledge/
│   │   ├── chunk.py
│   │   └── embedding.py
│   │
│   ├── rag_eval/
│   │   ├── rag_eval.py
│   │   ├── metric.py
│   │   └── dataset.py
│   │
│   ├── settings/
│   │
│   └── tool_manager/
│
├── data/
│   └── official_ecom/
│       ├── knowledge/
│       │   ├── jd/
│       │   └── pdd/
│       ├── exports/
│       ├── prepare.py
│       └── run_baseline.py
│
├── .env.example
├── main.py
└── README.md
```

## Installation

```bash
git clone https://github.com/zilongqingchaun/ecom_agent.git
cd ecom_agent
```

本项目使用 Python 3.10 验证。安装依赖后，在根目录创建 `.env`。

```bash
python -m pip install -r requirements.txt
```

可以参考：

```text
.env.example
```

配置：

```text
MODEL_API_KEY
EMBEDDING_MODEL_API_KEY
```

实际字段名称以 `app/settings/settings.py` 为准。

## Run Agent

```bash
python main.py
```

## Prepare Official Benchmark

生成 / 校验独立政策数据：

```bash
python data/official_ecom/prepare.py
```

该步骤：

- 不访问网络
- 不调用 LLM
- 不调用 Embedding API
- 验证 Chunk ID / Case ID 唯一性
- 验证 Evidence 是否存在
- 验证平台一致性
- 导出 Dev / Test 数据

## Build Benchmark Index

以京东数据为例：

```bash
python data/official_ecom/prepare.py --index jd
```

该步骤会调用 Embedding API。

拼多多：

```bash
python data/official_ecom/prepare.py --index pdd
```

两个平台分别建立独立 Collection，避免不同平台政策互相污染。

## Run RAG Evaluation

加载对应的独立索引配置后，可执行：

```bash
python -m app.rag_eval.rag_eval \
  --dataset data/official_ecom/exports/jd/rag_test.json \
  --retrieval-only \
  --search-mode CHROMA_HNSW
```

混合检索：

```bash
python -m app.rag_eval.rag_eval \
  --dataset data/official_ecom/exports/jd/rag_test.json \
  --retrieval-only \
  --search-mode CHROMA_BM25
```

> `CHROMA_BM25` 为历史命名，实际表示 Dense Retrieval + BM25 + RRF，而非纯 BM25。

## Data Source Notes

`data/official_ecom` 中的政策资料：

- 来源于公开的京东、拼多多页面
- 保留 `source_url`
- 内容为面向 RAG 的摘要型整理，而非官方原文镜像
- 不使用第三方论坛或博客作为核心政策依据
- 页面无法完整读取时，会在数据元信息中记录访问方式和限制

政策可能随平台更新而变化，因此该数据主要用于工程实验与检索评测，不代表平台当前完整规则。


项目重点是验证：

```text
Agent Runtime
+
Tool Calling
+
Memory
+
RAG
+
Retrieval Evaluation
+
Agent Evaluation
```

这一整套 Agent 工程链路。

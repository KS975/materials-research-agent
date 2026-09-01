# LangGraph Migration V2

## 目标

在 V1 生产图外壳的基础上加入真实条件边，同时保持当前业务执行主体、
权限边界、接口字段、答案和 SSE 协议不变。

## V2 图结构

```text
START
  -> receive_request
  -> classify_primary
       -> direct_attachment
       -> deterministic
       -> semantic
            -> classify_semantic
                 -> database_explorer
                 -> rag
                 -> current_attachment
                 -> general_conversation
                 -> material_tool
  -> validate_response
  -> END
```

## 分流原则

### 执行前一级分流

- “附件直问”开关开启：`direct_attachment`
- Round 2A/2B、单位真实数据、V0.2/V0.3、T17/T18：`deterministic`
- 其余需要 DeepSeek 意图识别的问题：`semantic`

一级分流只调用确定性、无副作用的分类器，不查询数据库、不调用模型。

### 语义执行后证据分流

语义请求只执行一次现有生产路由和一次 DeepSeek 路由。V2 根据实际返回的
`intent/router` 将结果进入 Database Explorer、RAG、当前附件、通用问答或
材料 Tool 分支，不会为了决定图路径再次调用 DeepSeek。

## 为什么仍保留受保护执行核心

当前生产路由包含已验证的权限、异常转换和能力优先级。V2 先建立稳定的条件
边和状态字段，不在同一版本同时重写所有业务处理器。后续版本可以沿这些节点
逐个提升为独立执行节点，并对 Database Explorer 加入图级重试状态。

## 变更文件

- 修改 `api/chat_ui.py`
- 修改 `orchestration/chat_ui_graph.py`
- 修改 `orchestration/chat_ui_state.py`
- 新增 `tests/unit/test_chat_ui_langgraph_v2.py`
- 新增本说明和 V2 验证报告

## 未改变

- 前端与 dist
- `.env`、Dockerfile、requirements.txt
- `/api/v1/chat-ui` 和 `/api/v1/chat-ui/stream` 协议
- 公司/项目权限
- DeepSeek 调用次数
- 原有高精度意图优先级

## 单位服务器部署

覆盖后端增量包中的同路径文件，然后执行：

```bash
docker build --progress=plain -t materials-agent:1.0 .
docker-compose up -d --no-deps --force-recreate mat-agent
docker-compose restart nginx
```

健康检查：

```bash
curl -i http://127.0.0.1:18000/health
curl -i http://127.0.0.1:81/agent-api/health
```

本版没有前端改动，不需要重新部署 dist。

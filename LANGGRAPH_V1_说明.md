# LangGraph Migration V1

## 目标

让网页实际使用的 `/api/v1/chat-ui` 与 `/api/v1/chat-ui/stream` 经过
LangGraph 执行，同时保持当前意图优先级、权限、回答和 SSE 协议不变。

## V1 工作流

```text
START
  -> receive_request
  -> legacy_dispatch
  -> validate_response
  -> END
```

`legacy_dispatch` 调用迁移前已验证的生产路由。V1 不拆分内部业务分支，
避免附件直问、业务 MySQL、Database Explorer、RAG、T17/T18 等能力发生
路由顺序漂移。

## 变更文件

- 新增 `schemas/chat_ui.py`
- 新增 `orchestration/chat_ui_state.py`
- 新增 `orchestration/chat_ui_graph.py`
- 修改 `api/chat_ui.py`
- 新增 `tests/unit/test_chat_ui_langgraph_v1.py`
- 补充 `tests/unit/test_attachment_reference_mode.py`

## 未改变

- 前端源码和 dist
- `.env` 与 Dockerfile
- `/api/v1/chat-ui` 请求与响应字段
- `/api/v1/chat-ui/stream` SSE 事件格式
- 公司和项目权限边界
- 现有高精度意图优先级
- 旧 `/api/v1/chat` 使用的 `orchestration/graph.py`

## 部署

单位服务器只需覆盖后端增量包中的同路径文件，然后重新构建并重启：

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

V1 没有前端改动，不需要替换前端 dist。

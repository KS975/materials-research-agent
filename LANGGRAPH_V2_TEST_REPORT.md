# LangGraph Migration V2 验证报告

## 已完成验证

- 全项目 316 个 Python 源文件 AST 语法扫描通过，0 个错误。
- 三个一级分支均通过执行链烟测：
  - `direct_attachment`
  - `deterministic`
  - `semantic`
- 五个语义证据分支均通过条件边烟测：
  - `database_explorer`
  - `rag`
  - `current_attachment`
  - `general_conversation`
  - `material_tool`
- 每个请求的受保护生产执行器只调用一次。
- 30 个并发请求完成状态隔离烟测。
- 未知一级分支会在执行前停止，生产执行器不会被调用。
- 执行器异常保持原类型和原消息向外透传。
- 语义证据分类器不调用 `.route()` 或 `.complete()`，不会产生第二次模型调用。
- 迁移前后 `HistoryMessage`、`ChatUIRequest`、`ChatUIResponse` 字段 AST 一致。
- 迁移前生产 `chat_ui()` 与当前 `_execute_chat_ui_legacy()` 函数体 AST 一致。
- `/chat-ui` 和 `/chat-ui/stream` 均进入同一个 V2 图入口。

## 新增自动化测试

`tests/unit/test_chat_ui_langgraph_v2.py` 覆盖：

- 附件直问最高优先级
- 相似样品等高精度 MySQL 能力进入确定性分支
- 未命中问题进入语义分支
- 实际响应到五类语义证据分支的映射
- 三类一级分支单次执行
- 未知分支拒绝执行

V1 与附件直问测试继续保留，用于累计回归。

## 环境说明

当前构建工作区未安装项目锁定的 FastAPI、LangGraph 和 pytest，因此未在
该工作区执行完整 pytest。图节点和条件边通过等价内存图执行器完成契约烟测；
自动化测试已随本地包交付，可在安装 `requirements.txt` 的项目环境执行：

```bash
python -m pytest -q \
  tests/unit/test_chat_ui_langgraph_v1.py \
  tests/unit/test_chat_ui_langgraph_v2.py \
  tests/unit/test_attachment_reference_mode.py \
  tests/unit/test_chat_ui_stream_progress_v212.py \
  tests/unit/test_round2a2_1_chat_ui_mysql_precedence.py
```

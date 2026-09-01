# LangGraph Migration V1 验证报告

## 结果

- 新增/修改的 6 个 Python 文件均通过 AST 语法解析。
- 项目内 315 个 Python 源文件完成全量 AST 语法扫描，0 个错误。
- V1 三节点工作流完成调用链烟测。
- 12 个并发请求完成状态隔离烟测。
- 执行节点异常保持原类型和原消息向外透传。
- `/chat-ui/stream` 仍调用公共 `chat_ui()` 图入口。
- 附件直问仍位于生产路由主体的最高优先级。
- 迁移前 `chat_ui()` 函数体与迁移后 `_execute_chat_ui_legacy()`
  函数体 AST 完全一致，业务分支没有被改写。

## 新增自动化测试

- `tests/unit/test_chat_ui_langgraph_v1.py`
  - 执行器只调用一次并保持响应
  - HTTPException 状态码与消息透传
  - 并发请求状态隔离
  - 公共 Chat UI 入口调用已编译生产图
- `tests/unit/test_attachment_reference_mode.py`
  - 附件正文完整提交给 DeepSeek
  - 附件直问绕过 T17/数据库业务路由
  - 开启模式但未上传附件时返回 400

## 环境说明

当前构建工作区未安装项目锁定的 FastAPI、LangGraph 和 pytest，因此未在
该工作区执行完整 pytest。工作流节点通过等价的内存图执行器完成契约烟测；
自动化测试已随本地增量包交付，可在安装 `requirements.txt` 的项目环境运行：

```bash
python -m pytest -q \
  tests/unit/test_chat_ui_langgraph_v1.py \
  tests/unit/test_attachment_reference_mode.py \
  tests/unit/test_chat_ui_stream_progress_v212.py \
  tests/unit/test_round2a2_1_chat_ui_mysql_precedence.py
```
